"""Durable training runs on a local POSIX filesystem.

A kernel-held flock excludes cooperating writers. Atomic replacement and fsync
make checkpoint pointers and the final completion marker durable; this is not a
distributed lock or a guarantee for object stores or network filesystems.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid

from .models import MLRuntimeError


_FILES = ("weights.safetensors", "training.json", "bundle.json")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_JSON_LIMIT = 16 * 1024 * 1024


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    """Reject coercions such as non-string keys, tuples, NaNs and NumPy types."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise MLRuntimeError("run metadata must contain only finite JSON values with string object keys")


def _encoded(document):
    try:
        _json_value(document)
        data = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > _JSON_LIMIT:
            raise MLRuntimeError("run metadata exceeds the 16 MiB size limit")
        return data
    except (ValueError, TypeError, RecursionError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"cannot encode run metadata: {error}") from error


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MLRuntimeError(f"run metadata contains duplicate field {key!r}")
        result[key] = value
    return result


def _regular_fd(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise MLRuntimeError(f"{path}: expected a regular file")
    return descriptor


def _read_json(path):
    try:
        with os.fdopen(_regular_fd(path), "rb") as stream:
            if os.fstat(stream.fileno()).st_size > _JSON_LIMIT:
                raise MLRuntimeError(f"{path}: run metadata exceeds the 16 MiB size limit")
            data = stream.read(_JSON_LIMIT + 1)
        if len(data) > _JSON_LIMIT:
            raise MLRuntimeError(f"{path}: run metadata exceeds the 16 MiB size limit")
        document = json.loads(data, object_pairs_hook=_pairs)
        _json_value(document)
        return document
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"{path}: cannot read run metadata: {error}") from error


def _fields(document, expected, location):
    if type(document) is not dict or set(document) != set(expected):
        raise MLRuntimeError(f"{location}: expected exactly these fields: {', '.join(sorted(expected))}")
    return document


def _identifier(value, location):
    if type(value) is not str or not _ID.fullmatch(value):
        raise MLRuntimeError(f"{location}: expected a 32-character lowercase UUID")


def _timestamp(value, location):
    try:
        if type(value) is not str or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("missing timezone")
    except ValueError as error:
        raise MLRuntimeError(f"{location}: expected an ISO timestamp with timezone") from error


def _root(path):
    candidate = Path(path).expanduser()
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise MLRuntimeError(f"{path}: expected a run directory, not a file or symlink")
    return candidate.resolve()


def _contained(root, relative):
    if not isinstance(relative, (str, Path)):
        raise MLRuntimeError("run path must be a relative path")
    text = str(relative)
    parts = text.split("/")
    if not text or "\\" in text or any(part in ("", ".", "..") for part in parts):
        raise MLRuntimeError(f"{relative}: path must remain inside the run directory")
    result = root
    for index, part in enumerate(parts):
        result = result / part
        if os.path.lexists(result):
            mode = result.lstat().st_mode
            if stat.S_ISLNK(mode) or (index < len(parts) - 1 and not stat.S_ISDIR(mode)):
                raise MLRuntimeError(f"{result}: run paths must not contain symlinks or non-directory parents")
    return result


def fsync_directory(path):
    """Flush directory entries after replacing a file or committing a directory."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, document):
    """Replace one JSON file with a flushed, same-directory temporary file."""
    path = Path(path)
    data = _encoded(document)
    if os.path.lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise MLRuntimeError(f"{path}: refusing to replace a symlink or non-regular file")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _run_document(root):
    document = _fields(_read_json(_contained(root, "run.json")),
                       {"schema_version", "kind", "run_id", "spec"}, "run.json")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise MLRuntimeError("run.json: unsupported schema_version")
    if document["kind"] != "rptu-training-run":
        raise MLRuntimeError("run.json: expected a training run")
    _identifier(document["run_id"], "run.json run_id")
    if type(document["spec"]) is not dict:
        raise MLRuntimeError("run.json spec: expected an object")
    return document


def read_run_spec(output):
    """Read the immutable specification without locking or modifying a run.

    Callers must compare it again with ``bind_spec`` under the writer lock.
    """
    try:
        return _run_document(_root(output))["spec"]
    except OSError as error:
        raise MLRuntimeError(f"{output}: cannot read training run: {error}") from error


def _sha256(path):
    with os.fdopen(_regular_fd(path), "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _attempt_document(root, identifier, run_id):
    _identifier(identifier, "attempt ID")
    document = _fields(_read_json(_contained(root, f"attempts/{identifier}/attempt.json")),
                       {"schema_version", "kind", "run_id", "attempt_id", "status",
                        "started_at", "finished_at", "error"}, "attempt.json")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise MLRuntimeError("attempt.json: unsupported schema_version")
    if document["kind"] != "rptu-training-attempt" or document["run_id"] != run_id or document["attempt_id"] != identifier:
        raise MLRuntimeError("attempt.json: attempt identity does not match the training run")
    if document["status"] not in ("running", "failed", "interrupted", "completed"):
        raise MLRuntimeError("attempt.json: invalid status")
    _timestamp(document["started_at"], "attempt started_at")
    if document["status"] == "running":
        if document["finished_at"] is not None or document["error"] is not None:
            raise MLRuntimeError("attempt.json: running attempt must not have an end time or error")
    else:
        _timestamp(document["finished_at"], "attempt finished_at")
        if document["status"] == "completed":
            if document["error"] is not None:
                raise MLRuntimeError("attempt.json: completed attempt must not have an error")
        else:
            error = _fields(document["error"], {"type", "message"}, "attempt error")
            if any(type(error[key]) is not str for key in error):
                raise MLRuntimeError("attempt.json: error fields must be strings")
    return document


def read_completion(output):
    """Verify a completed run's marker and every published artifact, read-only."""
    try:
        root = _root(output)
        run = _run_document(root)
        path = _contained(root, "complete.json")
        if not os.path.lexists(path):
            return None
        document = _fields(_read_json(path),
                           {"schema_version", "kind", "run_id", "attempt_id", "files", "summary", "completed_at"},
                           "complete.json")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise MLRuntimeError("complete.json: unsupported schema_version")
        if document["kind"] != "rptu-training-completion" or document["run_id"] != run["run_id"]:
            raise MLRuntimeError("complete.json: identity does not match the training run")
        _identifier(document["attempt_id"], "complete.json attempt_id")
        _timestamp(document["completed_at"], "complete.json completed_at")
        if type(document["summary"]) is not dict:
            raise MLRuntimeError("complete.json summary: expected an object")
        files = _fields(document["files"], set(_FILES), "complete.json files")
        for name, checksum in files.items():
            if type(checksum) is not str or not _HASH.fullmatch(checksum):
                raise MLRuntimeError(f"complete.json files.{name}: expected a lowercase SHA-256 checksum")
            if _sha256(_contained(root, name)) != checksum:
                raise MLRuntimeError(f"{name}: completed artifact SHA-256 checksum mismatch")
        attempt = _attempt_document(root, document["attempt_id"], run["run_id"])
        # A process can die after writing complete.json but before finalizing its
        # attempt. The completion marker wins, and read-only retries stay read-only.
        if attempt["status"] not in ("running", "completed"):
            raise MLRuntimeError("complete.json: its attempt is marked failed or interrupted")
        return document
    except OSError as error:
        raise MLRuntimeError(f"{output}: cannot verify training completion: {error}") from error


class TrainingRun:
    """Reserve or resume one training directory while holding its writer lock."""

    def __init__(self, output, *, resume=False):
        self.path = Path(output).expanduser()
        self.resume = resume
        self.run_id = None
        self.spec = None
        self._lock = None
        self._identity = None
        self._bound = False
        self._attempt = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError as error:
            raise MLRuntimeError("Durable training runs require a local POSIX filesystem and flock") from error
        try:
            if self.resume:
                self.path = _root(self.path)
                _run_document(self.path)  # No lock file is created in unrelated directories.
            else:
                if os.path.lexists(self.path):
                    raise MLRuntimeError(f"{self.path}: output already exists; choose a new path or use --resume")
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.mkdir(exist_ok=False)
                self.path = self.path.resolve()
                fsync_directory(self.path.parent)
            root_stat = self.path.stat()
            self._identity = (root_stat.st_dev, root_stat.st_ino)
            lock_path = _contained(self.path, ".run.lock")
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            # A legitimate run creates this inode before publishing run.json.
            # Recreating an externally removed lock could split writer ownership.
            self._lock = os.open(lock_path, flags if self.resume else flags | os.O_CREAT, 0o600)
            info = os.fstat(self._lock)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise MLRuntimeError(f"{lock_path}: expected a regular lock file with one link")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise MLRuntimeError(f"{self.path}: another process is already writing this training run") from error
                raise
            self._ensure_locked()
            return self
        except BaseException:
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None
            raise

    def __exit__(self, *exception):
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def _ensure_locked(self):
        if self._lock is None:
            raise MLRuntimeError("training run must be used while holding its lock")
        info = self.path.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._identity:
            raise MLRuntimeError("training run directory was replaced while its lock was held")
        path_info = (self.path / ".run.lock").lstat()
        lock_info = os.fstat(self._lock)
        if not stat.S_ISREG(path_info.st_mode) or path_info.st_nlink != 1 or (path_info.st_dev, path_info.st_ino) != (lock_info.st_dev, lock_info.st_ino):
            raise MLRuntimeError("training run lock file was replaced while its lock was held")

    def bind_spec(self, spec):
        """Initialize or compare an immutable spec before recording an attempt."""
        self._ensure_locked()
        if self._bound:
            raise MLRuntimeError("training run specification was already bound")
        if type(spec) is not dict:
            raise MLRuntimeError("training run specification must be an object")
        encoded = _encoded(spec)
        if self.resume:
            document = _run_document(self.path)
            if encoded != _encoded(document["spec"]):
                raise MLRuntimeError("resume settings do not match the immutable training run specification")
        else:
            document = {"schema_version": 1, "kind": "rptu-training-run", "run_id": uuid.uuid4().hex, "spec": spec}
            _contained(self.path, "attempts").mkdir()
            fsync_directory(self.path)
            atomic_json(_contained(self.path, "run.json"), document)
        self.run_id = document["run_id"]
        # Do not retain a caller-owned mutable dictionary as the authoritative spec.
        self.spec = json.loads(encoded)
        self._bound = True
        read_completion(self.path)

    def _recover_attempts(self):
        # The caller validates checkpoint contents before starting an attempt.
        # Defer all orphan bookkeeping until then so a rejected resume is read-only.
        directory = _contained(self.path, "attempts")
        if not os.path.lexists(directory):
            # Recover an initialization interrupted after run.json was committed.
            directory.mkdir()
            fsync_directory(self.path)
        elif not directory.is_dir():
            raise MLRuntimeError("training run is missing its attempts directory")
        # Validate every attempt before changing an abandoned one.
        attempts = [_attempt_document(self.path, child.name, self.run_id) for child in sorted(directory.iterdir())]
        for attempt in attempts:
            if attempt["status"] == "running":
                attempt.update(status="interrupted", finished_at=_now(),
                               error={"type": "ProcessInterrupted", "message": "The previous writer exited before completing this attempt."})
                self.atomic_json(f"attempts/{attempt['attempt_id']}/attempt.json", attempt)

    def atomic_json(self, relative, document):
        self._ensure_locked()
        if str(relative) in ("run.json", ".run.lock"):
            raise MLRuntimeError("training run identity and lock files are immutable")
        atomic_json(_contained(self.path, relative), document)

    def completed(self):
        self._ensure_locked()
        if not self._bound:
            raise MLRuntimeError("bind the training run specification before checking completion")
        return read_completion(self.path)

    @contextmanager
    def attempt(self):
        self._ensure_locked()
        if not self._bound or self._attempt is not None:
            raise MLRuntimeError("bind the training run specification and use one attempt at a time")
        if self.completed() is not None:
            raise MLRuntimeError("training run is already complete")
        self._recover_attempts()
        identifier = uuid.uuid4().hex
        directory = _contained(self.path, f"attempts/{identifier}")
        staging = _contained(self.path, f".attempt-{identifier}")
        staging.mkdir()
        fsync_directory(self.path)
        attempt = {"schema_version": 1, "kind": "rptu-training-attempt", "run_id": self.run_id,
                   "attempt_id": identifier, "status": "running", "started_at": _now(),
                   "finished_at": None, "error": None}
        # The record and its directory become visible together. A hard kill while
        # initializing only leaves an ignored .attempt-* staging directory.
        atomic_json(staging / "attempt.json", attempt)
        os.replace(staging, directory)
        fsync_directory(directory.parent)
        fsync_directory(self.path)
        self._attempt = identifier
        try:
            yield identifier
        except BaseException as error:
            # Do not downgrade a durably committed run if a later operation fails.
            try:
                complete = self.completed()
                if complete is not None and complete["attempt_id"] == identifier:
                    attempt.update(status="completed", finished_at=_now())
                else:
                    attempt.update(status="failed" if isinstance(error, Exception) else "interrupted",
                                   finished_at=_now(), error={"type": type(error).__name__, "message": str(error)[:2000]})
                self.atomic_json(f"attempts/{identifier}/attempt.json", attempt)
            except BaseException as record_error:
                error.add_note(f"Could not finalize training attempt: {record_error}")
            raise
        else:
            complete = self.completed()
            if complete is not None and complete["attempt_id"] == identifier:
                attempt.update(status="completed", finished_at=_now())
            else:
                attempt.update(status="interrupted", finished_at=_now(),
                               error={"type": "IncompleteAttempt", "message": "Attempt ended without publishing a completion marker."})
            self.atomic_json(f"attempts/{identifier}/attempt.json", attempt)
        finally:
            self._attempt = None

    def publish(self, staging_dir, summary):
        """Publish a fixed model bundle, then commit its completion marker last.

        Staging remains intact on failure, so an interrupted attempt can retry.
        Readers must require complete.json for a directory containing run.json.
        """
        self._ensure_locked()
        if not self._bound or self._attempt is None:
            raise MLRuntimeError("publishing requires an active training attempt")
        complete = self.completed()
        if complete is not None:
            return complete
        if type(summary) is not dict:
            raise MLRuntimeError("completion summary must be an object")
        _encoded(summary)
        source = Path(staging_dir)
        try:
            relative = source.relative_to(self.path)
        except ValueError as error:
            raise MLRuntimeError("staged model files must remain inside the training run") from error
        source = _contained(self.path, relative)
        if source == self.path or not source.is_dir():
            raise MLRuntimeError("staged model files must be in a separate run subdirectory")
        if {child.name for child in source.iterdir()} != set(_FILES):
            raise MLRuntimeError("staged model must contain exactly weights.safetensors, training.json and bundle.json")
        checksums = {}
        temporary = {}
        try:
            for name in _FILES:
                original = _contained(self.path, relative / name)
                checksums[name] = _sha256(original)
                destination = _contained(self.path, name)
                if os.path.lexists(destination) and not stat.S_ISREG(destination.lstat().st_mode):
                    raise MLRuntimeError(f"{destination}: expected a regular output file")
                descriptor, filename = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.path)
                temporary[name] = filename
                with os.fdopen(descriptor, "wb") as stream, os.fdopen(_regular_fd(original), "rb") as origin:
                    shutil.copyfileobj(origin, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _sha256(filename) != checksums[name]:
                    raise MLRuntimeError(f"{original}: staged artifact changed during publication")
            # Commit metadata last so even incomplete readers never see new bundle
            # metadata before its weights and training report are available.
            for name in _FILES:
                self._ensure_locked()
                os.replace(temporary[name], _contained(self.path, name))
                del temporary[name]
                fsync_directory(self.path)
            complete = {"schema_version": 1, "kind": "rptu-training-completion", "run_id": self.run_id,
                        "attempt_id": self._attempt, "files": checksums, "summary": summary,
                        "completed_at": _now()}
            self.atomic_json("complete.json", complete)
            return complete
        finally:
            for path in temporary.values():
                if os.path.lexists(path):
                    os.unlink(path)
