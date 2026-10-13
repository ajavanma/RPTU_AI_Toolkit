"""Small, deterministic artifact writers shared by CPU dataset commands."""

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Iterator
import zipfile

import numpy as np

from rptu_ai_toolkit.datasets import DatasetError


@contextmanager
def new_output_directory(path: Path | str) -> Iterator[Path]:
    """Reserve a new output directory and remove only our directory on failure.

    Existing directories, files, and symlinks (including broken symlinks) are
    never reused. Missing parents are created and may remain after a failure;
    cleanup owns only the exclusively reserved final output directory.
    """
    try:
        candidate = Path(path).expanduser()
        if os.path.lexists(candidate):
            raise DatasetError(f"{candidate}: output already exists; choose a new path")
        output = candidate.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        identity = output.stat()
    except DatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DatasetError(f"{path}: cannot create output directory: {error}") from error

    try:
        yield output
    except BaseException as error:
        # Do not remove a directory or symlink substituted for the one we own.
        try:
            current = output.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and current.st_dev == identity.st_dev
                and current.st_ino == identity.st_ino
            ):
                shutil.rmtree(output)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            error.add_note(f"Could not clean incomplete output {output}: {cleanup_error}")
        if isinstance(error, OSError):
            raise DatasetError(f"{output}: cannot write dataset output: {error}") from error
        raise


def write_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> None:
    """Write numeric arrays with fixed ZIP metadata and no pickle payloads.

    ZIP entries are stored uncompressed in sorted order with a fixed timestamp.
    NPY format 1.0 and little-endian arrays make bytes reproducible across host
    endianness and independent of wall-clock time. Dataset contracts impose the
    field-specific dtypes and shapes separately.
    """
    for name in arrays:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise DatasetError(f"{path}: invalid NPZ array name {name!r}")
    entries = []
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise DatasetError(f"{path}: array {name!r} must have a numeric dtype")
        array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        payload = io.BytesIO()
        np.lib.format.write_array(payload, array, version=(1, 0), allow_pickle=False)
        entries.append((name, payload.getvalue()))

    try:
        with Path(path).open("xb") as stream:
            with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
                for name, payload in entries:
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, payload)
    except OSError as error:
        raise DatasetError(f"{path}: cannot write NPZ artifact: {error}") from error


def write_json(path: Path | str, document: object) -> None:
    """Write stable, human-readable JSON without nonstandard NaN values."""
    try:
        contents = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
    except (OSError, TypeError, ValueError) as error:
        raise DatasetError(f"{path}: cannot write JSON artifact: {error}") from error
