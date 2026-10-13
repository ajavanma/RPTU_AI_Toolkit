import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from rptu_ai_toolkit.models import MLRuntimeError
from rptu_ai_toolkit.runs import TrainingRun, atomic_json, read_completion, read_run_spec


@unittest.skipUnless(os.name == "posix", "durable training runs require POSIX flock")
class TestTrainingRuns(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "run"
        self.spec = {"epochs": 2, "learning_rate": 0.01, "backend": "torch"}

    def snapshot(self):
        return {str(path.relative_to(self.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.output.rglob("*") if path.is_file()}

    def stage(self, run, attempt):
        directory = run.path / "attempts" / attempt / "model"
        directory.mkdir()
        (directory / "weights.safetensors").write_bytes(b"test-numeric-weights")
        (directory / "training.json").write_text('{"epochs": 2}\n')
        (directory / "bundle.json").write_text('{"weights": "weights.safetensors"}\n')
        return directory

    def complete_run(self):
        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
            with run.attempt() as attempt:
                completion = run.publish(self.stage(run, attempt), {"epochs": 2})
        return completion

    def attempt_documents(self):
        return [json.loads(path.read_text()) for path in sorted(self.output.glob("attempts/*/attempt.json"))]

    def test_new_run_immutable_spec_and_completed_resume_are_read_only(self):
        completion = self.complete_run()
        before = self.snapshot()
        self.assertEqual(read_run_spec(self.output), self.spec)
        self.assertEqual(read_completion(self.output), completion)
        self.assertEqual(self.attempt_documents()[0]["status"], "completed")
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(dict(self.spec))
            self.assertEqual(run.completed(), completion)
            with self.assertRaisesRegex(MLRuntimeError, "already complete"):
                with run.attempt():
                    self.fail("a completed run cannot start another attempt")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(set(completion["files"]), {"bundle.json", "training.json", "weights.safetensors"})

    def test_existing_output_without_resume_rejects_without_changes(self):
        self.complete_run()
        before = self.snapshot()
        with self.assertRaisesRegex(MLRuntimeError, "output already exists"):
            with TrainingRun(self.output):
                self.fail("existing output was accepted")
        self.assertEqual(self.snapshot(), before)

    def test_failed_attempt_retains_files_and_mismatched_resume_changes_nothing(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted training"):
            with TrainingRun(self.output) as run:
                run.bind_spec(self.spec)
                with run.attempt():
                    run.atomic_json("latest.json", {"epoch": 1})
                    raise RuntimeError("interrupted training")
        attempt = self.attempt_documents()[0]
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(attempt["error"], {"type": "RuntimeError", "message": "interrupted training"})
        self.assertTrue((self.output / "latest.json").exists())
        self.assertIsNone(read_completion(self.output))
        before = self.snapshot()
        with self.assertRaisesRegex(MLRuntimeError, "do not match"):
            with TrainingRun(self.output, resume=True) as run:
                run.bind_spec({**self.spec, "epochs": 3})
        self.assertEqual(self.snapshot(), before)

    def test_keyboard_interrupt_is_preserved_as_interrupted_attempt(self):
        with self.assertRaises(KeyboardInterrupt):
            with TrainingRun(self.output) as run:
                run.bind_spec(self.spec)
                with run.attempt():
                    raise KeyboardInterrupt()
        self.assertEqual(self.attempt_documents()[0]["status"], "interrupted")

    def test_corrupt_completion_and_artifact_are_not_silently_retrained(self):
        completion = self.complete_run()
        marker = self.output / "complete.json"
        for change in ({"unexpected": True}, {"files": {}}, {"run_id": "0" * 32}, {"schema_version": True}):
            with self.subTest(change=change):
                marker.write_text(json.dumps({**completion, **change}))
                before = self.snapshot()
                with self.assertRaises(MLRuntimeError):
                    with TrainingRun(self.output, resume=True) as run:
                        run.bind_spec(self.spec)
                self.assertEqual(self.snapshot(), before)
        marker.write_text(json.dumps(completion))
        (self.output / "weights.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(MLRuntimeError, "SHA-256 checksum mismatch"):
            read_completion(self.output)

    def test_strict_run_envelope_rejects_duplicate_unknown_boolean_and_nonfinite_fields(self):
        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
        original = (self.output / "run.json").read_text()
        for corrupted in (
            original.replace('"schema_version": 1', '"schema_version": true'),
            original.replace('"spec": {', '"spec": {"invalid": NaN,'),
            original.replace('"spec": {', '"spec": {"epochs": 2,'),
            original.replace('"spec": {', '"unexpected": 1, "spec": {'),
        ):
            with self.subTest(corrupted=corrupted):
                (self.output / "run.json").write_text(corrupted)
                with self.assertRaises(MLRuntimeError):
                    read_run_spec(self.output)

    def test_symlink_root_lock_and_artifacts_are_rejected(self):
        self.complete_run()
        alias = self.root / "alias"
        alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(MLRuntimeError, "symlink"):
            with TrainingRun(alias, resume=True):
                self.fail("symlink root was accepted")
        lock = self.output / ".run.lock"
        lock.unlink()
        lock.symlink_to(self.root / "outside")
        with self.assertRaisesRegex(MLRuntimeError, "symlinks"):
            with TrainingRun(self.output, resume=True):
                self.fail("symlink lock was accepted")
        self.assertFalse((self.root / "outside").exists())
        lock.unlink()
        weights = self.output / "weights.safetensors"
        external = self.root / "weights"
        weights.rename(external)
        weights.symlink_to(external)
        with self.assertRaisesRegex(MLRuntimeError, "symlinks"):
            read_completion(self.output)

    def test_lock_file_with_multiple_links_is_rejected(self):
        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
        os.link(self.output / ".run.lock", self.root / "lock-alias")
        with self.assertRaisesRegex(MLRuntimeError, "one link"):
            with TrainingRun(self.output, resume=True):
                self.fail("hardlinked lock was accepted")

    def test_contained_writes_reject_path_escape_and_symlink_parent(self):
        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
            (run.path / "alias").symlink_to(self.root, target_is_directory=True)
            for path in ("../escape.json", "/absolute.json", "alias/escape.json", "a/../escape.json", "a\\escape.json", "run.json", ".run.lock"):
                with self.subTest(path=path), self.assertRaises(MLRuntimeError):
                    run.atomic_json(path, {"unsafe": True})
        self.assertFalse((self.root / "escape.json").exists())

    def test_publication_failure_preserves_checkpoint_and_can_be_retried(self):
        original_replace = os.replace

        def interrupt_bundle(source, destination):
            if Path(destination).name == "bundle.json":
                raise OSError("publication interrupted")
            return original_replace(source, destination)

        with self.assertRaisesRegex(OSError, "publication interrupted"):
            with TrainingRun(self.output) as run:
                run.bind_spec(self.spec)
                with run.attempt() as attempt:
                    run.atomic_json("latest.json", {"epoch": 2})
                    staging = self.stage(run, attempt)
                    with mock.patch("rptu_ai_toolkit.runs.os.replace", side_effect=interrupt_bundle):
                        run.publish(staging, {"epochs": 2})
        self.assertIsNone(read_completion(self.output))
        self.assertTrue((staging / "weights.safetensors").exists())
        self.assertTrue((self.output / "latest.json").exists())
        failed_id = self.attempt_documents()[0]["attempt_id"]
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(self.spec)
            with run.attempt() as attempt:
                completion = run.publish(self.stage(run, attempt), {"epochs": 2})
        self.assertEqual(read_completion(self.output), completion)
        attempts = {item["attempt_id"]: item for item in self.attempt_documents()}
        self.assertEqual(attempts[failed_id]["status"], "failed")
        self.assertEqual(attempts[attempt]["status"], "completed")
        self.assertFalse(list(self.output.glob(".*.tmp")))

    def test_completion_marker_failure_does_not_advertise_partial_run_as_complete(self):
        from rptu_ai_toolkit import runs
        original_write = runs.atomic_json

        def interrupt_marker(path, document):
            if Path(path).name == "complete.json":
                raise OSError("completion interrupted")
            return original_write(path, document)

        with self.assertRaisesRegex(OSError, "completion interrupted"):
            with TrainingRun(self.output) as run:
                run.bind_spec(self.spec)
                with run.attempt() as attempt:
                    staging = self.stage(run, attempt)
                    with mock.patch("rptu_ai_toolkit.runs.atomic_json", side_effect=interrupt_marker):
                        run.publish(staging, {"epochs": 2})
        self.assertTrue((self.output / "bundle.json").exists())
        self.assertIsNone(read_completion(self.output))
        self.assertEqual(self.attempt_documents()[0]["status"], "failed")

    def test_interrupted_attempt_initialization_does_not_block_resume(self):
        from rptu_ai_toolkit import runs
        original_write = runs.atomic_json

        def interrupt_initial_record(path, document):
            if Path(path).parent.name.startswith(".attempt-"):
                # Simulate a kill after the temporary file was written but before
                # its atomic replacement, while a previous checkpoint is valid.
                (Path(path).parent / ".attempt.json.abandoned.tmp").write_bytes(b"partial")
                raise OSError("attempt initialization interrupted")
            return original_write(path, document)

        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
            with run.attempt():
                run.atomic_json("latest.json", {"epoch": 1})
        with self.assertRaisesRegex(OSError, "attempt initialization interrupted"):
            with TrainingRun(self.output, resume=True) as run:
                run.bind_spec(self.spec)
                with mock.patch("rptu_ai_toolkit.runs.atomic_json", side_effect=interrupt_initial_record):
                    with run.attempt():
                        self.fail("incomplete attempt became visible")
        self.assertEqual(len(self.attempt_documents()), 1)
        self.assertEqual(json.loads((self.output / "latest.json").read_text()), {"epoch": 1})
        self.assertEqual(len(list(self.output.glob(".attempt-*"))), 1)
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(self.spec)
            with run.attempt() as attempt:
                run.publish(self.stage(run, attempt), {"epochs": 2})
        self.assertIsNotNone(read_completion(self.output))
        self.assertEqual(len(self.attempt_documents()), 2)

    def test_missing_attempts_directory_after_initialization_can_be_recovered(self):
        with TrainingRun(self.output) as run:
            run.bind_spec(self.spec)
        (self.output / "attempts").rmdir()
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(self.spec)
            with run.attempt() as attempt:
                run.publish(self.stage(run, attempt), {"epochs": 2})
        self.assertIsNotNone(read_completion(self.output))

    def test_completed_marker_survives_death_before_attempt_finalization_without_rewrites(self):
        self.complete_run()
        metadata = next(self.output.glob("attempts/*/attempt.json"))
        document = json.loads(metadata.read_text())
        document.update(status="running", finished_at=None)
        metadata.write_text(json.dumps(document))
        before = self.snapshot()
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(self.spec)
            self.assertIsNotNone(run.completed())
        self.assertEqual(self.snapshot(), before)

    def start_writer(self):
        code = """
import json, sys
from rptu_ai_toolkit.runs import TrainingRun
with TrainingRun(sys.argv[1]) as run:
    run.bind_spec(json.loads(sys.argv[2]))
    with run.attempt():
        run.atomic_json('latest.json', {'epoch': 1})
        print('locked', flush=True)
        sys.stdin.read()
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        process = subprocess.Popen([sys.executable, "-c", code, str(self.output), json.dumps(self.spec)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=environment)

        def stop():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

        self.addCleanup(stop)
        ready, _, _ = select.select([process.stdout], [], [], 10)
        self.assertTrue(ready, "child writer did not acquire the lock")
        self.assertEqual(process.stdout.readline().strip(), "locked")
        return process

    def test_second_process_cannot_acquire_an_active_run(self):
        process = self.start_writer()
        before = self.snapshot()
        with self.assertRaisesRegex(MLRuntimeError, "another process"):
            with TrainingRun(self.output, resume=True):
                self.fail("concurrent writer entered")
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(process.poll())

    def test_sigkill_releases_kernel_lock_and_resume_records_abandoned_attempt(self):
        process = self.start_writer()
        old_attempt = self.attempt_documents()[0]["attempt_id"]
        process.kill()
        process.wait(timeout=10)
        self.assertEqual(self.attempt_documents()[0]["status"], "running")
        before = self.snapshot()
        with self.assertRaisesRegex(MLRuntimeError, "do not match"):
            with TrainingRun(self.output, resume=True) as run:
                run.bind_spec({**self.spec, "epochs": 3})
        self.assertEqual(self.snapshot(), before)
        with TrainingRun(self.output, resume=True) as run:
            run.bind_spec(self.spec)
            self.assertEqual(self.snapshot(), before)
            with run.attempt() as attempt:
                previous = next(item for item in self.attempt_documents() if item["attempt_id"] == old_attempt)
                self.assertEqual(previous["status"], "interrupted")
                run.publish(self.stage(run, attempt), {"epochs": 2})
        self.assertEqual(len(self.attempt_documents()), 2)
        self.assertIsNotNone(read_completion(self.output))

    def test_atomic_json_replace_failure_preserves_previous_document(self):
        target = self.root / "state.json"
        atomic_json(target, {"epoch": 1})
        with mock.patch("rptu_ai_toolkit.runs.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                atomic_json(target, {"epoch": 2})
        self.assertEqual(json.loads(target.read_text()), {"epoch": 1})
        self.assertFalse(list(self.root.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
