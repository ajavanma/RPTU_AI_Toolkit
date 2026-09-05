import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from unittest.mock import patch

from tqdm.auto import tqdm

from src.data.preprocessing_workers import process_file_pairs


# Module-level workers are importable by spawned processes without Open3D.
def inspect_job(matched_file_pair, voxel_size):
    return matched_file_pair, voxel_size, os.getpid()


def successful_job(matched_file_pair, voxel_size):
    return matched_file_pair[0] != 'bad.pcd'


def failing_job(matched_file_pair, voxel_size):
    raise RuntimeError('synthetic worker failure')


class TestPreprocessingWorkers(unittest.TestCase):
    def setUp(self):
        # Exercise serialization on Linux too, rather than relying on fork.
        context = multiprocessing.get_context('spawn')
        pool = patch(
            'concurrent.futures.ProcessPoolExecutor',
            partial(ProcessPoolExecutor, mp_context=context),
        )
        pool.start()
        self.addCleanup(pool.stop)

        # tqdm shares its progress lock with workers; it must use their context.
        previous_lock = tqdm.get_lock()
        tqdm.set_lock(context.RLock())
        self.addCleanup(tqdm.set_lock, previous_lock)

    def test_delivers_each_pair_and_voxel_size_to_child_processes(self):
        pairs = [('second.pcd', 'second.asc'), ('first.pcd', 'first.asc')]

        results = process_file_pairs(inspect_job, pairs, voxel_size=0.025, max_workers=2)

        self.assertEqual([(pair, voxel) for pair, voxel, _ in results], [(pair, 0.025) for pair in pairs])
        self.assertTrue(all(pid != os.getpid() for _, _, pid in results))

    def test_preserves_success_and_failure_results_in_input_order(self):
        pairs = [('good.pcd', 'good.asc'), ('bad.pcd', 'bad.asc'), ('other.pcd', 'other.asc')]

        results = process_file_pairs(successful_job, pairs, voxel_size=0.01, max_workers=2)

        self.assertEqual(results, [True, False, True])

    def test_unhandled_worker_exception_reaches_the_caller(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
            process_file_pairs(failing_job, [('bad.pcd', 'bad.asc')], voxel_size=0.01, max_workers=1)
