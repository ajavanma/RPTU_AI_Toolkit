import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from utils.file_matching import files_match_making


class TestFileMatching(unittest.TestCase):
    def test_matches_supplied_paths_without_requiring_files_on_disk(self):
        pcd = Path('custom/clouds/room.v2.pcd')
        asc = Path('custom/labels/room.v2.asc')

        self.assertEqual(files_match_making([pcd], [asc]), [(pcd, asc)])

    def test_orders_pairs_by_pcd_path_and_accepts_generators(self):
        pcd_files = (path for path in ['b.pcd', 'a.pcd'])
        asc_files = (path for path in ['a.asc', 'b.asc'])

        self.assertEqual(
            files_match_making(pcd_files, asc_files),
            [('a.pcd', 'a.asc'), ('b.pcd', 'b.asc')],
        )

    def test_warns_and_skips_a_point_cloud_without_labels(self):
        output = io.StringIO()
        with redirect_stdout(output):
            pairs = files_match_making(['missing.pcd', 'room.pcd'], ['room.asc'])

        self.assertEqual(pairs, [('room.pcd', 'room.asc')])
        self.assertIn('No corresponding ASC file found for PCD file missing.pcd.', output.getvalue())

    def test_ignores_unmatched_label_files(self):
        self.assertEqual(
            files_match_making(['room.pcd'], ['unused.asc', 'room.asc']),
            [('room.pcd', 'room.asc')],
        )

    def test_empty_point_cloud_input_returns_no_pairs(self):
        self.assertEqual(files_match_making([], ['room.asc']), [])

    def test_empty_label_input_warns_and_returns_no_pairs(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(files_match_making(['room.pcd'], []), [])
        self.assertIn('room.pcd', output.getvalue())

    def test_duplicate_label_stems_use_first_sorted_path(self):
        self.assertEqual(
            files_match_making(['room.pcd'], ['b/room.asc', 'a/room.asc']),
            [('room.pcd', 'a/room.asc')],
        )
