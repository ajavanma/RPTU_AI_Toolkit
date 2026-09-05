import unittest
from unittest.mock import Mock

from src.data.preprocessing_results import preprocess_file_pair, report_preprocessing_results


class TestPreprocessingResults(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.preprocessor_type = Mock()
        self.pair = ('room.pcd', 'room.asc')

    def process(self):
        return preprocess_file_pair(self.pair, 0.02, self.preprocessor_type, self.logger)

    def test_success_requires_processing_the_pair(self):
        self.assertIs(self.process(), True)

        self.preprocessor_type.assert_called_once_with(0.02)
        self.preprocessor_type.return_value.process_files.assert_called_once_with(*self.pair)
        self.logger.error.assert_not_called()

    def test_processing_exception_returns_failure_and_identifies_the_files(self):
        error = ValueError('invalid point cloud')
        self.preprocessor_type.return_value.process_files.side_effect = error

        self.assertIs(self.process(), False)

        self.assertEqual(self.logger.error.call_args.args[1:], (*self.pair, error))
        self.logger.error.assert_called_once()

    def test_constructor_exception_also_returns_failure(self):
        self.preprocessor_type.side_effect = ValueError('invalid voxel size')

        self.assertIs(self.process(), False)
        self.logger.error.assert_called_once()

    def test_keyboard_interrupt_is_not_swallowed(self):
        self.preprocessor_type.return_value.process_files.side_effect = KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.process()

    def test_successful_batch_reports_success(self):
        self.assertIs(report_preprocessing_results([True, True], self.logger), True)

        self.logger.info.assert_called_once_with('All files processed successfully.')
        self.logger.error.assert_not_called()

    def test_mixed_batch_counts_only_failed_files(self):
        self.assertIs(report_preprocessing_results([True, False, False], self.logger), False)

        self.logger.error.assert_called_once_with('%s files failed to process.', 2)
        self.logger.info.assert_not_called()

    def test_entirely_failed_batch_cannot_report_success(self):
        self.assertIs(report_preprocessing_results([False, False], self.logger), False)

        self.logger.error.assert_called_once_with('%s files failed to process.', 2)
        self.logger.info.assert_not_called()

    def test_empty_batch_reports_no_work(self):
        self.assertIs(report_preprocessing_results([], self.logger), False)

        self.logger.warning.assert_called_once_with('No matching files found; no files were processed.')
        self.logger.info.assert_not_called()
        self.logger.error.assert_not_called()
