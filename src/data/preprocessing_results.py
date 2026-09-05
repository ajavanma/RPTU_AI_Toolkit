"""Preprocessing outcomes, independent of the point-cloud dependencies."""


def preprocess_file_pair(matched_file_pair, voxel_size, preprocessor_type, logger):
    """Return True only when construction and processing both complete."""
    pcd_file, asc_file = matched_file_pair
    try:
        preprocessor = preprocessor_type(voxel_size)
        preprocessor.process_files(pcd_file, asc_file)
    except Exception as error:
        logger.error(
            "An error occurred during preprocessing %s or its corresponding asc file %s: %s",
            pcd_file, asc_file, error,
        )
        return False
    return True


def report_preprocessing_results(results, logger):
    """Report a completed batch; empty or failed batches return False."""
    if not results:
        logger.warning("No matching files found; no files were processed.")
        return False

    failed_files_count = results.count(False)
    if failed_files_count:
        logger.error("%s files failed to process.", failed_files_count)
        return False

    logger.info("All files processed successfully.")
    return True
