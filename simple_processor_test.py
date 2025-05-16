#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Simple File Processor Test

This script demonstrates the file processing capabilities without external dependencies.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_processor")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import FileProcessor directly from file to avoid dependencies
def import_file_processor():
    """Import FileProcessor class directly from file"""
    try:
        # Try the normal import first
        from utils.file_processor import FileProcessor, process_files
        logger.info("Imported FileProcessor from utils package")
        return FileProcessor, process_files
    except ImportError:
        # If that fails, try to import directly from the file
        import importlib.util

        file_path = Path(__file__).parent / 'utils' / 'file_processor.py'
        if not file_path.exists():
            logger.error(f"File processor module not found at {file_path}")
            return None, None

        logger.info(f"Importing FileProcessor directly from {file_path}")

        spec = importlib.util.spec_from_file_location("file_processor", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.FileProcessor, module.process_files

def main():
    """Test the file processor with minimal dependencies"""
    # Import the FileProcessor
    FileProcessor, _ = import_file_processor()
    if FileProcessor is None:
        logger.error("Could not import FileProcessor")
        return 1

    # Determine directory to process
    process_dir = Path(__file__).parent
    data_dir = process_dir / 'data'
    logs_dir = process_dir / 'logs'

    # Create processor
    processor = FileProcessor(process_dir)

    # Process log files
    if logs_dir.exists():
        logger.info(f"Processing log files in {logs_dir}")
        for log_file in logs_dir.glob('*.log'):
            logger.info(f"Scanning log file: {log_file.name}")
            error_count, _ = processor.process_log_file(log_file)
            logger.info(f"Found {error_count} errors in {log_file.name}")
    else:
        logger.warning(f"Logs directory not found: {logs_dir}")

    # Process CSV files
    if data_dir.exists():
        logger.info(f"Processing data files in {data_dir}")
        for csv_file in data_dir.glob('*.csv'):
            logger.info(f"Found CSV file: {csv_file.name}")
            # Just log the file without processing content
            logger.info(f"CSV file path: {csv_file}")
            processor.processed_files['data'].append(str(csv_file))
    else:
        logger.warning(f"Data directory not found: {data_dir}")

    # Process Python files in utils directory
    utils_dir = process_dir / 'utils'
    if utils_dir.exists():
        logger.info(f"Processing Python files in {utils_dir}")
        for py_file in utils_dir.glob('*.py'):
            logger.info(f"Found Python module: {py_file.name}")
            # Just log the file without processing content
            logger.info(f"Python module path: {py_file}")
            processor.processed_files['python'].append(str(py_file))
    else:
        logger.warning(f"Utils directory not found: {utils_dir}")

    # Print summary
    print("\n--- File Processing Summary ---")
    total_files = sum(len(files) for files in processor.processed_files.values())
    print(f"Total files processed: {total_files}")
    print(f"Python modules: {len(processor.processed_files['python'])}")
    print(f"Data files: {len(processor.processed_files['data'])}")
    print(f"Log files: {len(processor.processed_files['logs'])}")

    return 0

if __name__ == "__main__":
    sys.exit(main())