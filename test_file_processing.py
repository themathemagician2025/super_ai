#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Test File Processing Module

This script demonstrates the file processing capabilities of the Super AI system.
It scans directories and processes different file types according to their extensions.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/file_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("file_processor_test")

# Ensure utils is in path
sys.path.append(str(Path(__file__).parent))

from utils.file_processor import FileProcessor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Super AI File Processing Test")
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help='Directory to process (default: current working directory)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Process directories recursively')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--filter', '-f', type=str, default=None,
                        help='Filter files by extension (e.g., .py,.csv)')
    return parser.parse_args()

def main():
    """Main entry point for file processing test"""
    args = parse_args()

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Determine directory to process
    process_dir = Path(args.dir) if args.dir else Path.cwd()
    logger.info(f"Processing directory: {process_dir}")

    # Check for filter
    filters = None
    if args.filter:
        filters = [f.strip() for f in args.filter.split(',')]
        logger.info(f"Filtering files by extensions: {filters}")

    # Create file processor
    processor = FileProcessor(process_dir)

    # Process files
    try:
        if filters:
            # Process only specific file types
            for root, _, files in os.walk(process_dir):
                for file in files:
                    file_path = Path(root) / file
                    if any(file_path.suffix.lower() == f.lower() for f in filters):
                        logger.info(f"Processing filtered file: {file_path}")
                        processor.process_file(file_path)

                # Stop recursion if not requested
                if not args.recursive:
                    break
        else:
            # Process all files
            processor.process_directory(process_dir, args.recursive)

        # Print summary
        print("\n--- File Processing Summary ---")
        total_files = sum(len(files) for files in processor.processed_files.values())
        print(f"Total files processed: {total_files}")
        print(f"Python modules: {len(processor.processed_files['python'])}")
        print(f"Data files: {len(processor.processed_files['data'])}")
        print(f"Image files: {len(processor.processed_files['images'])}")
        print(f"Log files: {len(processor.processed_files['logs'])}")
        print(f"Frontend assets: {len(processor.processed_files['frontend'])}")

        # Print details if verbose
        if args.verbose:
            print("\n--- Detailed File List ---")
            for category, files in processor.processed_files.items():
                if files:
                    print(f"\n{category.upper()} files:")
                    for file in files:
                        print(f"  - {file}")

        return 0

    except Exception as e:
        logger.error(f"Error processing files: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())