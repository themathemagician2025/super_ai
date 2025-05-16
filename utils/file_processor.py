# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
File Processor Module

Provides utilities for processing different types of files:
- Python modules (.py)
- Data files (.csv, .json)
- Image files (.png, .jpg)
- Log files (.log)
- Frontend assets (.html, .js, .css)
"""

import os
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# For data processing
try:
    import pandas as pd
except ImportError:
    pd = None

# For image processing
try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

class FileProcessor:
    """
    Processes different types of files and logs information about them.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the file processor with a base directory.

        Args:
            base_dir: Base directory for relative paths. If None, uses current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.processed_files = {
            'python': [],
            'data': [],
            'images': [],
            'logs': [],
            'frontend': []
        }

    def process_python_file(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Dynamically load a Python module and log success or failure.

        Args:
            file_path: Path to the Python file

        Returns:
            Loaded module or None if loading failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for module: {module_name}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            logger.info(f"Successfully loaded module: {module_name}")
            self.processed_files['python'].append(str(file_path))
            return module

        except Exception as e:
            logger.error(f"Failed to load module {file_path}: {str(e)}")
            return None

    def process_csv_file(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load a CSV file using pandas and log its shape.

        Args:
            file_path: Path to the CSV file

        Returns:
            Pandas DataFrame or None if loading failed
        """
        if pd is None:
            logger.error("Pandas not installed. Cannot process CSV files.")
            return None

        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            df = pd.read_csv(file_path)
            rows, cols = df.shape

            logger.info(f"Processed CSV {file_path}: {rows} rows, {cols} columns")
            self.processed_files['data'].append(str(file_path))
            return df

        except Exception as e:
            logger.error(f"Failed to process CSV {file_path}: {str(e)}")
            return None

    def process_image_file(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Open an image file using PIL and log its size.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object or None if opening failed
        """
        if Image is None:
            logger.error("PIL not installed. Cannot process image files.")
            return None

        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            img = Image.open(file_path)
            width, height = img.size

            logger.info(f"Processed image {file_path.name}: {width}x{height}")
            self.processed_files['images'].append(str(file_path))
            return img

        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {str(e)}")
            return None

    def process_log_file(self, file_path: Union[str, Path]) -> Tuple[int, List[str]]:
        """
        Scan a log file for ERROR lines and log findings.

        Args:
            file_path: Path to the log file

        Returns:
            Tuple of (error count, list of error lines)
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            error_lines = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'ERROR' in line:
                        error_lines.append(line.strip())

            error_count = len(error_lines)
            logger.info(f"Found {error_count} errors in {file_path.name}")
            self.processed_files['logs'].append(str(file_path))
            return error_count, error_lines

        except Exception as e:
            logger.error(f"Failed to process log file {file_path}: {str(e)}")
            return 0, []

    def process_frontend_asset(self, file_path: Union[str, Path]) -> bool:
        """
        Log the presence of a frontend asset.

        Args:
            file_path: Path to the frontend asset

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            suffix = file_path.suffix.lower()
            asset_type = {
                '.html': 'HTML',
                '.htm': 'HTML',
                '.js': 'JavaScript',
                '.css': 'CSS',
                '.jsx': 'React',
                '.tsx': 'React/TypeScript',
                '.vue': 'Vue'
            }.get(suffix, 'Other')

            logger.info(f"Found frontend asset: {file_path.name} ({asset_type})")
            self.processed_files['frontend'].append(str(file_path))
            return True

        except Exception as e:
            logger.error(f"Failed to process frontend asset {file_path}: {str(e)}")
            return False

    def process_file(self, file_path: Union[str, Path]) -> Any:
        """
        Process a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Result of the appropriate processing function
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == '.py':
            return self.process_python_file(file_path)
        elif suffix == '.csv':
            return self.process_csv_file(file_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return self.process_image_file(file_path)
        elif suffix == '.log':
            return self.process_log_file(file_path)
        elif suffix in ['.html', '.htm', '.js', '.css', '.jsx', '.tsx', '.vue']:
            return self.process_frontend_asset(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix} for file {file_path}")
            return None

    def process_directory(self, directory: Union[str, Path], recursive: bool = True) -> Dict[str, List[str]]:
        """
        Process all files in a directory.

        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories recursively

        Returns:
            Dictionary of processed files by type
        """
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_dir / dir_path

        logger.info(f"Processing directory: {dir_path}")

        try:
            if recursive:
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = Path(root) / file
                        self.process_file(file_path)
            else:
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        self.process_file(file_path)

            # Log summary
            logger.info(f"Directory processing complete. Processed {sum(len(files) for files in self.processed_files.values())} files:")
            logger.info(f"  - Python modules: {len(self.processed_files['python'])}")
            logger.info(f"  - Data files: {len(self.processed_files['data'])}")
            logger.info(f"  - Image files: {len(self.processed_files['images'])}")
            logger.info(f"  - Log files: {len(self.processed_files['logs'])}")
            logger.info(f"  - Frontend assets: {len(self.processed_files['frontend'])}")

            return self.processed_files

        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {str(e)}")
            return self.processed_files

# Helper function for processing files
def process_files(directory: Union[str, Path], recursive: bool = True) -> Dict[str, List[str]]:
    """
    Process all files in a directory.

    Args:
        directory: Directory to process
        recursive: Whether to process subdirectories recursively

    Returns:
        Dictionary of processed files by type
    """
    processor = FileProcessor()
    return processor.process_directory(directory, recursive)

if __name__ == "__main__":
    # Example usage when run as a script
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path.cwd()

    processor = FileProcessor(target_dir)
    results = processor.process_directory(target_dir)

    print(f"Processed {sum(len(files) for files in results.values())} files")
