# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Directory Scanner Module

Provides utilities for scanning directories recursively and categorizing files
for the Super AI application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import fnmatch
from datetime import datetime

logger = logging.getLogger(__name__)

class DirectoryScanner:
    """
    Scans directories recursively and categorizes files by type.
    Provides detailed information about project structure.
    """

    def __init__(self, root_dir: Path):
        """
        Initialize the directory scanner with a root directory.

        Args:
            root_dir: The root directory to scan
        """
        self.root_dir = Path(root_dir)
        self.exclude_dirs = {'__pycache__', 'venv', 'env', '.git', '.idea', '.vscode', 'node_modules'}
        self.exclude_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class']

        # File categorization
        self.python_modules: List[Path] = []
        self.data_files: List[Path] = []
        self.log_files: List[Path] = []
        self.frontend_assets: List[Path] = []
        self.config_files: List[Path] = []
        self.other_files: List[Path] = []

        # File type mappings
        self.python_extensions = {'.py'}
        self.data_extensions = {'.csv', '.json', '.jsonl', '.xml', '.txt', '.data'}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        self.log_extensions = {'.log'}
        self.frontend_extensions = {'.html', '.htm', '.js', '.css', '.jsx', '.tsx', '.vue'}
        self.config_extensions = {'.yaml', '.yml', '.ini', '.cfg', '.conf', '.env'}

    def scan(self, detailed_logging: bool = True) -> Dict[str, List[Path]]:
        """
        Scan the root directory recursively and categorize all files.

        Args:
            detailed_logging: Whether to log details of each found file

        Returns:
            Dictionary with categorized file lists
        """
        logger.info(f"Starting directory scan at: {self.root_dir}")

        try:
            # Reset lists before scanning
            self.python_modules = []
            self.data_files = []
            self.log_files = []
            self.frontend_assets = []
            self.config_files = []
            self.other_files = []

            # Track count of processed files
            file_count = 0
            dir_count = 0

            # Traverse directory
            for root, dirs, files in os.walk(self.root_dir):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
                current_path = Path(root)
                dir_count += 1

                for filename in files:
                    # Check if file should be excluded
                    if any(fnmatch.fnmatch(filename, pattern) for pattern in self.exclude_patterns):
                        continue

                    file_path = current_path / filename
                    file_count += 1

                    # Categorize by file extension
                    self._categorize_file(file_path, detailed_logging)

            logger.info(f"Directory scan complete. Processed {dir_count} directories and {file_count} files.")
            logger.info(f"Found {len(self.python_modules)} Python modules")
            logger.info(f"Found {len(self.data_files)} data files")
            logger.info(f"Found {len(self.log_files)} log files")
            logger.info(f"Found {len(self.frontend_assets)} frontend assets")
            logger.info(f"Found {len(self.config_files)} configuration files")
            logger.info(f"Found {len(self.other_files)} other files")

            return {
                'python_modules': self.python_modules,
                'data_files': self.data_files,
                'log_files': self.log_files,
                'frontend_assets': self.frontend_assets,
                'config_files': self.config_files,
                'other_files': self.other_files
            }
        except Exception as e:
            logger.error(f"Error scanning directory {self.root_dir}: {str(e)}")
            return {}

    def _categorize_file(self, file_path: Path, detailed_logging: bool) -> None:
        """
        Categorize a file based on its extension and location.

        Args:
            file_path: Path to the file
            detailed_logging: Whether to log details about the file
        """
        try:
            extension = file_path.suffix.lower()
            relative_path = file_path.relative_to(self.root_dir)

            # Categorize by extension
            if extension in self.python_extensions:
                self.python_modules.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found module: {file_path.stem} at {relative_path}")

            elif extension in self.data_extensions:
                self.data_files.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found data file: {file_path.name} at {relative_path}")

            elif extension in self.image_extensions:
                self.data_files.append(file_path)  # Images are also considered data
                if detailed_logging:
                    logger.debug(f"Found image: {file_path.name} at {relative_path}")

            elif extension in self.log_extensions:
                self.log_files.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found log file: {file_path.name} at {relative_path}")

            elif extension in self.frontend_extensions:
                self.frontend_assets.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found frontend asset: {file_path.name} at {relative_path}")

            elif extension in self.config_extensions:
                self.config_files.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found config file: {file_path.name} at {relative_path}")

            else:
                self.other_files.append(file_path)
                if detailed_logging:
                    logger.debug(f"Found other file: {file_path.name} at {relative_path}")

        except Exception as e:
            logger.error(f"Error categorizing file {file_path}: {str(e)}")

    def get_python_modules_by_prefix(self, prefix: str) -> List[Path]:
        """
        Get Python modules with a specific prefix or in a specific directory.

        Args:
            prefix: The prefix or directory path to filter by

        Returns:
            List of matching Python module paths
        """
        if not self.python_modules:
            logger.warning("No scan performed yet. Call scan() first.")
            return []

        return [
            module for module in self.python_modules
            if str(module.relative_to(self.root_dir)).startswith(prefix)
        ]

    def get_module_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get detailed information about Python modules.

        Returns:
            Dictionary mapping module names to information about them
        """
        if not self.python_modules:
            logger.warning("No scan performed yet. Call scan() first.")
            return {}

        module_info = {}

        for module_path in self.python_modules:
            try:
                module_name = module_path.stem
                module_size = module_path.stat().st_size

                # Extract more info like module docstring, classes, or functions
                # This would require parsing the Python file which is beyond the scope

                # Simple module info
                module_info[str(module_path.relative_to(self.root_dir))] = {
                    'name': module_name,
                    'size': f"{module_size} bytes",
                    'last_modified': str(datetime.fromtimestamp(module_path.stat().st_mtime))
                }
            except Exception as e:
                logger.error(f"Error getting module info for {module_path}: {str(e)}")

        return module_info

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a detailed report of the scanned files.

        Args:
            output_file: Optional path to write the report to

        Returns:
            Report as string
        """
        if not self.python_modules and not self.data_files and not self.other_files:
            logger.warning("No scan performed yet. Call scan() first.")
            return "No scan data available."

        report = []
        report.append("# Super AI Directory Scan Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Root directory: {self.root_dir}")
        report.append("\n## Summary")
        report.append(f"- Python Modules: {len(self.python_modules)}")
        report.append(f"- Data Files: {len(self.data_files)}")
        report.append(f"- Log Files: {len(self.log_files)}")
        report.append(f"- Frontend Assets: {len(self.frontend_assets)}")
        report.append(f"- Configuration Files: {len(self.config_files)}")
        report.append(f"- Other Files: {len(self.other_files)}")

        # Add detailed listings
        if self.python_modules:
            report.append("\n## Python Modules")
            for module in sorted(self.python_modules):
                report.append(f"- {module.relative_to(self.root_dir)}")

        if self.data_files:
            report.append("\n## Data Files")
            for data_file in sorted(self.data_files):
                report.append(f"- {data_file.relative_to(self.root_dir)}")

        if self.log_files:
            report.append("\n## Log Files")
            for log_file in sorted(self.log_files):
                report.append(f"- {log_file.relative_to(self.root_dir)}")

        if self.frontend_assets:
            report.append("\n## Frontend Assets")
            for asset in sorted(self.frontend_assets):
                report.append(f"- {asset.relative_to(self.root_dir)}")

        if self.config_files:
            report.append("\n## Configuration Files")
            for config_file in sorted(self.config_files):
                report.append(f"- {config_file.relative_to(self.root_dir)}")

        report_text = "\n".join(report)

        # Write to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Scan report written to {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to {output_file}: {str(e)}")

        return report_text


# Helper function to scan directories
def scan_directory(root_dir: Path, detailed_logging: bool = False,
                   output_report: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    Scan a directory and return categorized files.

    Args:
        root_dir: Root directory to scan
        detailed_logging: Whether to enable detailed logging
        output_report: Optional path to write the report to

    Returns:
        Dictionary with categorized file lists
    """
    scanner = DirectoryScanner(root_dir)
    result = scanner.scan(detailed_logging)

    if output_report:
        scanner.generate_report(output_report)

    return result


if __name__ == "__main__":
    # Example usage when run as script
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path.cwd()

    scanner = DirectoryScanner(target_dir)
    scanner.scan()

    report_file = Path(f"directory_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    report = scanner.generate_report(report_file)
    print(f"Scan report written to {report_file}")
