# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from pathlib import Path
import logging
import sys
import pkg_resources
import asyncio
import multiprocessing
from typing import Dict, List, Set, Tuple, Optional, Any
import importlib

logger = logging.getLogger(__name__)

class SystemScanner:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.python_files: Set[Path] = set()
        self.config_files: Set[Path] = set()
        self.data_files: Set[Path] = set()

    async def scan_directory(self) -> Tuple[Set[Path], Set[Path], Set[Path]]:
        """Asynchronously scan directory for different file types."""
        logger.info(f"Scanning directory: {self.root_dir}")

        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                if file_path.suffix == '.py':
                    self.python_files.add(file_path)
                elif file_path.suffix in {'.json', '.yaml', '.yml', '.ini', '.conf'}:
                    self.config_files.add(file_path)
                elif file_path.suffix in {'.csv', '.data', '.txt', '.log'}:
                    self.data_files.add(file_path)

        logger.info(f"Found {len(self.python_files)} Python files, "
                   f"{len(self.config_files)} config files, "
                   f"{len(self.data_files)} data files")

        return self.python_files, self.config_files, self.data_files

class DependencyManager:
    @staticmethod
    def check_dependencies(requirements_file: Path) -> Tuple[List[str], List[str]]:
        """Check if all dependencies are installed."""
        installed_packages = {pkg.key: pkg.version for pkg
                            in pkg_resources.working_set}
        missing = []
        installed = []

        with open(requirements_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    package = line.split('>=')[0].split('==')[0].strip()
                    if package.lower() not in installed_packages:
                        missing.append(package)
                    else:
                        installed.append(f"{package}=={installed_packages[package.lower()]}")

        return installed, missing

    @staticmethod
    def install_missing_dependencies(missing: List[str]) -> bool:
        """Install missing dependencies."""
        try:
            import pip
            for package in missing:
                logger.info(f"Installing {package}")
                if pip.main(['install', package]) != 0:
                    logger.error(f"Failed to install {package}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False

class ModuleLoader:
    @staticmethod
    def load_python_module(file_path: Path) -> bool:
        """Dynamically load a Python module."""
        try:
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"Successfully loaded module: {module_name}")
                return True
        except Exception as e:
            logger.error(f"Error loading module {file_path}: {e}")
        return False

async def scan_and_setup_system(root_dir: Path) -> Dict[str, Any]:
    """Main function to scan and set up the system."""
    scanner = SystemScanner(root_dir)
    python_files, config_files, data_files = await scanner.scan_directory()

    # Check dependencies
    requirements = root_dir / 'requirements.txt'
    if requirements.exists():
        dep_manager = DependencyManager()
        installed, missing = dep_manager.check_dependencies(requirements)
        if missing:
            logger.warning(f"Missing dependencies: {missing}")
            if dep_manager.install_missing_dependencies(missing):
                logger.info("Successfully installed missing dependencies")
            else:
                logger.error("Failed to install some dependencies")

    return {
        'python_files': python_files,
        'config_files': config_files,
        'data_files': data_files
    }

def get_cpu_count() -> int:
    """Get the number of CPU cores for parallel processing."""
    return max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
