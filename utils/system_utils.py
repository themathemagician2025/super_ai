# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
System Utilities

This module provides system-related utilities for the Super AI system,
including directory scanning, module discovery, and system initialization.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class SystemScanner:
    """Class for scanning the system for modules and directories."""

    def __init__(self, base_dir: str = None):
        """
        Initialize the system scanner.

        Args:
            base_dir: Base directory to scan (defaults to current directory)
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.module_cache = {}

    async def find_modules(self, module_dirs: Union[str, List[str]], recursive: bool = False) -> List[str]:
        """
        Find Python modules in the specified directories.

        Args:
            module_dirs: Directory or list of directories to scan
            recursive: Whether to scan recursively

        Returns:
            List of module paths
        """
        if isinstance(module_dirs, str):
            module_dirs = [module_dirs]

        results = []

        for module_dir in module_dirs:
            try:
                dir_path = os.path.join(self.base_dir, module_dir)
                if not os.path.exists(dir_path):
                    logger.warning(f"Directory {dir_path} does not exist")
                    continue

                cache_key = f"{module_dir}_{recursive}"
                if cache_key in self.module_cache:
                    results.extend(self.module_cache[cache_key])
                    continue

                # Find Python files
                modules = []
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            rel_path = os.path.relpath(os.path.join(root, file), self.base_dir)
                            modules.append(rel_path)

                    if not recursive:
                        break

                self.module_cache[cache_key] = modules
                results.extend(modules)

            except Exception as e:
                logger.error(f"Error scanning directory {module_dir}: {str(e)}")

        return results

    async def scan_project(self) -> Dict[str, Any]:
        """
        Scan the project directory structure.

        Returns:
            Dictionary with project structure information
        """
        logger.info(f"Scanning project structure in {self.base_dir}")

        try:
            # Get top-level directories
            top_dirs = [d for d in os.listdir(self.base_dir)
                        if os.path.isdir(os.path.join(self.base_dir, d)) and not d.startswith('.')]

            # Find Python modules in each directory
            modules_by_dir = {}
            for d in top_dirs:
                modules = await self.find_modules(d, recursive=True)
                if modules:
                    modules_by_dir[d] = modules

            # Get main script files
            main_scripts = [f for f in os.listdir(self.base_dir)
                          if f.endswith('.py') and os.path.isfile(os.path.join(self.base_dir, f))]

            return {
                'structure': {
                    'directories': top_dirs,
                    'modules_by_dir': modules_by_dir,
                    'main_scripts': main_scripts
                },
                'timestamp': asyncio.get_event_loop().time()
            }

        except Exception as e:
            logger.error(f"Error scanning project: {str(e)}")
            return {
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }

async def scan_system(base_dir: str = None) -> Dict[str, Any]:
    """
    Utility function to scan the system.

    Args:
        base_dir: Base directory to scan

    Returns:
        Dictionary with system information
    """
    scanner = SystemScanner(base_dir)
    return await scanner.scan_project()
