# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
File Manager

This module provides utilities for file operations used throughout the system.
"""

import os
import shutil
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class FileManager:
    """
    Handles file operations with safety measures and validation.
    """
    
    def __init__(self, workspace_root: str = None):
        """
        Initialize the file manager.
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        
    def resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a path relative to the workspace root.
        
        Args:
            path: Path to resolve
            
        Returns:
            Absolute Path object
        """
        path = Path(path)
        
        if path.is_absolute():
            return path
            
        return Path(self.workspace_root) / path
        
    def read_file(self, path: Union[str, Path]) -> str:
        """
        Read a text file with error handling.
        
        Args:
            path: Path to the file
            
        Returns:
            File contents as string
        """
        path = self.resolve_path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise
            
    def write_file(self, path: Union[str, Path], content: str, 
                 mode: str = 'w', safe_write: bool = True) -> bool:
        """
        Write to a file with safety measures.
        
        Args:
            path: Path to the file
            content: Content to write
            mode: Write mode ('w' or 'a')
            safe_write: Whether to use safe writing (temp file + rename)
            
        Returns:
            Success status
        """
        path = self.resolve_path(path)
        
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(path.parent, exist_ok=True)
            
            if safe_write:
                # Write to a temporary file first
                with tempfile.NamedTemporaryFile(mode=mode, 
                                              suffix='.tmp', 
                                              dir=path.parent, 
                                              delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(content)
                    
                # Atomic rename
                shutil.move(tmp_path, path)
            else:
                # Direct write
                with open(path, mode, encoding='utf-8') as f:
                    f.write(content)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error writing to file {path}: {e}")
            # Cleanup temp file if it exists
            if safe_write and 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            return False
            
    def copy_file(self, src_path: Union[str, Path], 
                dest_path: Union[str, Path], 
                overwrite: bool = True) -> bool:
        """
        Copy a file with safety measures.
        
        Args:
            src_path: Source file path
            dest_path: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Success status
        """
        src_path = self.resolve_path(src_path)
        dest_path = self.resolve_path(dest_path)
        
        try:
            # Check if source exists
            if not src_path.exists():
                logger.error(f"Source file does not exist: {src_path}")
                return False
                
            # Check if destination exists and overwrite is not allowed
            if dest_path.exists() and not overwrite:
                logger.error(f"Destination file exists and overwrite not allowed: {dest_path}")
                return False
                
            # Create parent directory if it doesn't exist
            os.makedirs(dest_path.parent, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dest_path)
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {src_path} to {dest_path}: {e}")
            return False
            
    def move_file(self, src_path: Union[str, Path], 
                dest_path: Union[str, Path], 
                overwrite: bool = True) -> bool:
        """
        Move a file with safety measures.
        
        Args:
            src_path: Source file path
            dest_path: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Success status
        """
        src_path = self.resolve_path(src_path)
        dest_path = self.resolve_path(dest_path)
        
        try:
            # Check if source exists
            if not src_path.exists():
                logger.error(f"Source file does not exist: {src_path}")
                return False
                
            # Check if destination exists and overwrite is not allowed
            if dest_path.exists() and not overwrite:
                logger.error(f"Destination file exists and overwrite not allowed: {dest_path}")
                return False
                
            # Create parent directory if it doesn't exist
            os.makedirs(dest_path.parent, exist_ok=True)
            
            # Move file
            shutil.move(src_path, dest_path)
            return True
            
        except Exception as e:
            logger.error(f"Error moving file from {src_path} to {dest_path}: {e}")
            return False
            
    def delete_file(self, path: Union[str, Path], secure: bool = False) -> bool:
        """
        Delete a file with safety measures.
        
        Args:
            path: Path to the file
            secure: Whether to securely delete the file (overwrite with zeros)
            
        Returns:
            Success status
        """
        path = self.resolve_path(path)
        
        try:
            # Check if file exists
            if not path.exists():
                logger.warning(f"File does not exist: {path}")
                return True  # Consider it a success if the file doesn't exist
                
            if secure:
                # Securely delete by overwriting with zeros
                file_size = os.path.getsize(path)
                with open(path, 'wb') as f:
                    f.write(b'\0' * file_size)
                    
            # Delete the file
            os.remove(path)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {path}: {e}")
            return False
            
    def list_files(self, directory: Union[str, Path], 
                 pattern: str = "*", 
                 recursive: bool = False) -> List[Path]:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list files from
            pattern: Glob pattern to match files
            recursive: Whether to recursively list files
            
        Returns:
            List of file paths
        """
        directory = self.resolve_path(directory)
        
        try:
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory}")
                return []
                
            if recursive:
                return list(directory.glob(f"**/{pattern}"))
            else:
                return list(directory.glob(pattern))
                
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            return []
            
    def get_file_hash(self, path: Union[str, Path], 
                    algorithm: str = 'sha256') -> str:
        """
        Calculate the hash of a file.
        
        Args:
            path: Path to the file
            algorithm: Hash algorithm to use
            
        Returns:
            Hash string
        """
        path = self.resolve_path(path)
        
        try:
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return ""
                
            hash_obj = hashlib.new(algorithm)
            
            with open(path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
                    
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {path}: {e}")
            return ""
            
    def ensure_directory(self, path: Union[str, Path]) -> bool:
        """
        Ensure a directory exists.
        
        Args:
            path: Path to the directory
            
        Returns:
            Success status
        """
        path = self.resolve_path(path)
        
        try:
            os.makedirs(path, exist_ok=True)
            return True
            
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")
            return False
            
    def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            path: Path to the file
            
        Returns:
            Dictionary of file information
        """
        path = self.resolve_path(path)
        
        try:
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return {}
                
            stat = path.stat()
            
            return {
                "path": str(path),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "is_dir": path.is_dir(),
                "extension": path.suffix,
                "name": path.name,
                "stem": path.stem
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {path}: {e}")
            return {}
            
    def find_files_by_content(self, directory: Union[str, Path], 
                            search_text: str, 
                            file_pattern: str = "*.py") -> List[Path]:
        """
        Find files containing specific text.
        
        Args:
            directory: Directory to search in
            search_text: Text to search for
            file_pattern: Pattern to match files
            
        Returns:
            List of matching file paths
        """
        directory = self.resolve_path(directory)
        matching_files = []
        
        try:
            # Get all matching files
            files = self.list_files(directory, file_pattern, recursive=True)
            
            for file_path in files:
                try:
                    # Read file content
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                    # Check if search text is in content
                    if search_text in content:
                        matching_files.append(file_path)
                except Exception as e:
                    logger.debug(f"Error reading file {file_path}: {e}")
                    continue
                    
            return matching_files
            
        except Exception as e:
            logger.error(f"Error searching for files in {directory}: {e}")
            return [] 