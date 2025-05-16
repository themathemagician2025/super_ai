# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Rollback Manager

This module provides functionality for backing up and restoring
code modules, allowing the system to safely rollback to previous
versions if code modifications produce errors or performance degradation.
"""

import os
import sys
import shutil
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from ..utils.file_manager import FileManager

logger = logging.getLogger(__name__)

class RollbackManager:
    """
    Manages code backups and rollback operations to ensure
    the system can revert to a working state after failed modifications.
    """
    
    def __init__(self, 
                backup_dir: str = None, 
                max_backups_per_module: int = 10, 
                auto_prune: bool = True):
        """
        Initialize the rollback manager.
        
        Args:
            backup_dir: Directory to store backups
            max_backups_per_module: Maximum number of backups to keep per module
            auto_prune: Whether to automatically prune old backups
        """
        self.backup_dir = backup_dir or os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "..", 
                "data", 
                "backups"
            )
        )
        self.max_backups_per_module = max_backups_per_module
        self.auto_prune = auto_prune
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Create manifest file if it doesn't exist
        self.manifest_path = os.path.join(self.backup_dir, "rollback_manifest.json")
        if not os.path.exists(self.manifest_path):
            self._create_manifest()
            
        # Load manifest
        self.manifest = self._load_manifest()
        
        # Initialize file manager
        self.file_manager = FileManager()
        
        logger.info(f"RollbackManager initialized with backup directory: {self.backup_dir}")
        
    def _create_manifest(self) -> None:
        """Create an empty manifest file"""
        with open(self.manifest_path, 'w') as f:
            json.dump({
                "modules": {},
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
            
    def _load_manifest(self) -> Dict:
        """Load the backup manifest"""
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load manifest: {e}")
            return {"modules": {}, "last_updated": datetime.now().isoformat()}
            
    def _save_manifest(self) -> None:
        """Save the backup manifest"""
        # Update last updated timestamp
        self.manifest["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            
    def backup_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Create a backup of a file before modification.
        
        Args:
            file_path: Path to the file to back up
            
        Returns:
            Path to the backup file, or None if backup failed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Cannot backup non-existent file: {file_path}")
            return None
            
        try:
            # Generate backup filename with timestamp and hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = self._hash_file(file_path)[:8]
            backup_filename = f"{file_path.stem}_{timestamp}_{file_hash}{file_path.suffix}"
            
            # Create module-specific backup directory
            module_name = file_path.stem
            module_backup_dir = os.path.join(self.backup_dir, module_name)
            os.makedirs(module_backup_dir, exist_ok=True)
            
            # Create the backup path
            backup_path = os.path.join(module_backup_dir, backup_filename)
            
            # Copy the file
            shutil.copy2(file_path, backup_path)
            
            # Update manifest
            if module_name not in self.manifest["modules"]:
                self.manifest["modules"][module_name] = []
                
            backup_info = {
                "original_path": str(file_path),
                "backup_path": backup_path,
                "timestamp": timestamp,
                "hash": file_hash,
                "size": os.path.getsize(file_path)
            }
            
            self.manifest["modules"][module_name].insert(0, backup_info)
            self._save_manifest()
            
            # Prune old backups if needed
            if self.auto_prune:
                self._prune_old_backups(module_name)
                
            logger.info(f"Created backup of {file_path} at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")
            return None
            
    def restore_file(self, file_path: Union[str, Path], 
                   version: Union[int, str] = 0) -> bool:
        """
        Restore a file from backup.
        
        Args:
            file_path: Path to the file to restore
            version: Backup version to restore (0 = most recent, or specific timestamp)
            
        Returns:
            Success status
        """
        file_path = Path(file_path)
        module_name = file_path.stem
        
        if module_name not in self.manifest["modules"]:
            logger.error(f"No backups found for module: {module_name}")
            return False
            
        backups = self.manifest["modules"][module_name]
        if not backups:
            logger.error(f"Backup list is empty for module: {module_name}")
            return False
            
        # Find the backup to restore
        backup_to_restore = None
        
        if isinstance(version, int):
            # Use index as version
            if version < 0 or version >= len(backups):
                logger.error(f"Invalid version index: {version}. " +
                           f"Valid range: 0-{len(backups)-1}")
                return False
                
            backup_to_restore = backups[version]
            
        else:
            # Use timestamp or hash as version
            for backup in backups:
                if backup["timestamp"] == version or backup["hash"] == version:
                    backup_to_restore = backup
                    break
                    
            if not backup_to_restore:
                logger.error(f"No backup found with timestamp or hash: {version}")
                return False
                
        # Restore the file
        try:
            backup_path = backup_to_restore["backup_path"]
            
            # Ensure backup file exists
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
                
            # Create a backup of the current file before restoring
            # This allows rolling forward again if needed
            current_backup = self.backup_file(file_path)
            if not current_backup:
                logger.warning(f"Failed to backup current state before restore")
                
            # Copy the backup file to the original location
            shutil.copy2(backup_path, file_path)
            
            logger.info(f"Restored {file_path} from backup {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore {file_path}: {e}")
            return False
            
    def list_backups(self, module_name: str) -> List[Dict]:
        """
        List all backups for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of backup information dictionaries
        """
        if module_name not in self.manifest["modules"]:
            return []
            
        return self.manifest["modules"][module_name]
        
    def get_diff(self, file_path: Union[str, Path], 
               version: Union[int, str] = 0) -> str:
        """
        Get diff between current file and a backup version.
        
        Args:
            file_path: Path to the file
            version: Backup version to compare against
            
        Returns:
            Unified diff as string
        """
        import difflib
        
        file_path = Path(file_path)
        module_name = file_path.stem
        
        if module_name not in self.manifest["modules"]:
            return f"No backups found for {module_name}"
            
        backups = self.manifest["modules"][module_name]
        if not backups:
            return f"Backup list is empty for {module_name}"
            
        # Find the backup to compare
        backup_to_compare = None
        
        if isinstance(version, int):
            # Use index as version
            if version < 0 or version >= len(backups):
                return f"Invalid version index: {version}. Valid range: 0-{len(backups)-1}"
                
            backup_to_compare = backups[version]
            
        else:
            # Use timestamp or hash as version
            for backup in backups:
                if backup["timestamp"] == version or backup["hash"] == version:
                    backup_to_compare = backup
                    break
                    
            if not backup_to_compare:
                return f"No backup found with timestamp or hash: {version}"
                
        # Get diff
        try:
            backup_path = backup_to_compare["backup_path"]
            
            # Ensure backup file exists
            if not os.path.exists(backup_path):
                return f"Backup file not found: {backup_path}"
                
            # Read both files
            with open(file_path, 'r') as f:
                current_lines = f.readlines()
                
            with open(backup_path, 'r') as f:
                backup_lines = f.readlines()
                
            # Generate diff
            diff = difflib.unified_diff(
                backup_lines, 
                current_lines, 
                fromfile=f"{backup_path} (backup)",
                tofile=f"{file_path} (current)",
                n=3
            )
            
            return ''.join(diff)
            
        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return f"Error generating diff: {str(e)}"
            
    def _prune_old_backups(self, module_name: str) -> None:
        """
        Prune old backups for a module to stay within max_backups_per_module.
        
        Args:
            module_name: Name of the module to prune backups for
        """
        if module_name not in self.manifest["modules"]:
            return
            
        backups = self.manifest["modules"][module_name]
        if len(backups) <= self.max_backups_per_module:
            return
            
        # Remove excess backups
        backups_to_remove = backups[self.max_backups_per_module:]
        self.manifest["modules"][module_name] = backups[:self.max_backups_per_module]
        
        # Delete backup files
        for backup in backups_to_remove:
            backup_path = backup["backup_path"]
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    logger.debug(f"Removed old backup: {backup_path}")
            except Exception as e:
                logger.error(f"Failed to remove old backup {backup_path}: {e}")
                
        self._save_manifest()
        
    def _hash_file(self, file_path: Union[str, Path]) -> str:
        """
        Generate SHA-256 hash of file contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash string
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""
            
    def cleanup(self) -> None:
        """Clean up invalid entries in the manifest"""
        modules_to_update = {}
        
        for module_name, backups in self.manifest["modules"].items():
            valid_backups = []
            
            for backup in backups:
                backup_path = backup["backup_path"]
                if os.path.exists(backup_path):
                    valid_backups.append(backup)
                else:
                    logger.debug(f"Removing invalid backup entry: {backup_path}")
                    
            modules_to_update[module_name] = valid_backups
            
        self.manifest["modules"] = modules_to_update
        self._save_manifest()
        
    def create_checkpoint(self, label: str, files: List[Union[str, Path]]) -> str:
        """
        Create a checkpoint of multiple files with a label.
        
        Args:
            label: Label for the checkpoint
            files: List of file paths to include in the checkpoint
            
        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID
        checkpoint_id = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.backup_dir, "checkpoints", checkpoint_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Back up each file
        backup_info = []
        for file_path in files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Skipping non-existent file: {file_path}")
                continue
                
            try:
                # Create relative directory structure in checkpoint
                rel_path = os.path.relpath(file_path.parent, start=Path(self.backup_dir).parent)
                target_dir = os.path.join(checkpoint_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy file to checkpoint
                target_path = os.path.join(target_dir, file_path.name)
                shutil.copy2(file_path, target_path)
                
                # Add to backup info
                backup_info.append({
                    "original_path": str(file_path),
                    "backup_path": target_path,
                    "hash": self._hash_file(file_path)
                })
                
            except Exception as e:
                logger.error(f"Failed to backup {file_path} to checkpoint: {e}")
                
        # Create checkpoint manifest
        checkpoint_manifest = {
            "id": checkpoint_id,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "files": backup_info
        }
        
        # Save checkpoint manifest
        with open(os.path.join(checkpoint_dir, "checkpoint.json"), 'w') as f:
            json.dump(checkpoint_manifest, f, indent=2)
            
        logger.info(f"Created checkpoint {checkpoint_id} with {len(backup_info)} files")
        return checkpoint_id
        
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore all files from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            Success status
        """
        # Find checkpoint directory
        checkpoint_dir = os.path.join(self.backup_dir, "checkpoints", checkpoint_id)
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
            
        # Read checkpoint manifest
        manifest_path = os.path.join(checkpoint_dir, "checkpoint.json")
        try:
            with open(manifest_path, 'r') as f:
                checkpoint_manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read checkpoint manifest: {e}")
            return False
            
        # Restore each file
        success_count = 0
        total_files = len(checkpoint_manifest["files"])
        
        for file_info in checkpoint_manifest["files"]:
            original_path = file_info["original_path"]
            backup_path = file_info["backup_path"]
            
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                continue
                
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(original_path), exist_ok=True)
                
                # Copy file back to original location
                shutil.copy2(backup_path, original_path)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to restore {original_path} from checkpoint: {e}")
                
        logger.info(f"Restored {success_count}/{total_files} files from checkpoint {checkpoint_id}")
        return success_count == total_files
        
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        checkpoints_dir = os.path.join(self.backup_dir, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return []
            
        for checkpoint_id in os.listdir(checkpoints_dir):
            manifest_path = os.path.join(checkpoints_dir, checkpoint_id, "checkpoint.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        checkpoints.append({
                            "id": manifest["id"],
                            "label": manifest["label"],
                            "timestamp": manifest["timestamp"],
                            "file_count": len(manifest["files"])
                        })
                except Exception as e:
                    logger.error(f"Failed to read checkpoint manifest {manifest_path}: {e}")
                    
        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints 