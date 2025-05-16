# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Changelog Tracker

This module tracks all code changes made by the self-modification system,
maintaining a detailed history of modifications for auditing and debugging.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class ChangelogTracker:
    """
    Tracks and logs all code modifications made by the self-modification system.
    
    This class provides a comprehensive audit trail of code changes, allowing
    for transparent review of AI-driven code evolution.
    """
    
    def __init__(self, 
                logs_dir: str = None, 
                max_log_files: int = 10,
                auto_rotate: bool = True):
        """
        Initialize the changelog tracker.
        
        Args:
            logs_dir: Directory to store changelog files
            max_log_files: Maximum number of log files to keep
            auto_rotate: Whether to automatically rotate log files
        """
        self.logs_dir = logs_dir or os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "..", 
                "logs", 
                "modifications"
            )
        )
        self.max_log_files = max_log_files
        self.auto_rotate = auto_rotate
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize current log file
        self.current_log_file = os.path.join(
            self.logs_dir, 
            f"changelog_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        # Initialize or load the current log
        self._init_current_log()
        
        logger.info(f"ChangelogTracker initialized with logs directory: {self.logs_dir}")
        
    def _init_current_log(self) -> None:
        """Initialize or load the current log file"""
        if not os.path.exists(self.current_log_file):
            # Create a new log file
            self.current_log = {
                "created_at": datetime.now().isoformat(),
                "modifications": []
            }
            self._save_current_log()
        else:
            # Load the existing log file
            try:
                with open(self.current_log_file, 'r') as f:
                    self.current_log = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading log file {self.current_log_file}, creating new one")
                self.current_log = {
                    "created_at": datetime.now().isoformat(),
                    "modifications": []
                }
                self._save_current_log()
                
    def _save_current_log(self) -> None:
        """Save the current log to file"""
        try:
            with open(self.current_log_file, 'w') as f:
                json.dump(self.current_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving log file: {e}")
            
    def log_modification(self, 
                         target_path: Union[str, Path], 
                         modification_type: str,
                         description: str,
                         details: Dict = None) -> None:
        """
        Log a code modification.
        
        Args:
            target_path: Path to the modified file
            modification_type: Type of modification (e.g., 'deploy_variant', 'refactor')
            description: Description of the modification
            details: Additional details about the modification
        """
        # Create modification entry
        modification = {
            "timestamp": datetime.now().isoformat(),
            "target_path": str(target_path),
            "modification_type": modification_type,
            "description": description,
            "details": details or {}
        }
        
        # Add to the current log
        self.current_log["modifications"].append(modification)
        
        # Save the updated log
        self._save_current_log()
        
        logger.info(f"Logged modification to {target_path}: {description}")
        
        # Rotate logs if needed
        if self.auto_rotate and self._should_rotate_logs():
            self._rotate_logs()
            
    def get_modifications(self, 
                         target_path: Optional[Union[str, Path]] = None,
                         modification_type: Optional[str] = None,
                         days: int = 7,
                         limit: int = 100) -> List[Dict]:
        """
        Get modifications matching the specified criteria.
        
        Args:
            target_path: Filter by target path
            modification_type: Filter by modification type
            days: Number of days to look back
            limit: Maximum number of modifications to return
            
        Returns:
            List of modification entries
        """
        # List of all log files, sorted by date (newest first)
        log_files = sorted(
            [f for f in os.listdir(self.logs_dir) if f.startswith("changelog_") and f.endswith(".json")],
            reverse=True
        )
        
        # Load modifications from log files
        modifications = []
        for log_file in log_files[:days]:  # Limit to specified number of days
            file_path = os.path.join(self.logs_dir, log_file)
            try:
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    modifications.extend(log_data.get("modifications", []))
            except Exception as e:
                logger.error(f"Error reading log file {file_path}: {e}")
                
        # Add modifications from current log
        modifications.extend(self.current_log.get("modifications", []))
        
        # Apply filters
        if target_path:
            target_path = str(target_path)
            modifications = [m for m in modifications if m.get("target_path") == target_path]
            
        if modification_type:
            modifications = [m for m in modifications if m.get("modification_type") == modification_type]
            
        # Sort by timestamp (newest first) and limit
        modifications.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        return modifications[:limit]
    
    def get_modification_history(self, target_path: Union[str, Path]) -> List[Dict]:
        """
        Get the complete modification history for a specific file.
        
        Args:
            target_path: Path to the file
            
        Returns:
            List of modification entries
        """
        return self.get_modifications(target_path=target_path, days=365, limit=1000)
    
    def get_recent_modifications(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent modifications.
        
        Args:
            limit: Maximum number of modifications to return
            
        Returns:
            List of recent modification entries
        """
        return self.get_modifications(days=7, limit=limit)
    
    def _should_rotate_logs(self) -> bool:
        """Check if logs should be rotated"""
        # Check if the current log is for a different day
        current_date = datetime.now().strftime("%Y%m%d")
        log_date = os.path.basename(self.current_log_file).replace("changelog_", "").replace(".json", "")
        
        return current_date != log_date
    
    def _rotate_logs(self) -> None:
        """Rotate log files, creating a new one and removing old ones if necessary"""
        # Save the current log
        self._save_current_log()
        
        # Create a new log file for the current day
        self.current_log_file = os.path.join(
            self.logs_dir, 
            f"changelog_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        self.current_log = {
            "created_at": datetime.now().isoformat(),
            "modifications": []
        }
        
        self._save_current_log()
        
        # Remove old log files if exceeded max_log_files
        log_files = sorted([
            f for f in os.listdir(self.logs_dir) 
            if f.startswith("changelog_") and f.endswith(".json")
        ])
        
        if len(log_files) > self.max_log_files:
            # Remove the oldest log files
            for old_file in log_files[:-self.max_log_files]:
                try:
                    os.remove(os.path.join(self.logs_dir, old_file))
                    logger.debug(f"Removed old log file: {old_file}")
                except Exception as e:
                    logger.error(f"Error removing old log file {old_file}: {e}")
                    
        logger.info("Rotated changelog logs")
    
    def export_modifications(self, 
                            output_path: Union[str, Path],
                            days: int = 30) -> bool:
        """
        Export modifications to a single JSON file.
        
        Args:
            output_path: Path to export the modifications
            days: Number of days to include
            
        Returns:
            Success status
        """
        # Get modifications for the specified days
        modifications = self.get_modifications(days=days, limit=10000)
        
        # Export to file
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "exported_at": datetime.now().isoformat(),
                    "days_included": days,
                    "total_modifications": len(modifications),
                    "modifications": modifications
                }, f, indent=2)
                
            logger.info(f"Exported {len(modifications)} modifications to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting modifications: {e}")
            return False
    
    def generate_changelog_markdown(self, 
                                   output_path: Union[str, Path],
                                   days: int = 7) -> bool:
        """
        Generate a markdown changelog file from modification logs.
        
        Args:
            output_path: Path to export the markdown file
            days: Number of days to include
            
        Returns:
            Success status
        """
        # Get modifications for the specified days
        modifications = self.get_modifications(days=days, limit=1000)
        
        try:
            with open(output_path, 'w') as f:
                f.write(f"# Changelog\n\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Group modifications by date
                modifications_by_date = {}
                for mod in modifications:
                    timestamp = datetime.fromisoformat(mod.get("timestamp", ""))
                    date_str = timestamp.strftime("%Y-%m-%d")
                    
                    if date_str not in modifications_by_date:
                        modifications_by_date[date_str] = []
                        
                    modifications_by_date[date_str].append(mod)
                
                # Write modifications grouped by date
                for date_str, mods in sorted(modifications_by_date.items(), reverse=True):
                    f.write(f"## {date_str}\n\n")
                    
                    for mod in mods:
                        mod_type = mod.get("modification_type", "unknown")
                        target = os.path.basename(mod.get("target_path", "unknown"))
                        desc = mod.get("description", "No description")
                        
                        f.write(f"- **[{mod_type}]** {target}: {desc}\n")
                        
                    f.write("\n")
                    
            logger.info(f"Generated markdown changelog at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating markdown changelog: {e}")
            return False 