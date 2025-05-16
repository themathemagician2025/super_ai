import os
import sqlite3
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced cache management utility for the Super AI system"""
    
    def __init__(self, cache_dir: Union[str, Path] = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_db = self.cache_dir / "cache.db"
        self._init_cache_db()
        
    def _init_cache_db(self):
        """Initialize cache database with versioning support"""
        os.makedirs(self.cache_dir, exist_ok=True)
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        # Create tables with additional metadata
        c.executescript('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT PRIMARY KEY,
                last_modified REAL,
                hash TEXT,
                cache_version TEXT,
                process_date TIMESTAMP,
                file_type TEXT,
                status TEXT,
                error_message TEXT
            );
            
            CREATE TABLE IF NOT EXISTS processing_stats (
                timestamp TIMESTAMP PRIMARY KEY,
                total_files INTEGER,
                processed_files INTEGER,
                failed_files INTEGER,
                cache_hits INTEGER,
                processing_time REAL
            );
        ''')
        
        # Initialize cache version if not exists
        c.execute("INSERT OR IGNORE INTO cache_metadata (key, value) VALUES (?, ?)",
                 ("cache_version", "1.0.0"))
        
        conn.commit()
        conn.close()
    
    def get_cache_version(self) -> str:
        """Get current cache version"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        c.execute("SELECT value FROM cache_metadata WHERE key = ?", ("cache_version",))
        version = c.fetchone()[0]
        conn.close()
        return version
    
    def update_cache_version(self, new_version: str):
        """Update cache version and invalidate old entries"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        # Update version
        c.execute("UPDATE cache_metadata SET value = ? WHERE key = ?",
                 (new_version, "cache_version"))
        
        # Mark old entries as invalid
        c.execute("""
            UPDATE processed_files 
            SET status = 'INVALID'
            WHERE cache_version != ?
        """, (new_version,))
        
        conn.commit()
        conn.close()
        
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on modification time and hash"""
        if not file_path.exists():
            return False
            
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        c.execute("""
            SELECT last_modified, hash, status 
            FROM processed_files 
            WHERE file_path = ? AND cache_version = ?
        """, (str(file_path), self.get_cache_version()))
        
        result = c.fetchone()
        conn.close()
        
        if result is None:
            return True
            
        last_modified, stored_hash, status = result
        current_modified = file_path.stat().st_mtime
        
        if status == 'INVALID':
            return True
        
        if current_modified > last_modified:
            return True
            
        current_hash = self.compute_file_hash(file_path)
        return current_hash != stored_hash
    
    def mark_file_processed(self, file_path: Path, status: str = 'SUCCESS', error_msg: str = None):
        """Mark file as processed with detailed metadata"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        file_hash = self.compute_file_hash(file_path)
        last_modified = file_path.stat().st_mtime
        
        c.execute("""
            INSERT OR REPLACE INTO processed_files 
            (file_path, last_modified, hash, cache_version, process_date, file_type, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            last_modified,
            file_hash,
            self.get_cache_version(),
            datetime.now().isoformat(),
            file_path.suffix,
            status,
            error_msg
        ))
        
        conn.commit()
        conn.close()
    
    def clear_cache(self, file_types: Optional[List[str]] = None, older_than: Optional[datetime] = None):
        """Selectively clear cache based on criteria"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        query = "DELETE FROM processed_files WHERE 1=1"
        params = []
        
        if file_types:
            query += " AND file_type IN ({})".format(",".join("?" * len(file_types)))
            params.extend(file_types)
            
        if older_than:
            query += " AND process_date < ?"
            params.append(older_than.isoformat())
            
        c.execute(query, params)
        conn.commit()
        conn.close()
        
    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        
        stats = {}
        
        # Get total counts
        c.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'INVALID' THEN 1 ELSE 0 END) as invalid
            FROM processed_files
        """)
        row = c.fetchone()
        stats['total_files'] = row[0]
        stats['successful_files'] = row[1]
        stats['failed_files'] = row[2]
        stats['invalid_files'] = row[3]
        
        # Get counts by file type
        c.execute("""
            SELECT file_type, COUNT(*) 
            FROM processed_files 
            GROUP BY file_type
        """)
        stats['by_file_type'] = dict(c.fetchall())
        
        # Get recent errors
        c.execute("""
            SELECT file_path, error_message, process_date 
            FROM processed_files 
            WHERE status = 'ERROR'
            ORDER BY process_date DESC
            LIMIT 10
        """)
        stats['recent_errors'] = [
            {'file': row[0], 'error': row[1], 'date': row[2]}
            for row in c.fetchall()
        ]
        
        conn.close()
        return stats 