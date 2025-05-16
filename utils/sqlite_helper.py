# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
SQLite Helper

This module provides utilities for SQLite database operations used throughout the system.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class SQLiteHelper:
    """
    Provides a simplified interface for SQLite database operations with error handling.
    """
    
    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """
        Initialize the SQLite helper.
        
        Args:
            db_path: Path to the SQLite database file, or ":memory:" for in-memory database
        """
        self.db_path = str(db_path)
        
        # Create parent directory if it doesn't exist and db is not in-memory
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        self.connection = None
        
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def connect(self) -> bool:
        """
        Connect to the SQLite database.
        
        Returns:
            Success status
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Allow dict-like access to rows
            return True
        except Exception as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}")
            return False
            
    def close(self) -> None:
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            
    def _ensure_connection(self) -> bool:
        """Ensure a connection exists, creating one if needed"""
        if not self.connection:
            return self.connect()
        return True
        
    def execute(self, query: str, params: tuple = None) -> Optional[int]:
        """
        Execute a query (insert, update, delete, create).
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Number of affected rows, or None on error
        """
        if not self._ensure_connection():
            return None
            
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.debug(f"Query: {query}, Params: {params}")
            self.connection.rollback()
            return None
            
    def executemany(self, query: str, params_list: List[tuple]) -> Optional[int]:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows, or None on error
        """
        if not self._ensure_connection():
            return None
            
        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing query multiple times: {e}")
            logger.debug(f"Query: {query}, Params list length: {len(params_list)}")
            self.connection.rollback()
            return None
            
    def query(self, query: str, params: tuple = None) -> List[sqlite3.Row]:
        """
        Execute a query and return the results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of rows
        """
        if not self._ensure_connection():
            return []
            
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.debug(f"Query: {query}, Params: {params}")
            return []
            
    def query_one(self, query: str, params: tuple = None) -> Optional[sqlite3.Row]:
        """
        Execute a query and return a single result.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Single row or None if no results
        """
        if not self._ensure_connection():
            return None
            
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.debug(f"Query: {query}, Params: {params}")
            return None
            
    def insert(self, table: str, data: Dict[str, Any]) -> Optional[int]:
        """
        Insert data into a table.
        
        Args:
            table: Table name
            data: Dictionary of column names and values
            
        Returns:
            Last row ID or None on error
        """
        if not data:
            logger.error(f"No data provided for insert into {table}")
            return None
            
        if not self._ensure_connection():
            return None
            
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            values = tuple(data.values())
            
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting data into {table}: {e}")
            logger.debug(f"Data: {data}")
            self.connection.rollback()
            return None
            
    def update(self, table: str, data: Dict[str, Any], 
              where: str, where_params: tuple) -> Optional[int]:
        """
        Update data in a table.
        
        Args:
            table: Table name
            data: Dictionary of column names and values to update
            where: WHERE clause
            where_params: WHERE clause parameters
            
        Returns:
            Number of affected rows or None on error
        """
        if not data:
            logger.error(f"No data provided for update in {table}")
            return None
            
        if not self._ensure_connection():
            return None
            
        try:
            set_clause = ', '.join([f"{key} = ?" for key in data.keys()])
            values = tuple(data.values()) + where_params
            
            query = f"UPDATE {table} SET {set_clause} WHERE {where}"
            
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error updating data in {table}: {e}")
            logger.debug(f"Data: {data}, Where: {where}, Where params: {where_params}")
            self.connection.rollback()
            return None
            
    def delete(self, table: str, where: str, where_params: tuple) -> Optional[int]:
        """
        Delete data from a table.
        
        Args:
            table: Table name
            where: WHERE clause
            where_params: WHERE clause parameters
            
        Returns:
            Number of affected rows or None on error
        """
        if not self._ensure_connection():
            return None
            
        try:
            query = f"DELETE FROM {table} WHERE {where}"
            
            cursor = self.connection.cursor()
            cursor.execute(query, where_params)
            self.connection.commit()
            
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error deleting data from {table}: {e}")
            logger.debug(f"Where: {where}, Where params: {where_params}")
            self.connection.rollback()
            return None
            
    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table: Table name
            
        Returns:
            Whether the table exists
        """
        if not self._ensure_connection():
            return False
            
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = self.query_one(query, (table,))
            return result is not None
        except Exception as e:
            logger.error(f"Error checking if table {table} exists: {e}")
            return False
            
    def create_table(self, table: str, schema: str) -> bool:
        """
        Create a table if it doesn't exist.
        
        Args:
            table: Table name
            schema: Schema definition (column definitions)
            
        Returns:
            Success status
        """
        if not self._ensure_connection():
            return False
            
        try:
            query = f"CREATE TABLE IF NOT EXISTS {table} ({schema})"
            self.execute(query)
            return True
        except Exception as e:
            logger.error(f"Error creating table {table}: {e}")
            return False
            
    def drop_table(self, table: str) -> bool:
        """
        Drop a table.
        
        Args:
            table: Table name
            
        Returns:
            Success status
        """
        if not self._ensure_connection():
            return False
            
        try:
            query = f"DROP TABLE IF EXISTS {table}"
            self.execute(query)
            return True
        except Exception as e:
            logger.error(f"Error dropping table {table}: {e}")
            return False
            
    def vacuum(self) -> bool:
        """
        Vacuum the database to optimize space.
        
        Returns:
            Success status
        """
        if not self._ensure_connection():
            return False
            
        try:
            self.execute("VACUUM")
            return True
        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            return False
            
    def get_table_info(self, table: str) -> List[Dict[str, Any]]:
        """
        Get information about a table's columns.
        
        Args:
            table: Table name
            
        Returns:
            List of column information dictionaries
        """
        if not self._ensure_connection():
            return []
            
        try:
            result = self.query(f"PRAGMA table_info({table})")
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting table info for {table}: {e}")
            return []
            
    def get_tables(self) -> List[str]:
        """
        Get a list of all tables in the database.
        
        Returns:
            List of table names
        """
        if not self._ensure_connection():
            return []
            
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            result = self.query(query)
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
            
    def backup(self, backup_path: Union[str, Path]) -> bool:
        """
        Backup the database to a file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Success status
        """
        if not self._ensure_connection():
            return False
            
        try:
            backup_path = str(backup_path)
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Connect to backup database
            backup_connection = sqlite3.connect(backup_path)
            
            # Copy data
            self.connection.backup(backup_connection)
            
            # Close backup connection
            backup_connection.close()
            
            return True
        except Exception as e:
            logger.error(f"Error backing up database to {backup_path}: {e}")
            return False
            
    def restore(self, restore_path: Union[str, Path]) -> bool:
        """
        Restore the database from a file.
        
        Args:
            restore_path: Path to the restore file
            
        Returns:
            Success status
        """
        if not self._ensure_connection():
            return False
            
        try:
            restore_path = str(restore_path)
            
            # Check if restore file exists
            if not os.path.exists(restore_path):
                logger.error(f"Restore file does not exist: {restore_path}")
                return False
                
            # Connect to restore database
            restore_connection = sqlite3.connect(restore_path)
            
            # Copy data
            restore_connection.backup(self.connection)
            
            # Close restore connection
            restore_connection.close()
            
            return True
        except Exception as e:
            logger.error(f"Error restoring database from {restore_path}: {e}")
            return False 