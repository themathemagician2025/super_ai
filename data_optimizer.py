"""
Data Optimization Module for Super AI Prediction System

This module processes raw data files (CSV, JSON, etc.), optimizes them using advanced data processing
techniques, converts them to AI-readable notation format, and cleans up original files to save space.

Key features:
- Multi-format data ingestion (CSV, JSON, Parquet)
- Data optimization using Pandas, NumPy, Dask, Vaex
- Automated feature engineering with Featuretools
- Conversion to compact AI notation format
- Original data cleanup for space efficiency
"""

import os
import sys
import json
import logging
import shutil
import glob
import time
from typing import Dict, List, Union, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

# Data processing libraries
import numpy as np
import pandas as pd

# Advanced data processing
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

# Compression and serialization
import pickle
import gzip
import lzma
import zlib
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AINotationFormat:
    """
    Handles conversion to and from AI-readable notation format.
    This format is optimized for machine learning consumption and space efficiency.
    """

    FORMAT_VERSION = "1.0.0"

    @staticmethod
    def convert_to_ai_notation(data: pd.DataFrame, metadata: Dict = None) -> bytes:
        """
        Convert a DataFrame to compressed AI notation format.

        Args:
            data: DataFrame to convert
            metadata: Additional metadata to include

        Returns:
            Compressed binary data in AI notation format
        """
        if metadata is None:
            metadata = {}

        # Add standard metadata
        metadata.update({
            "format_version": AINotationFormat.FORMAT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "rows": len(data),
            "columns": data.columns.tolist(),
            "dtypes": {col: str(data[col].dtype) for col in data.columns},
            "hash": hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()
        })

        # Serialize the DataFrame efficiently
        df_bytes = BytesIO()
        data.to_parquet(df_bytes, engine='pyarrow', compression='snappy')
        df_bytes.seek(0)

        # Combine metadata and data
        combined = {
            "metadata": metadata,
            "data": df_bytes.getvalue()
        }

        # Compress the entire package with LZMA for better compression
        return lzma.compress(pickle.dumps(combined))

    @staticmethod
    def load_from_ai_notation(data_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data from AI notation format.

        Args:
            data_bytes: Compressed binary data in AI notation format

        Returns:
            Tuple of (DataFrame, metadata)
        """
        # Decompress and unpickle
        combined = pickle.loads(lzma.decompress(data_bytes))

        # Extract metadata and data
        metadata = combined["metadata"]
        df_bytes = BytesIO(combined["data"])

        # Read the DataFrame
        df = pd.read_parquet(df_bytes)

        # Validate hash if present
        if "hash" in metadata:
            current_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
            if current_hash != metadata["hash"]:
                logger.warning(f"Data integrity check failed. Hash mismatch.")

        return df, metadata


class DataOptimizer:
    """
    Optimizes data by loading, processing, feature engineering, and converting to AI notation format.
    """

    def __init__(self, ai_notation_dir: str = "ai_notation",
                 backup_dir: Optional[str] = "data_backup",
                 compression_level: int = 9,
                 use_dask: bool = True,
                 use_vaex: bool = True,
                 use_featuretools: bool = True):
        """
        Initialize the data optimizer.

        Args:
            ai_notation_dir: Directory to store AI notation files
            backup_dir: Directory for backups (None to skip backups)
            compression_level: Compression level (1-9)
            use_dask: Whether to use Dask for large datasets
            use_vaex: Whether to use Vaex for out-of-core processing
            use_featuretools: Whether to use Featuretools for automated feature engineering
        """
        self.ai_notation_dir = ai_notation_dir
        self.backup_dir = backup_dir
        self.compression_level = compression_level

        # Feature flags for advanced processing libraries
        self.use_dask = use_dask and DASK_AVAILABLE
        self.use_vaex = use_vaex and VAEX_AVAILABLE
        self.use_featuretools = use_featuretools and FEATURETOOLS_AVAILABLE

        # Create directories
        os.makedirs(ai_notation_dir, exist_ok=True)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)

        # Stats
        self.stats = {
            "files_processed": 0,
            "total_original_size": 0,
            "total_optimized_size": 0,
            "files_removed": 0,
            "features_generated": 0,
            "processing_time": 0
        }

    def optimize_directory(self, data_dir: str, pattern: str = "*.csv",
                          recursive: bool = True, remove_originals: bool = True) -> Dict:
        """
        Optimize all matching files in a directory.

        Args:
            data_dir: Directory containing data files
            pattern: Glob pattern to match files
            recursive: Whether to process subdirectories
            remove_originals: Whether to remove original files after processing

        Returns:
            Statistics about the optimization process
        """
        start_time = time.time()

        # Find all matching files
        if recursive:
            search_pattern = os.path.join(data_dir, "**", pattern)
            files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(data_dir, pattern)
            files = glob.glob(search_pattern)

        logger.info(f"Found {len(files)} files matching pattern {pattern} in {data_dir}")

        # Process each file
        for file_path in files:
            try:
                self.optimize_file(file_path, remove_original=remove_originals)
            except Exception as e:
                logger.error(f"Error optimizing {file_path}: {str(e)}")

        # Update processing time
        self.stats["processing_time"] = time.time() - start_time

        # Calculate compression ratio
        if self.stats["total_original_size"] > 0:
            compression_ratio = self.stats["total_original_size"] / max(1, self.stats["total_optimized_size"])
            self.stats["compression_ratio"] = compression_ratio

            space_saved = self.stats["total_original_size"] - self.stats["total_optimized_size"]
            self.stats["space_saved_mb"] = space_saved / (1024 * 1024)

            logger.info(f"Optimization complete. Compression ratio: {compression_ratio:.2f}x, "
                       f"Space saved: {space_saved / (1024 * 1024):.2f} MB")

        return self.stats

    def optimize_file(self, file_path: str, remove_original: bool = True) -> str:
        """
        Optimize a single data file.

        Args:
            file_path: Path to the data file
            remove_original: Whether to remove the original file after optimization

        Returns:
            Path to the optimized AI notation file
        """
        logger.info(f"Optimizing file: {file_path}")

        # Get original file size
        original_size = os.path.getsize(file_path)
        self.stats["total_original_size"] += original_size

        # Determine file type and load accordingly
        file_extension = os.path.splitext(file_path)[1].lower()

        # Load the data using the appropriate method based on size and type
        df = self._load_data(file_path, file_extension)

        if df is None or len(df) == 0:
            logger.warning(f"No data loaded from {file_path}")
            return None

        # Optimize the dataframe
        optimized_df = self._optimize_dataframe(df)

        # Perform feature engineering if enabled
        features_df = None
        if self.use_featuretools and FEATURETOOLS_AVAILABLE:
            try:
                optimized_df, features_generated = self._perform_feature_engineering(optimized_df)
                self.stats["features_generated"] += features_generated
            except Exception as e:
                logger.error(f"Feature engineering failed: {str(e)}")

        # Prepare metadata
        metadata = {
            "source_file": file_path,
            "original_size": original_size,
            "rows": len(optimized_df),
            "columns": len(optimized_df.columns),
            "optimization_timestamp": datetime.now().isoformat(),
            "data_type": file_extension[1:],  # Remove the dot
            "source_hash": hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        }

        # Convert to AI notation format
        ai_data = AINotationFormat.convert_to_ai_notation(optimized_df, metadata)

        # Save to file
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ai_filename = f"{name_without_ext}_{timestamp}.ainotation"
        ai_path = os.path.join(self.ai_notation_dir, ai_filename)

        with open(ai_path, 'wb') as f:
            f.write(ai_data)

        # Update stats
        ai_size = os.path.getsize(ai_path)
        self.stats["total_optimized_size"] += ai_size
        self.stats["files_processed"] += 1

        logger.info(f"Saved optimized data to {ai_path}. Original: {original_size/1024:.2f} KB, "
                   f"Optimized: {ai_size/1024:.2f} KB, "
                   f"Ratio: {original_size/max(1, ai_size):.2f}x")

        # Backup original if requested
        if self.backup_dir:
            backup_path = os.path.join(self.backup_dir, base_name)
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Backed up original file to {backup_path}")

        # Remove original if requested
        if remove_original:
            try:
                os.remove(file_path)
                self.stats["files_removed"] += 1
                logger.info(f"Removed original file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing original file {file_path}: {str(e)}")

        return ai_path

    def _load_data(self, file_path: str, file_extension: str) -> pd.DataFrame:
        """
        Load data from a file using the appropriate library based on file size and type.

        Args:
            file_path: Path to the data file
            file_extension: File extension

        Returns:
            DataFrame containing the data
        """
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.debug(f"Loading file {file_path}, size: {file_size_mb:.2f} MB")

        try:
            # For large files, use Dask or Vaex if available
            if file_size_mb > 500 and self.use_dask and DASK_AVAILABLE:
                logger.info(f"Using Dask for large file ({file_size_mb:.2f} MB)")
                if file_extension == '.csv':
                    ddf = dd.read_csv(file_path)
                elif file_extension == '.json':
                    ddf = dd.read_json(file_path, lines=True)
                elif file_extension == '.parquet':
                    ddf = dd.read_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file extension for Dask: {file_extension}")
                return ddf.compute()

            elif file_size_mb > 1000 and self.use_vaex and VAEX_AVAILABLE:
                logger.info(f"Using Vaex for very large file ({file_size_mb:.2f} MB)")
                if file_extension == '.csv':
                    vdf = vaex.from_csv(file_path)
                elif file_extension == '.json':
                    # Vaex doesn't directly support JSON, convert via pandas first
                    df = pd.read_json(file_path)
                    vdf = vaex.from_pandas(df)
                elif file_extension == '.parquet':
                    vdf = vaex.open(file_path)
                else:
                    raise ValueError(f"Unsupported file extension for Vaex: {file_extension}")
                return vdf.to_pandas_df()

            else:
                # Use pandas for normal-sized files
                if file_extension == '.csv':
                    return pd.read_csv(file_path)
                elif file_extension == '.json':
                    return pd.read_json(file_path)
                elif file_extension == '.parquet':
                    return pd.read_parquet(file_path)
                elif file_extension == '.excel' or file_extension == '.xlsx' or file_extension == '.xls':
                    return pd.read_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize a DataFrame for memory usage and performance.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        # Make a copy to avoid modifying the original
        df_opt = df.copy()

        # Drop duplicate rows
        initial_rows = len(df_opt)
        df_opt = df_opt.drop_duplicates()
        if len(df_opt) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_opt)} duplicate rows")

        # Optimize data types to reduce memory usage
        for col in df_opt.columns:
            col_data = df_opt[col]

            # Skip non-numeric columns
            if pd.api.types.is_object_dtype(col_data):
                # For string columns, check if they could be categorical
                if col_data.nunique() < len(col_data) * 0.5:
                    df_opt[col] = col_data.astype('category')
                continue

            # Optimize numeric columns
            if pd.api.types.is_integer_dtype(col_data):
                c_min, c_max = col_data.min(), col_data.max()

                # Downcast integers to the smallest possible type
                if c_min >= 0:
                    if c_max < 2**8:
                        df_opt[col] = col_data.astype(np.uint8)
                    elif c_max < 2**16:
                        df_opt[col] = col_data.astype(np.uint16)
                    elif c_max < 2**32:
                        df_opt[col] = col_data.astype(np.uint32)
                else:
                    if c_min > -2**7 and c_max < 2**7:
                        df_opt[col] = col_data.astype(np.int8)
                    elif c_min > -2**15 and c_max < 2**15:
                        df_opt[col] = col_data.astype(np.int16)
                    elif c_min > -2**31 and c_max < 2**31:
                        df_opt[col] = col_data.astype(np.int32)

            # Optimize float columns
            elif pd.api.types.is_float_dtype(col_data):
                df_opt[col] = pd.to_numeric(col_data, downcast='float')

        return df_opt

    def _perform_feature_engineering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Perform automated feature engineering using Featuretools.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame with features, number of features generated)
        """
        if not FEATURETOOLS_AVAILABLE:
            return df, 0

        logger.info("Performing automated feature engineering")

        try:
            # Create an EntitySet
            es = ft.EntitySet(id="data")

            # Add dataframe as entity
            es.add_dataframe(
                dataframe_name="main",
                dataframe=df,
                index="index" if "index" in df.columns else None,
                make_index=True if "index" not in df.columns else False
            )

            # Run deep feature synthesis to generate features
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                max_depth=2,  # Limit depth to avoid combinatorial explosion
                verbose=False
            )

            # Merge new features with original dataframe
            # Only keep features that add value (non-null and not constant)
            useful_features = []
            for col in feature_matrix.columns:
                if col in df.columns:
                    continue

                if feature_matrix[col].nunique() > 1 and feature_matrix[col].isna().sum() < len(feature_matrix) * 0.5:
                    useful_features.append(col)

            # Get just the useful features
            new_features = feature_matrix[useful_features]

            # Combine with original dataframe
            result = pd.concat([df, new_features], axis=1)

            logger.info(f"Generated {len(useful_features)} useful features")
            return result, len(useful_features)

        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            return df, 0

    def load_ai_notation_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data from an AI notation file.

        Args:
            file_path: Path to the AI notation file

        Returns:
            Tuple of (DataFrame, metadata)
        """
        with open(file_path, 'rb') as f:
            data_bytes = f.read()

        return AINotationFormat.load_from_ai_notation(data_bytes)

    def get_ai_notation_files(self) -> List[str]:
        """
        Get list of all AI notation files.

        Returns:
            List of file paths
        """
        return glob.glob(os.path.join(self.ai_notation_dir, "*.ainotation"))

    def get_stats(self) -> Dict:
        """
        Get statistics about the optimization process.

        Returns:
            Dictionary of statistics
        """
        return self.stats


def main():
    """Command-line interface for data optimization"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Optimization Tool")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing data files to optimize")
    parser.add_argument("--ai-dir", type=str, default="ai_notation",
                        help="Directory to store AI notation files")
    parser.add_argument("--backup-dir", type=str, default="data_backup",
                        help="Directory to backup original files (use 'none' to skip backup)")
    parser.add_argument("--pattern", type=str, default="*.csv",
                        help="File pattern to match (e.g., '*.csv', '*.json')")
    parser.add_argument("--recursive", action="store_true", default=False,
                        help="Recursively process subdirectories")
    parser.add_argument("--keep-originals", action="store_true", default=False,
                        help="Keep original files after optimization")
    parser.add_argument("--compression", type=int, default=9,
                        help="Compression level (1-9)")
    parser.add_argument("--no-dask", action="store_true", default=False,
                        help="Disable Dask for large file processing")
    parser.add_argument("--no-vaex", action="store_true", default=False,
                        help="Disable Vaex for out-of-core processing")
    parser.add_argument("--no-featuretools", action="store_true", default=False,
                        help="Disable Featuretools for automated feature engineering")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process backup_dir
    backup_dir = None if args.backup_dir.lower() == 'none' else args.backup_dir

    # Create optimizer
    optimizer = DataOptimizer(
        ai_notation_dir=args.ai_dir,
        backup_dir=backup_dir,
        compression_level=args.compression,
        use_dask=not args.no_dask,
        use_vaex=not args.no_vaex,
        use_featuretools=not args.no_featuretools
    )

    # Run optimization
    stats = optimizer.optimize_directory(
        data_dir=args.data_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        remove_originals=not args.keep_originals
    )

    # Print summary
    print("\nOptimization Summary:")
    print(f"Files processed:      {stats.get('files_processed', 0)}")
    print(f"Files removed:        {stats.get('files_removed', 0)}")
    print(f"Features generated:   {stats.get('features_generated', 0)}")
    print(f"Original size:        {stats.get('total_original_size', 0) / (1024*1024):.2f} MB")
    print(f"Optimized size:       {stats.get('total_optimized_size', 0) / (1024*1024):.2f} MB")

    if stats.get('compression_ratio'):
        print(f"Compression ratio:    {stats.get('compression_ratio'):.2f}x")

    if stats.get('space_saved_mb'):
        print(f"Space saved:          {stats.get('space_saved_mb'):.2f} MB")

    print(f"Processing time:      {stats.get('processing_time', 0):.2f} seconds")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
