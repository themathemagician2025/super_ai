# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Exception raised for data validation errors"""
    pass

class DataTransformationError(Exception):
    """Exception raised for data transformation errors"""
    pass

class DataConnector:
    """Base class for data source connectors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Establish connection to data source"""
        raise NotImplementedError("Subclasses must implement connect method")

    def disconnect(self) -> bool:
        """Close connection to data source"""
        raise NotImplementedError("Subclasses must implement disconnect method")

    def extract_data(self, query: str) -> pd.DataFrame:
        """Extract data from source"""
        raise NotImplementedError("Subclasses must implement extract_data method")

class FileConnector(DataConnector):
    """Connector for file-based data sources"""

    def connect(self) -> bool:
        """Verify file exists and is accessible"""
        file_path = Path(self.config.get('file_path', ''))
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        self.connection = True
        return True

    def disconnect(self) -> bool:
        """No actual connection to close for files"""
        self.connection = None
        return True

    def extract_data(self, file_format: str = None, **kwargs) -> pd.DataFrame:
        """Extract data from file source"""
        file_path = Path(self.config.get('file_path', ''))
        format_type = file_format or self.config.get('format', '').lower()

        try:
            if format_type == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif format_type == 'json':
                return pd.read_json(file_path, **kwargs)
            elif format_type == 'excel' or format_type == 'xlsx':
                return pd.read_excel(file_path, **kwargs)
            elif format_type == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
        except Exception as e:
            logger.error(f"Failed to extract data from {file_path}: {str(e)}")
            raise

class APIConnector(DataConnector):
    """Connector for API-based data sources"""

    def connect(self) -> bool:
        """Initialize API connection parameters"""
        self.api_url = self.config.get('api_url')
        self.api_key = self.config.get('api_key')
        self.headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }
        # In a real implementation, might test connection here
        self.connection = True
        return True

    def disconnect(self) -> bool:
        """Clean up API connection resources"""
        self.connection = None
        return True

    def extract_data(self, endpoint: str = None, params: Dict = None, **kwargs) -> pd.DataFrame:
        """Extract data from API source"""
        import requests

        endpoint = endpoint or self.config.get('endpoint', '')
        params = params or self.config.get('params', {})
        url = f"{self.api_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, **kwargs)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame - assuming JSON structure is compatible
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'results' in data:
                return pd.DataFrame(data['results'])
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
        except Exception as e:
            logger.error(f"Failed to extract data from API {url}: {str(e)}")
            raise

class DataValidator:
    """Handles data validation operations"""

    def __init__(self):
        self.validation_rules = {}

    def add_rule(self, column: str, rule_type: str, params: Dict = None):
        """Add validation rule for a column"""
        if column not in self.validation_rules:
            self.validation_rules[column] = []

        self.validation_rules[column].append({
            'type': rule_type,
            'params': params or {}
        })

    def validate(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Apply validation rules to dataframe

        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {}
        is_valid = True

        for column, rules in self.validation_rules.items():
            if column not in data.columns:
                results[column] = {'status': 'error', 'message': f"Column {column} not found in data"}
                is_valid = False
                continue

            column_results = []
            for rule in rules:
                rule_type = rule['type']
                params = rule['params']

                try:
                    if rule_type == 'not_null':
                        null_count = data[column].isnull().sum()
                        rule_valid = null_count == 0
                        message = f"Column has {null_count} null values"

                    elif rule_type == 'unique':
                        duplicate_count = len(data) - data[column].nunique()
                        rule_valid = duplicate_count == 0
                        message = f"Column has {duplicate_count} duplicate values"

                    elif rule_type == 'range':
                        min_val = params.get('min')
                        max_val = params.get('max')
                        outside_range = 0

                        if min_val is not None:
                            outside_range += (data[column] < min_val).sum()
                        if max_val is not None:
                            outside_range += (data[column] > max_val).sum()

                        rule_valid = outside_range == 0
                        message = f"Column has {outside_range} values outside range"

                    elif rule_type == 'regex':
                        pattern = params.get('pattern')
                        if not pattern:
                            rule_valid = False
                            message = "No regex pattern provided"
                        else:
                            non_matching = data[column].str.contains(pattern, regex=True).value_counts().get(False, 0)
                            rule_valid = non_matching == 0
                            message = f"Column has {non_matching} values not matching pattern"

                    elif rule_type == 'custom':
                        func = params.get('function')
                        if not callable(func):
                            rule_valid = False
                            message = "No valid custom function provided"
                        else:
                            rule_valid, message = func(data[column])

                    else:
                        rule_valid = False
                        message = f"Unknown rule type: {rule_type}"

                    column_results.append({
                        'rule_type': rule_type,
                        'valid': rule_valid,
                        'message': message
                    })

                    if not rule_valid:
                        is_valid = False

                except Exception as e:
                    column_results.append({
                        'rule_type': rule_type,
                        'valid': False,
                        'message': f"Error applying rule: {str(e)}"
                    })
                    is_valid = False

            results[column] = column_results

        return is_valid, results

class DataTransformer:
    """Handles data transformation operations"""

    def __init__(self):
        self.transformation_steps = []
        self.column_transformations = {}

    def add_step(self, name: str, transform_func: callable, columns: List[str] = None):
        """Add a transformation step that applies to multiple columns"""
        self.transformation_steps.append({
            'name': name,
            'function': transform_func,
            'columns': columns
        })

    def add_column_transformation(self, output_column: str, transform_func: callable,
                                  input_columns: List[str] = None):
        """Add a transformation that creates or modifies a specific column"""
        self.column_transformations[output_column] = {
            'function': transform_func,
            'input_columns': input_columns
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to the dataframe"""
        result = data.copy()

        # Apply global transformation steps
        for step in self.transformation_steps:
            try:
                func = step['function']
                columns = step['columns']

                if columns:
                    # Apply only to specified columns
                    for col in columns:
                        if col in result.columns:
                            result[col] = result[col].apply(func)
                else:
                    # Apply to entire dataframe
                    result = func(result)

            except Exception as e:
                logger.error(f"Error in transformation step '{step['name']}': {str(e)}")
                raise DataTransformationError(f"Failed in step '{step['name']}': {str(e)}")

        # Apply column-specific transformations
        for output_col, config in self.column_transformations.items():
            try:
                func = config['function']
                input_cols = config['input_columns']

                if input_cols:
                    # Transform uses multiple input columns
                    if all(col in result.columns for col in input_cols):
                        result[output_col] = result[input_cols].apply(
                            lambda row: func(*[row[col] for col in input_cols]), axis=1
                        )
                    else:
                        missing = [col for col in input_cols if col not in result.columns]
                        logger.warning(f"Skipping transformation for {output_col}, missing columns: {missing}")
                else:
                    # Transform creates a new column without inputs
                    result[output_col] = func()

            except Exception as e:
                logger.error(f"Error in column transformation for '{output_col}': {str(e)}")
                raise DataTransformationError(f"Failed transforming '{output_col}': {str(e)}")

        return result

class DataLoader:
    """Handles data loading operations"""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, data: pd.DataFrame, destination: str, format: str = 'csv',
             mode: str = 'overwrite', partition_by: str = None, **kwargs) -> bool:
        """
        Load data to destination

        Args:
            data: DataFrame to load
            destination: Destination name (e.g. table or filename)
            format: Output format (csv, parquet, json)
            mode: Write mode (overwrite, append)
            partition_by: Column to partition by
            **kwargs: Format-specific arguments

        Returns:
            Success status
        """
        try:
            format = format.lower()

            # Handle partitioning
            if partition_by and partition_by in data.columns:
                result = True
                for partition_value, partition_data in data.groupby(partition_by):
                    # Create partition directory
                    partition_path = self.output_dir / f"{partition_by}={partition_value}"
                    partition_path.mkdir(parents=True, exist_ok=True)

                    # Write partition data
                    partition_file = partition_path / f"{destination}.{format}"
                    result = result and self._write_file(partition_data, partition_file, format, mode, **kwargs)
                return result
            else:
                # Write to single file
                file_path = self.output_dir / f"{destination}.{format}"
                return self._write_file(data, file_path, format, mode, **kwargs)

        except Exception as e:
            logger.error(f"Failed to load data to {destination}: {str(e)}")
            return False

    def _write_file(self, data: pd.DataFrame, file_path: Path,
                   format: str, mode: str, **kwargs) -> bool:
        """Write dataframe to file"""
        try:
            if format == 'csv':
                if mode == 'append' and file_path.exists():
                    header = not file_path.exists()
                    data.to_csv(file_path, mode='a', header=header, index=False, **kwargs)
                else:
                    data.to_csv(file_path, index=False, **kwargs)

            elif format == 'parquet':
                if mode == 'append' and file_path.exists():
                    # Parquet append requires special handling
                    existing = pd.read_parquet(file_path)
                    combined = pd.concat([existing, data], ignore_index=True)
                    combined.to_parquet(file_path, index=False, **kwargs)
                else:
                    data.to_parquet(file_path, index=False, **kwargs)

            elif format == 'json':
                if mode == 'append' and file_path.exists():
                    with open(file_path, 'r') as f:
                        existing = json.load(f)

                    # Handle both list and dict JSON formats
                    if isinstance(existing, list) and data.to_dict('records'):
                        combined = existing + data.to_dict('records')
                        with open(file_path, 'w') as f:
                            json.dump(combined, f)
                    else:
                        logger.error(f"Cannot append to JSON with incompatible formats")
                        return False
                else:
                    data.to_json(file_path, orient='records', **kwargs)

            else:
                logger.error(f"Unsupported output format: {format}")
                return False

            logger.info(f"Successfully wrote data to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write to {file_path}: {str(e)}")
            return False

class DataMonitor:
    """Monitors data pipeline execution and quality"""

    def __init__(self, metrics_dir: Union[str, Path]):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.execution_metrics = {}
        self.quality_metrics = {}
        self.start_time = None

    def start_monitoring(self):
        """Start monitoring pipeline execution"""
        self.start_time = datetime.now()
        self.execution_metrics = {
            'start_time': self.start_time.isoformat(),
            'steps': []
        }

    def record_step(self, step_name: str, status: str,
                   metrics: Dict = None, error: str = None):
        """Record execution step metrics"""
        if not self.start_time:
            self.start_monitoring()

        step_metrics = {
            'name': step_name,
            'start_time': datetime.now().isoformat(),
            'status': status,
            'duration_ms': (datetime.now() - self.start_time).total_seconds() * 1000
        }

        if metrics:
            step_metrics['metrics'] = metrics

        if error:
            step_metrics['error'] = error

        self.execution_metrics['steps'].append(step_metrics)

    def record_data_quality(self, dataset_name: str, row_count: int,
                           column_metrics: Dict = None, validation_results: Dict = None):
        """Record data quality metrics"""
        if dataset_name not in self.quality_metrics:
            self.quality_metrics[dataset_name] = []

        quality_entry = {
            'timestamp': datetime.now().isoformat(),
            'row_count': row_count
        }

        if column_metrics:
            quality_entry['column_metrics'] = column_metrics

        if validation_results:
            quality_entry['validation_results'] = validation_results

        self.quality_metrics[dataset_name].append(quality_entry)

    def calculate_column_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate metrics for each column"""
        metrics = {}

        for column in data.columns:
            col_metrics = {
                'null_count': data[column].isnull().sum(),
                'null_percentage': data[column].isnull().mean() * 100
            }

            # Numeric column metrics
            if np.issubdtype(data[column].dtype, np.number):
                col_metrics.update({
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'mean': data[column].mean(),
                    'median': data[column].median(),
                    'std': data[column].std()
                })

            # String column metrics
            elif data[column].dtype == 'object':
                col_metrics.update({
                    'unique_count': data[column].nunique(),
                    'unique_percentage': data[column].nunique() / len(data) * 100,
                    'most_common': data[column].value_counts().index[0] if not data[column].empty else None,
                    'most_common_count': data[column].value_counts().iloc[0] if not data[column].empty else 0
                })

            metrics[column] = col_metrics

        return metrics

    def save_metrics(self):
        """Save all collected metrics to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save execution metrics
        self.execution_metrics['end_time'] = datetime.now().isoformat()
        self.execution_metrics['total_duration_ms'] = (
            datetime.now() - self.start_time).total_seconds() * 1000

        exec_file = self.metrics_dir / f"execution_metrics_{timestamp}.json"
        with open(exec_file, 'w') as f:
            json.dump(self.execution_metrics, f, indent=2)

        # Save quality metrics
        quality_file = self.metrics_dir / f"quality_metrics_{timestamp}.json"
        with open(quality_file, 'w') as f:
            json.dump(self.quality_metrics, f, indent=2)

        logger.info(f"Metrics saved to {self.metrics_dir}")
        return True

class DataPipeline:
    """Main data pipeline orchestrator"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_path = Path(self.config.get('data_path', '../data/raw'))
        self.processed_path = Path(self.config.get('processed_path', '../data/processed'))
        self.metrics_path = Path(self.config.get('metrics_path', '../data/metrics'))

        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.loader = DataLoader(self.processed_path)
        self.monitor = DataMonitor(self.metrics_path)

        # Pipeline tracking
        self.source_data = None
        self.validated_data = None
        self.transformed_data = None
        self.pipeline_id = self._generate_pipeline_id()

        logger.info(f"Initializing Data Pipeline with ID: {self.pipeline_id}")

    def _generate_pipeline_id(self):
        """Generate unique pipeline run ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"pipeline_{timestamp}_{random_suffix}"

    def extract(self, source_type: str, source_config: Dict, **kwargs) -> pd.DataFrame:
        """Extract data from source"""
        self.monitor.start_monitoring()
        self.monitor.record_step("extract", "started")

        try:
            # Create appropriate connector
            if source_type.lower() == 'file':
                connector = FileConnector(source_config)
            elif source_type.lower() == 'api':
                connector = APIConnector(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Connect and extract
            connector.connect()
            data = connector.extract_data(**kwargs)
            connector.disconnect()

            # Record metrics
            self.source_data = data
            self.monitor.record_step("extract", "completed", {
                'source_type': source_type,
                'row_count': len(data),
                'column_count': len(data.columns)
            })

            return data

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            self.monitor.record_step("extract", "failed", error=str(e))
            raise

    def validate(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Validate data against rules"""
        data = data if data is not None else self.source_data
        self.monitor.record_step("validate", "started")

        try:
            # Apply validation
            is_valid, results = self.validator.validate(data)

            # Record metrics
            column_metrics = self.monitor.calculate_column_metrics(data)
            self.monitor.record_data_quality(
                f"validation_{self.pipeline_id}",
                len(data),
                column_metrics,
                results
            )

            if is_valid:
                self.validated_data = data
                self.monitor.record_step("validate", "completed", {
                    'validation_status': 'passed',
                    'rule_count': sum(len(rules) for rules in self.validator.validation_rules.values())
                })
            else:
                message = "Data validation failed"
                self.monitor.record_step("validate", "failed", {
                    'validation_status': 'failed',
                    'failure_details': results
                }, error=message)
                raise DataValidationError(message)

            return data

        except Exception as e:
            if not isinstance(e, DataValidationError):
                logger.error(f"Validation processing failed: {str(e)}")
                self.monitor.record_step("validate", "failed", error=str(e))
            raise

    def transform(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Transform data"""
        data = data if data is not None else self.validated_data or self.source_data
        self.monitor.record_step("transform", "started")

        try:
            # Apply transformations
            result = self.transformer.transform(data)

            # Record metrics
            self.transformed_data = result
            self.monitor.record_step("transform", "completed", {
                'step_count': len(self.transformer.transformation_steps),
                'column_count_before': len(data.columns),
                'column_count_after': len(result.columns),
                'row_count': len(result)
            })

            return result

        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            self.monitor.record_step("transform", "failed", error=str(e))
            raise

    def load(self, data: pd.DataFrame = None, destination: str = None, **kwargs) -> bool:
        """Load data to destination"""
        data = data if data is not None else self.transformed_data
        self.monitor.record_step("load", "started")

        try:
            # Load data
            destination = destination or f"processed_data_{self.pipeline_id}"
            success = self.loader.load(data, destination, **kwargs)

            # Record metrics
            metrics = {
                'destination': destination,
                'row_count': len(data),
                'success': success
            }
            metrics.update(kwargs)

            if success:
                self.monitor.record_step("load", "completed", metrics)
            else:
                self.monitor.record_step("load", "failed", metrics,
                                        error="Loader reported failure")

            return success

        except Exception as e:
            logger.error(f"Load failed: {str(e)}")
            self.monitor.record_step("load", "failed", error=str(e))
            raise

    def run_pipeline(self, source_type: str, source_config: Dict,
                    destination: str = None, load_kwargs: Dict = None,
                    extract_kwargs: Dict = None) -> bool:
        """Run full ETL pipeline"""
        try:
            # Extract
            data = self.extract(source_type, source_config, **(extract_kwargs or {}))

            # Validate
            if self.validator.validation_rules:
                data = self.validate(data)

            # Transform
            if self.transformer.transformation_steps or self.transformer.column_transformations:
                data = self.transform(data)

            # Load
            success = self.load(data, destination, **(load_kwargs or {}))

            # Save metrics
            self.monitor.save_metrics()

            return success

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self.monitor.save_metrics()
            raise

    def process_data(self, data_source):
        """Process data from various sources"""
        try:
            logger.info(f"Processing data from {data_source}")
            # Configure for this specific data source
            source_config = {'file_path': data_source}

            # Run the full pipeline
            return self.run_pipeline(
                source_type='file',
                source_config=source_config,
                destination=f"processed_{os.path.basename(data_source)}",
                load_kwargs={'format': 'parquet'}
            )
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create pipeline
    pipeline = DataPipeline({
        'data_path': './data/raw',
        'processed_path': './data/processed',
        'metrics_path': './data/metrics'
    })

    # Add validation rules
    pipeline.validator.add_rule('price', 'range', {'min': 0})
    pipeline.validator.add_rule('quantity', 'not_null')

    # Add transformations
    pipeline.transformer.add_step('normalize_prices',
                                lambda x: (x - x.min()) / (x.max() - x.min()),
                                ['price'])

    pipeline.transformer.add_column_transformation(
        'total_value',
        lambda price, qty: price * qty,
        ['price', 'quantity']
    )

    # Run pipeline
    pipeline.run_pipeline(
        source_type='file',
        source_config={'file_path': './data/raw/sales.csv'},
        destination='processed_sales',
        load_kwargs={'format': 'parquet', 'partition_by': 'date'}
    )
