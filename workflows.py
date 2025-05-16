# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Workflow Orchestration Module for Super AI Prediction System
Implements data pipelines using Airflow and Dagster
"""

import os
import sys
import json
import logging
from typing import Dict, List, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path

# Import our scraper and processor modules
from data_pipeline.scrapers import ScraperFactory, BaseScraper
from data_pipeline.processors import ProcessorFactory, BaseProcessor

# Airflow imports
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.utils.dates import days_ago
    from airflow.models import Variable
    from airflow.hooks.base import BaseHook
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# Dagster imports
try:
    import dagster
    from dagster import job, op, In, Out, graph, resource, Field, IOManager, io_manager
    DAGSTER_AVAILABLE = True
except ImportError:
    DAGSTER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowBase:
    """Base class for workflow orchestration"""

    def __init__(self, workflow_name: str, config_path: Optional[str] = None):
        """
        Initialize the workflow base.

        Args:
            workflow_name: Name of the workflow
            config_path: Path to configuration file
        """
        self.workflow_name = workflow_name
        self.config = self._load_config(config_path)
        self.workflow = None

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "data_dir": "data",
            "scraped_dir": "scraped_data",
            "processed_dir": "processed_data",
            "schedule_interval": "0 0 * * *",  # Daily at midnight
            "max_active_runs": 1,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "retries": 1,
            "retry_delay_minutes": 5,
            "email_on_failure": False,
            "email": "admin@example.com"
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                return default_config

        return default_config

    def create_workflow(self, tasks: List[Dict]) -> Any:
        """
        Create a workflow from tasks.

        Args:
            tasks: List of task definitions

        Returns:
            Created workflow
        """
        raise NotImplementedError("Subclasses must implement create_workflow")

    def execute_workflow(self, **kwargs) -> Any:
        """
        Execute the workflow.

        Args:
            **kwargs: Additional arguments

        Returns:
            Execution result
        """
        raise NotImplementedError("Subclasses must implement execute_workflow")

    def _setup_directories(self) -> None:
        """Ensure all required directories exist"""
        for dir_key in ["data_dir", "scraped_dir", "processed_dir"]:
            os.makedirs(self.config[dir_key], exist_ok=True)


class AirflowWorkflow(WorkflowBase):
    """Workflow orchestration using Apache Airflow"""

    def __init__(self, workflow_name: str, config_path: Optional[str] = None):
        """
        Initialize the Airflow workflow.

        Args:
            workflow_name: Name of the workflow
            config_path: Path to configuration file
        """
        super().__init__(workflow_name, config_path)

        if not AIRFLOW_AVAILABLE:
            raise ImportError(
                "Airflow is not installed. Install with 'pip install apache-airflow'"
            )

        self._setup_directories()

    def create_workflow(self, tasks: List[Dict]) -> DAG:
        """
        Create an Airflow DAG from task definitions.

        Args:
            tasks: List of task definitions

        Returns:
            Airflow DAG
        """
        default_args = {
            'owner': 'super_ai',
            'depends_on_past': False,
            'start_date': days_ago(0),
            'email': [self.config["email"]],
            'email_on_failure': self.config["email_on_failure"],
            'email_on_retry': False,
            'retries': self.config["retries"],
            'retry_delay': timedelta(minutes=self.config["retry_delay_minutes"]),
        }

        dag = DAG(
            self.workflow_name,
            default_args=default_args,
            description=f'Super AI {self.workflow_name} workflow',
            schedule_interval=self.config["schedule_interval"],
            catchup=False,
            max_active_runs=self.config["max_active_runs"],
            concurrency=3
        )

        # Create tasks
        airflow_tasks = {}

        with dag:
            for task in tasks:
                task_id = task.get("id")
                task_type = task.get("type")

                if task_type == "scrape":
                    # Web scraping task
                    scraper_type = task.get("scraper", "soup")
                    urls = task.get("urls", [])
                    output_file = task.get("output", f"{scraper_type}_data")
                    selector = task.get("selector")

                    # Create Python operator for scraping
                    airflow_tasks[task_id] = PythonOperator(
                        task_id=task_id,
                        python_callable=self._scrape_task,
                        op_kwargs={
                            'scraper_type': scraper_type,
                            'urls': urls,
                            'output_file': output_file,
                            'selector': selector,
                            'additional_params': task.get("params", {})
                        }
                    )

                elif task_type == "process":
                    # Data processing task
                    processor_type = task.get("processor", "dataframe")
                    input_file = task.get("input")
                    output_file = task.get("output", f"{processor_type}_processed")
                    operations = task.get("operations", [])

                    # Create Python operator for processing
                    airflow_tasks[task_id] = PythonOperator(
                        task_id=task_id,
                        python_callable=self._process_task,
                        op_kwargs={
                            'processor_type': processor_type,
                            'input_file': input_file,
                            'output_file': output_file,
                            'operations': operations,
                            'additional_params': task.get("params", {})
                        }
                    )

                elif task_type == "wait_for_file":
                    # File sensor task
                    filepath = task.get("filepath")
                    timeout = task.get("timeout", 60 * 60)  # Default: 1 hour

                    airflow_tasks[task_id] = FileSensor(
                        task_id=task_id,
                        filepath=filepath,
                        poke_interval=60,
                        timeout=timeout,
                        mode='poke'
                    )

                elif task_type == "bash":
                    # Bash command task
                    bash_command = task.get("command")

                    airflow_tasks[task_id] = BashOperator(
                        task_id=task_id,
                        bash_command=bash_command
                    )

                elif task_type == "python":
                    # Python callable task
                    function_path = task.get("function")
                    module_path, function_name = function_path.rsplit('.', 1)

                    # Dynamically import the function
                    module = __import__(module_path, fromlist=[function_name])
                    function = getattr(module, function_name)

                    airflow_tasks[task_id] = PythonOperator(
                        task_id=task_id,
                        python_callable=function,
                        op_kwargs=task.get("params", {})
                    )

            # Set up dependencies
            for task in tasks:
                task_id = task.get("id")
                dependencies = task.get("depends_on", [])

                if dependencies and task_id in airflow_tasks:
                    for dep in dependencies:
                        if dep in airflow_tasks:
                            airflow_tasks[dep] >> airflow_tasks[task_id]

        self.workflow = dag
        return dag

    def execute_workflow(self, execution_date=None, **kwargs) -> None:
        """
        Execute the Airflow workflow.

        Args:
            execution_date: Execution date for the DAG run
            **kwargs: Additional arguments
        """
        if not self.workflow:
            raise ValueError("Workflow not created. Call create_workflow first.")

        logger.info(f"To execute this workflow, run the Airflow scheduler and trigger the DAG '{self.workflow_name}'")
        logger.info("This method does not directly execute the workflow as Airflow requires a scheduler and webserver.")
        logger.info("Alternatively, use the Airflow CLI to trigger the DAG:")
        logger.info(f"  airflow dags trigger {self.workflow_name}")

    def _scrape_task(self, scraper_type: str, urls: List[str], output_file: str,
                    selector: Optional[str] = None, additional_params: Dict = None) -> str:
        """
        Airflow task function for web scraping.

        Args:
            scraper_type: Type of scraper to use
            urls: List of URLs to scrape
            output_file: Output file name
            selector: CSS selector for scraping
            additional_params: Additional parameters for the scraper

        Returns:
            Path to the output file
        """
        try:
            scraper = ScraperFactory.create_scraper(
                scraper_type,
                output_dir=self.config["scraped_dir"],
                **(additional_params or {})
            )

            for url in urls:
                scraper.scrape(
                    url=url,
                    selector=selector,
                    **(additional_params or {})
                )

            output_path = scraper.save_results(output_file)
            return output_path

        except Exception as e:
            logger.error(f"Error in scrape task: {str(e)}")
            raise

    def _process_task(self, processor_type: str, input_file: str, output_file: str,
                     operations: List[Dict], additional_params: Dict = None) -> str:
        """
        Airflow task function for data processing.

        Args:
            processor_type: Type of processor to use
            input_file: Input file path
            output_file: Output file name
            operations: List of processing operations
            additional_params: Additional parameters for the processor

        Returns:
            Path to the output file
        """
        try:
            processor = ProcessorFactory.create_processor(
                processor_type,
                output_dir=self.config["processed_dir"],
                **(additional_params or {})
            )

            # Determine input format from file extension
            input_format = "csv"
            if input_file.endswith(".json"):
                input_format = "json"
            elif input_file.endswith(".parquet"):
                input_format = "parquet"

            # Process the data
            result = processor.process(
                data=input_file,
                input_format=input_format,
                operations=operations,
                **(additional_params or {})
            )

            # Determine output format
            output_format = "csv"
            if output_file.endswith(".json"):
                output_format = "json"
            elif output_file.endswith(".parquet"):
                output_format = "parquet"
            else:
                output_file = output_file.split('.')[0]  # Remove extension if any

            # Save the results
            output_path = processor.save_results(result, output_file, output_format)
            return output_path

        except Exception as e:
            logger.error(f"Error in process task: {str(e)}")
            raise


class DagsterWorkflow(WorkflowBase):
    """Workflow orchestration using Dagster"""

    def __init__(self, workflow_name: str, config_path: Optional[str] = None):
        """
        Initialize the Dagster workflow.

        Args:
            workflow_name: Name of the workflow
            config_path: Path to configuration file
        """
        super().__init__(workflow_name, config_path)

        if not DAGSTER_AVAILABLE:
            raise ImportError(
                "Dagster is not installed. Install with 'pip install dagster'"
            )

        self._setup_directories()

    def create_workflow(self, tasks: List[Dict]) -> dagster.GraphDefinition:
        """
        Create a Dagster workflow from task definitions.

        Args:
            tasks: List of task definitions

        Returns:
            Dagster GraphDefinition
        """
        # Create Dagster operations
        ops_dict = {}
        dependencies = {}

        for task in tasks:
            task_id = task.get("id")
            task_type = task.get("type")

            if task_type == "scrape":
                # Web scraping task
                scraper_type = task.get("scraper", "soup")
                urls = task.get("urls", [])
                output_file = task.get("output", f"{scraper_type}_data")
                selector = task.get("selector")

                @op(name=task_id)
                def scrape_op():
                    scraper = ScraperFactory.create_scraper(
                        scraper_type,
                        output_dir=self.config["scraped_dir"],
                        **(task.get("params", {}))
                    )

                    for url in urls:
                        scraper.scrape(
                            url=url,
                            selector=selector,
                            **(task.get("params", {}))
                        )

                    output_path = scraper.save_results(output_file)
                    return output_path

                ops_dict[task_id] = scrape_op

            elif task_type == "process":
                # Data processing task
                processor_type = task.get("processor", "dataframe")
                input_file = task.get("input")
                output_file = task.get("output", f"{processor_type}_processed")
                operations = task.get("operations", [])

                @op(name=task_id)
                def process_op(input_path=None):
                    nonlocal input_file

                    # If input_path is provided (from a previous op), use it instead of the configured input_file
                    if input_path is not None:
                        input_file = input_path

                    processor = ProcessorFactory.create_processor(
                        processor_type,
                        output_dir=self.config["processed_dir"],
                        **(task.get("params", {}))
                    )

                    # Determine input format from file extension
                    input_format = "csv"
                    if input_file.endswith(".json"):
                        input_format = "json"
                    elif input_file.endswith(".parquet"):
                        input_format = "parquet"

                    # Process the data
                    result = processor.process(
                        data=input_file,
                        input_format=input_format,
                        operations=operations,
                        **(task.get("params", {}))
                    )

                    # Determine output format
                    output_format = "csv"
                    if output_file.endswith(".json"):
                        output_format = "json"
                    elif output_file.endswith(".parquet"):
                        output_format = "parquet"
                    else:
                        output_file_base = output_file.split('.')[0]  # Remove extension if any
                        output_file = output_file_base

                    # Save the results
                    output_path = processor.save_results(result, output_file, output_format)
                    return output_path

                ops_dict[task_id] = process_op

            elif task_type == "python":
                # Python callable task
                function_path = task.get("function")
                module_path, function_name = function_path.rsplit('.', 1)

                # Dynamically import the function
                try:
                    module = __import__(module_path, fromlist=[function_name])
                    function = getattr(module, function_name)

                    @op(name=task_id)
                    def python_op():
                        return function(**(task.get("params", {})))

                    ops_dict[task_id] = python_op
                except Exception as e:
                    logger.error(f"Error importing function {function_path}: {str(e)}")
                    continue

        # Set up dependencies
        for task in tasks:
            task_id = task.get("id")
            task_deps = task.get("depends_on", [])

            if task_deps and task_id in ops_dict:
                dependencies[task_id] = {dep: "input_path" for dep in task_deps if dep in ops_dict}

        # Create the graph
        @graph(name=self.workflow_name)
        def workflow_graph():
            # Create op instances
            op_instances = {}
            for op_id, op_func in ops_dict.items():
                op_instances[op_id] = op_func()

            # Connect ops based on dependencies
            for op_id, deps in dependencies.items():
                for dep_id, dep_param in deps.items():
                    if dep_param == "input_path":
                        # If the parameter name is input_path, use the return value from the upstream op
                        op_instances[op_id] = ops_dict[op_id](input_path=op_instances[dep_id])

            # Return the final outputs
            return {op_id: instance for op_id, instance in op_instances.items()}

        self.workflow = workflow_graph
        return workflow_graph

    def execute_workflow(self, **kwargs) -> Any:
        """
        Execute the Dagster workflow.

        Args:
            **kwargs: Additional arguments

        Returns:
            Execution result
        """
        if not self.workflow:
            raise ValueError("Workflow not created. Call create_workflow first.")

        logger.info(f"Executing Dagster workflow '{self.workflow_name}'")

        # Create a job from the graph
        workflow_job = self.workflow.to_job(
            name=f"{self.workflow_name}_job",
            config=kwargs.get("config")
        )

        # Execute the job
        result = dagster.execute_job(
            workflow_job,
            run_config=kwargs.get("run_config")
        )

        return result


class WorkflowFactory:
    """Factory for creating workflow orchestrators"""

    @staticmethod
    def create_workflow(orchestrator: str, workflow_name: str,
                      config_path: Optional[str] = None) -> WorkflowBase:
        """
        Create a workflow orchestrator of the specified type.

        Args:
            orchestrator: Type of orchestrator (airflow, dagster)
            workflow_name: Name of the workflow
            config_path: Path to configuration file

        Returns:
            Initialized workflow orchestrator
        """
        if orchestrator.lower() == "airflow":
            return AirflowWorkflow(workflow_name, config_path)
        elif orchestrator.lower() == "dagster":
            return DagsterWorkflow(workflow_name, config_path)
        else:
            raise ValueError(f"Unknown orchestrator type: {orchestrator}")


# Example workflow configurations
FOREX_SCRAPING_WORKFLOW = [
    {
        "id": "scrape_forex_data",
        "type": "scrape",
        "scraper": "selenium",
        "urls": [
            "https://www.investing.com/currencies/eur-usd-historical-data",
            "https://www.investing.com/currencies/gbp-usd-historical-data",
            "https://www.investing.com/currencies/usd-jpy-historical-data"
        ],
        "output": "forex_data",
        "params": {
            "headless": True,
            "wait_for_selector": "table#curr_table"
        }
    },
    {
        "id": "process_forex_data",
        "type": "process",
        "processor": "forex",
        "input": "scraped_data/forex_data_latest.json",
        "output": "forex_processed",
        "depends_on": ["scrape_forex_data"],
        "operations": [
            {
                "type": "clean_text",
                "function": "clean_text",
                "columns": ["price", "open", "high", "low", "change"]
            },
            {
                "type": "extract_numbers",
                "function": "extract_numbers",
                "columns": ["price", "open", "high", "low", "change"]
            }
        ],
        "params": {
            "calculate_indicators": True
        }
    }
]

SPORTS_BETTING_WORKFLOW = [
    {
        "id": "scrape_football_data",
        "type": "scrape",
        "scraper": "selenium",
        "urls": [
            "https://www.oddsportal.com/soccer/england/premier-league/results/",
            "https://www.oddsportal.com/soccer/spain/laliga/results/",
            "https://www.oddsportal.com/soccer/germany/bundesliga/results/"
        ],
        "output": "football_data",
        "params": {
            "headless": True,
            "wait_for_selector": "table.table-main"
        }
    },
    {
        "id": "process_football_data",
        "type": "process",
        "processor": "sports",
        "input": "scraped_data/football_data_latest.json",
        "output": "football_processed",
        "depends_on": ["scrape_football_data"],
        "operations": [
            {
                "type": "extract_teams",
                "function": "clean_text",
                "columns": ["home_team", "away_team"]
            },
            {
                "type": "extract_numbers",
                "function": "extract_numbers",
                "columns": ["home_score", "away_score", "odds_home", "odds_draw", "odds_away"]
            }
        ],
        "params": {
            "sport_type": "football"
        }
    },
    {
        "id": "train_prediction_model",
        "type": "python",
        "function": "models.train_sports_model",
        "depends_on": ["process_football_data"],
        "params": {
            "data_path": "processed_data/football_processed_latest.csv",
            "model_type": "xgboost",
            "target": "home_win"
        }
    }
]
