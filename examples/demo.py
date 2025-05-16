#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Super AI Prediction System - Data Pipeline Demo

This script demonstrates how to use the different components of the data pipeline:
- Web scrapers
- Data processors
- Workflow orchestration

It provides examples for both forex and sports betting domains.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import the data_pipeline package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(parent_dir))

# Import data pipeline components
from data_pipeline.scrapers import ScraperFactory
from data_pipeline.processors import ProcessorFactory
from data_pipeline.workflows import WorkflowFactory, FOREX_SCRAPING_WORKFLOW, SPORTS_BETTING_WORKFLOW

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(parent_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def setup_directories(config):
    """Create necessary directories based on configuration"""
    dirs = [
        config.get("general", {}).get("data_dir", "data"),
        config.get("general", {}).get("scraped_dir", "scraped_data"),
        config.get("general", {}).get("processed_dir", "processed_data"),
        config.get("general", {}).get("models_dir", "models"),
        config.get("general", {}).get("logs_dir", "logs"),
        config.get("general", {}).get("temp_dir", "temp")
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def demo_forex_scraping():
    """Demonstrate forex data scraping"""
    logger.info("=== Forex Data Scraping Demo ===")

    try:
        # Create a Selenium scraper for JavaScript-rendered pages
        scraper = ScraperFactory.create_scraper("selenium", headless=True)

        # URLs to scrape
        urls = [
            "https://www.investing.com/currencies/eur-usd-historical-data",
            "https://www.investing.com/currencies/gbp-usd-historical-data"
        ]

        for url in urls:
            logger.info(f"Scraping {url}")
            scraper.scrape(
                url=url,
                wait_for_selector="table#curr_table",
                selector="table#curr_table tr"
            )

        # Save results
        output_path = scraper.save_results("demo_forex_data")
        logger.info(f"Forex data saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error in forex scraping demo: {str(e)}")
        return None

def demo_forex_processing(input_file):
    """Demonstrate forex data processing"""
    logger.info("=== Forex Data Processing Demo ===")

    if not input_file or not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found")
        return None

    try:
        # Create a forex data processor
        processor = ProcessorFactory.create_processor("forex")

        # Define operations
        operations = [
            {
                "type": "clean_text",
                "columns": ["price", "open", "high", "low", "change"]
            },
            {
                "type": "extract_numbers",
                "columns": ["price", "open", "high", "low", "change"]
            }
        ]

        # Process data
        result = processor.process(
            data=input_file,
            input_format="json",
            operations=operations,
            calculate_indicators=True
        )

        # Save processed data
        output_path = processor.save_results(result, "demo_forex_processed", "csv")
        logger.info(f"Processed forex data saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error in forex processing demo: {str(e)}")
        return None

def demo_sports_scraping():
    """Demonstrate sports data scraping"""
    logger.info("=== Sports Data Scraping Demo ===")

    try:
        # Create a Selenium scraper for JavaScript-rendered pages
        scraper = ScraperFactory.create_scraper("selenium", headless=True)

        # URLs to scrape
        urls = [
            "https://www.oddsportal.com/soccer/england/premier-league/results/"
        ]

        for url in urls:
            logger.info(f"Scraping {url}")
            scraper.scrape(
                url=url,
                wait_for_selector="table.table-main",
                selector="table.table-main tr.deactivate"
            )

        # Save results
        output_path = scraper.save_results("demo_sports_data")
        logger.info(f"Sports data saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error in sports scraping demo: {str(e)}")
        return None

def demo_sports_processing(input_file):
    """Demonstrate sports data processing"""
    logger.info("=== Sports Data Processing Demo ===")

    if not input_file or not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found")
        return None

    try:
        # Create a sports data processor
        processor = ProcessorFactory.create_processor("sports")

        # Define operations
        operations = [
            {
                "type": "extract_teams",
                "columns": ["text"]
            },
            {
                "type": "extract_numbers",
                "columns": ["text"]
            }
        ]

        # Process data
        result = processor.process(
            data=input_file,
            input_format="json",
            operations=operations,
            sport_type="football"
        )

        # Save processed data
        output_path = processor.save_results(result, "demo_sports_processed", "csv")
        logger.info(f"Processed sports data saved to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error in sports processing demo: {str(e)}")
        return None

def demo_workflow():
    """Demonstrate workflow orchestration"""
    logger.info("=== Workflow Orchestration Demo ===")

    try:
        # Create a workflow using Dagster
        workflow = WorkflowFactory.create_workflow("dagster", "demo_workflow")

        # Create a simple workflow
        demo_workflow = [
            {
                "id": "demo_scrape",
                "type": "scrape",
                "scraper": "selenium",
                "urls": ["https://www.investing.com/currencies/eur-usd-historical-data"],
                "output": "demo_workflow_data",
                "params": {
                    "headless": True,
                    "wait_for_selector": "table#curr_table"
                }
            },
            {
                "id": "demo_process",
                "type": "process",
                "processor": "forex",
                "input": "scraped_data/demo_workflow_data_latest.json",
                "output": "demo_workflow_processed",
                "depends_on": ["demo_scrape"],
                "operations": [
                    {
                        "type": "clean_text",
                        "columns": ["price", "open", "high", "low", "change"]
                    }
                ]
            }
        ]

        # Set up the workflow
        workflow.create_workflow(demo_workflow)

        # Execute the workflow
        logger.info("Executing demo workflow")
        workflow.execute_workflow()

        logger.info("Workflow execution complete")

    except Exception as e:
        logger.error(f"Error in workflow demo: {str(e)}")

def run_all_demos():
    """Run all demos in sequence"""
    # Load configuration
    config = load_config()

    # Set up directories
    setup_directories(config)

    # Demo forex scraping
    forex_data = demo_forex_scraping()

    # Demo forex processing
    if forex_data:
        demo_forex_processing(forex_data)

    # Demo sports scraping
    sports_data = demo_sports_scraping()

    # Demo sports processing
    if sports_data:
        demo_sports_processing(sports_data)

    # Demo workflow orchestration
    demo_workflow()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Data Pipeline Demo")
    parser.add_argument("--demo", choices=["forex", "sports", "workflow", "all"],
                      default="all", help="Demo to run")
    args = parser.parse_args()

    if args.demo == "forex":
        forex_data = demo_forex_scraping()
        if forex_data:
            demo_forex_processing(forex_data)

    elif args.demo == "sports":
        sports_data = demo_sports_scraping()
        if sports_data:
            demo_sports_processing(sports_data)

    elif args.demo == "workflow":
        demo_workflow()

    else:  # all
        run_all_demos()

if __name__ == "__main__":
    main()