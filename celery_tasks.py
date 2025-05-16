#!/usr/bin/env python3
"""
Celery Tasks Configuration

This module configures Celery for handling asynchronous tasks like
web scraping, model retraining, and data processing.
"""

import os
import logging
from celery import Celery
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("celery_tasks")

# Create Celery app - in production, use Redis or RabbitMQ as broker
celery_app = Celery(
    'super_ai',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,

    # Task result settings
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,

    # Task time limits
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes

    # Concurrency settings
    worker_concurrency=os.cpu_count() or 4,
)

# Task status tracking
task_status = {}

@celery_app.task(bind=True, name='scrape_data')
def scrape_data(self, domain: str = 'forex', urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Asynchronous task to scrape data from various sources

    Args:
        domain: The domain to scrape data for ('forex' or 'sports')
        urls: Optional list of specific URLs to scrape

    Returns:
        Dictionary with scraping results
    """
    task_id = self.request.id

    # Update task status
    task_status[task_id] = {
        'status': 'RUNNING',
        'progress': 0,
        'start_time': datetime.now().isoformat(),
        'domain': domain,
        'urls': urls
    }

    try:
        logger.info(f"Starting scraping task for domain: {domain}")

        # Dynamically import scraper to avoid circular imports
        from web_scraper import WebScraper
        scraper = WebScraper()

        # Track results
        results = []

        # Scrape based on domain or URLs
        if urls:
            total_urls = len(urls)
            for i, url in enumerate(urls, 1):
                # Update progress
                progress = int((i / total_urls) * 100)
                task_status[task_id]['progress'] = progress
                self.update_state(state='PROGRESS', meta={'progress': progress})

                # Scrape URL
                try:
                    result = scraper.scrape_url(url)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {str(e)}")
                    results.append({'url': url, 'error': str(e)})

                # Small delay between requests
                time.sleep(1)
        else:
            # Scrape by domain
            if domain == 'forex':
                data = scraper.scrape_forex_data(save=True)
                results = {'forex_records': len(data)}
            elif domain == 'sports':
                data = scraper.scrape_sports_data('Soccer', save=True)
                results = {'sports_records': len(data)}
            else:
                raise ValueError(f"Unknown domain: {domain}")

        # Update task status
        task_status[task_id].update({
            'status': 'COMPLETED',
            'progress': 100,
            'end_time': datetime.now().isoformat(),
            'results': results
        })

        logger.info(f"Scraping task completed for domain: {domain}")
        return {
            'status': 'success',
            'domain': domain,
            'results': results
        }

    except Exception as e:
        error_msg = f"Error in scraping task: {str(e)}"
        logger.error(error_msg)

        # Update task status
        task_status[task_id].update({
            'status': 'FAILED',
            'error': error_msg,
            'end_time': datetime.now().isoformat()
        })

        # Re-raise for Celery to handle
        raise

@celery_app.task(bind=True, name='retrain_model')
def retrain_model(self, model_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Asynchronous task to retrain prediction models

    Args:
        model_type: Type of model to retrain ('forex', 'sports', 'combined')
        parameters: Optional parameters for training

    Returns:
        Dictionary with training results
    """
    task_id = self.request.id

    # Update task status
    task_status[task_id] = {
        'status': 'RUNNING',
        'progress': 0,
        'start_time': datetime.now().isoformat(),
        'model_type': model_type
    }

    try:
        logger.info(f"Starting model retraining task for model: {model_type}")

        # Here we would import and call the actual model training code
        # For now, we'll simulate training

        # Simulate training phases
        training_phases = ['data_loading', 'preprocessing', 'training', 'validation', 'saving']

        for i, phase in enumerate(training_phases):
            # Update progress
            progress = int(((i + 1) / len(training_phases)) * 100)
            task_status[task_id]['progress'] = progress
            task_status[task_id]['current_phase'] = phase
            self.update_state(state='PROGRESS', meta={
                'progress': progress,
                'phase': phase
            })

            # Simulate work
            logger.info(f"Model training phase: {phase}")
            time.sleep(5)  # Simulate work for this phase

        # Update task status
        task_status[task_id].update({
            'status': 'COMPLETED',
            'progress': 100,
            'end_time': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.87,
                'f1_score': 0.88
            }
        })

        logger.info(f"Model retraining completed for model: {model_type}")
        return {
            'status': 'success',
            'model_type': model_type,
            'metrics': task_status[task_id].get('metrics')
        }

    except Exception as e:
        error_msg = f"Error in model retraining task: {str(e)}"
        logger.error(error_msg)

        # Update task status
        task_status[task_id].update({
            'status': 'FAILED',
            'error': error_msg,
            'end_time': datetime.now().isoformat()
        })

        # Re-raise for Celery to handle
        raise

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a task by ID"""
    return task_status.get(task_id, {'status': 'UNKNOWN'})

def get_all_tasks() -> Dict[str, Dict[str, Any]]:
    """Get the status of all tasks"""
    return task_status

def clear_completed_tasks() -> int:
    """Clear completed tasks and return count of cleared tasks"""
    global task_status
    completed_tasks = [
        task_id for task_id, status in task_status.items()
        if status.get('status') in ['COMPLETED', 'FAILED']
    ]

    for task_id in completed_tasks:
        task_status.pop(task_id, None)

    return len(completed_tasks)

# Register Flask routes for Celery tasks
def register_celery_routes(app):
    """Register Celery task routes with Flask app"""
    from flask import jsonify, request

    # Start scraping task
    @app.route('/api/scraping/start', methods=['POST'])
    def start_scraping():
        data = request.json or {}
        domain = data.get('domain', 'forex')
        urls = data.get('urls')

        task = scrape_data.delay(domain=domain, urls=urls)

        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'message': f'Scraping task started for domain: {domain}'
        })

    # Start model retraining
    @app.route('/api/model/retrain', methods=['POST'])
    def start_retraining():
        data = request.json or {}
        model_type = data.get('model_type', 'forex')
        parameters = data.get('parameters', {})

        task = retrain_model.delay(model_type=model_type, parameters=parameters)

        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'message': f'Retraining task started for model: {model_type}'
        })

    # Get task status
    @app.route('/api/task/<task_id>', methods=['GET'])
    def get_task(task_id):
        status = get_task_status(task_id)
        return jsonify(status)

    # Get all tasks
    @app.route('/api/tasks', methods=['GET'])
    def get_tasks():
        tasks = get_all_tasks()
        return jsonify(tasks)

    # Clear completed tasks
    @app.route('/api/tasks/clear', methods=['POST'])
    def clear_tasks():
        count = clear_completed_tasks()
        return jsonify({
            'status': 'success',
            'cleared_count': count
        })

if __name__ == "__main__":
    # This allows the file to be imported by Flask app but also run directly
    # for testing or starting a Celery worker
    print("Starting Celery worker...")

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
    celery_app.start()
