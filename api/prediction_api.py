# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from flask import Flask, request, jsonify
from datetime import datetime

# Import core components
from core.prediction_engine import PredictionEngine
from data_pipeline.scraper.web_scraper import WebScraperManager

logger = logging.getLogger(__name__)

class PredictionAPI:
    """API interface for prediction system"""

    def __init__(self, config_path: Path = Path("config/config.yaml")):
        self.config_path = config_path

        # Initialize core components
        self.prediction_engine = PredictionEngine(config_path)

        # Initialize scraper with default config
        scraper_config = {
            'headless': True,
            'data_dir': 'data/scraped',
            'log_dir': 'logs/scraper',
            'request_delay': 2
        }
        self.scraper_manager = WebScraperManager(scraper_config)

        logger.info("API interface initialized")

    def handle_prediction_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request from UI"""
        try:
            domain = data.get("domain", "sports")

            # Get required parameters from request
            input_data = data.get("data", {})

            # Log request
            logger.info(f"Received prediction request for domain: {domain}")

            # Call prediction engine
            prediction = self.prediction_engine.predict(input_data)

            # Format response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction
            }

            return response

        except Exception as e:
            logger.error(f"Error handling prediction request: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def handle_scraping_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scraping request from UI"""
        try:
            # Extract parameters
            domain = data.get("domain", "all")
            target_date = data.get("date")

            logger.info(f"Received scraping request for domain: {domain}, date: {target_date}")

            # Call scraper manager
            result = self.scraper_manager.trigger_scrape(domain=domain, target_date=target_date)

            # If the result already has a status field, use it
            if "status" in result:
                return result

            # Otherwise, format the response properly
            return {
                "status": "success" if result.get("success", True) else "failed",
                "message": result.get("message", f"Scraping initiated for {domain}"),
                "details": result.get("details", {}),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error handling scraping request: {str(e)}")
            return {
                "status": "failed",
                "error": f"Scraping failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def handle_retraining_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model retraining request from UI"""
        try:
            # Extract parameters
            domain = data.get("domain", "all")
            custom_data = data.get("data")

            logger.info(f"Received retraining request for domain: {domain}")

            # Call prediction engine's retrain method
            result = self.prediction_engine.retrain(domain=domain, data=custom_data)

            return result

        except Exception as e:
            logger.error(f"Error handling retraining request: {str(e)}")
            return {
                "status": "failed",
                "error": f"Retraining failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def handle_query_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle natural language query request from UI"""
        try:
            # Extract parameters
            query = data.get("query", "")
            context = data.get("context", {})

            if not query:
                return {
                    "error": "No query provided",
                    "timestamp": datetime.now().isoformat()
                }

            logger.info(f"Received NLP query: {query}")

            # Process through prediction engine
            try:
                # Check if prediction engine has NLP processor
                if hasattr(self.prediction_engine, 'process_nlp_query'):
                    result = self.prediction_engine.process_nlp_query(query, context)
                    return result
                else:
                    # Fallback if no NLP processor available
                    return {
                        "query": query,
                        "intent": "general_query",
                        "confidence": 0.5,
                        "response": "I can help with predictions. Try asking about forex or sports predictions.",
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                return {
                    "error": f"Error processing query: {str(e)}",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error handling query request: {str(e)}")
            return {
                "error": f"Query failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Flask application setup
app = Flask(__name__)
api_instance = PredictionAPI()

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    data = request.json
    response = api_instance.handle_prediction_request(data)
    return jsonify(response)

@app.route('/api/scrape', methods=['POST'])
def scrape():
    """API endpoint for triggering scraping"""
    data = request.json
    response = api_instance.handle_scraping_request(data)
    return jsonify(response)

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """API endpoint for retraining models"""
    data = request.json
    response = api_instance.handle_retraining_request(data)
    return jsonify(response)

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint for natural language queries"""
    data = request.json
    response = api_instance.handle_query_request(data)
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    app.run(debug=True, host='0.0.0.0', port=5000)
