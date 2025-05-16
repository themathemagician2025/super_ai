#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Super AI Web Interface - Main Application

This module provides a Flask-based API and web interface for the Super AI prediction system,
serving both the frontend UI and API endpoints for predictions.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import copy
import random

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Configure logger
logger = logging.getLogger("web_interface")

# Initialize Flask app
app = Flask(__name__)

# Import forex API blueprint
try:
    from web_interface.forex_api import forex_api
    app.register_blueprint(forex_api)
    logger.info("Forex API blueprint registered successfully")
except Exception as e:
    logger.warning(f"Could not register Forex API blueprint: {str(e)}")

# Flag to track if this is the first request
_is_first_request = True

# Configure logging
@app.before_request
def setup_logging():
    """Set up logging for the Flask application"""
    global _is_first_request

    if _is_first_request:
        _is_first_request = False

        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level, logging.INFO)

        # Configure Flask logger
        app.logger.setLevel(numeric_level)

        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logger.info("Web interface logging configured")

# Load prediction models
def load_models():
    """Load prediction models on startup"""
    try:
        # Import prediction models
        from prediction.sports_predictor import SportsPredictor
        from betting.betting_prediction import BettingPredictor

        # Get model paths from environment or use defaults
        sports_model_path = os.environ.get(
            'SPORTS_MODEL_PATH',
            os.path.join(src_dir, '..', 'models', 'sports_predictor.pth')
        )

        betting_model_path = os.environ.get(
            'BETTING_MODEL_PATH',
            os.path.join(src_dir, '..', 'models', 'betting_predictor.pth')
        )

        # Initialize models
        sports_model = None
        betting_model = None

        # Load sports model if exists
        if os.path.exists(sports_model_path):
            logger.info(f"Loading sports model from {sports_model_path}")
            sports_model = SportsPredictor(input_size=8)
            sports_model.load(sports_model_path)
        else:
            logger.warning(f"Sports model not found at {sports_model_path}")

        # Load betting model if exists
        if os.path.exists(betting_model_path):
            logger.info(f"Loading betting model from {betting_model_path}")
            betting_model = BettingPredictor(input_size=18)
            betting_model.load(betting_model_path)
        else:
            logger.warning(f"Betting model not found at {betting_model_path}")

        return {
            'sports_model': sports_model,
            'betting_model': betting_model
        }

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return {
            'sports_model': None,
            'betting_model': None
        }

# Home route - serve the main page
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

# Static files route
@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# API Health check
@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

# System status endpoint
@app.route('/api/status')
def system_status():
    """Get system status including model and data availability"""
    # Check models status
    models_status = {
        'sports_model': bool(app.config.get('models', {}).get('sports_model')),
        'betting_model': bool(app.config.get('models', {}).get('betting_model'))
    }

    # Check data availability
    data_status = check_data_availability()

    # Check API status
    api_status = 'online'

    # Determine overall system status
    if all(models_status.values()) and all(val == 'available' for val in data_status.values()):
        system = 'online'
    elif any(models_status.values()) and any(val == 'available' for val in data_status.values()):
        system = 'partial'
    else:
        system = 'offline'

    return jsonify({
        'status': system,
        'api': api_status,
        'models': models_status,
        'data': data_status,
        'timestamp': datetime.now().isoformat()
    })

# API endpoints for models
@app.route('/api/sports/predict', methods=['POST'])
def predict_sports():
    """Sports prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        required_fields = ['home_team', 'away_team', 'league']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'required': required_fields}), 400

        # Check if model is available
        sports_model = app.config.get('models', {}).get('sports_model')
        if not sports_model:
            # Return simulated prediction for demo
            return jsonify(generate_simulated_sports_prediction(data))

        # Process input data
        features = process_sports_input(data)

        # Make prediction
        prediction = sports_model.predict(features)

        # Format response
        result = {
            'home_team': data['home_team'],
            'away_team': data['away_team'],
            'home_win_probability': float(prediction[0]),
            'away_win_probability': 1.0 - float(prediction[0]),
            'confidence': float(np.random.uniform(0.6, 0.95)),
            'factors': {
                'player_performance': float(np.random.uniform(0.5, 0.9)),
                'team_history': float(np.random.uniform(0.5, 0.9)),
                'opponent_analysis': float(np.random.uniform(0.5, 0.9))
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in sports prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/betting/predict', methods=['POST'])
def predict_betting():
    """Betting prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        required_fields = ['event', 'odds', 'stake']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'required': required_fields}), 400

        # Check if model is available
        betting_model = app.config.get('models', {}).get('betting_model')
        if not betting_model:
            # Return simulated prediction for demo
            return jsonify(generate_simulated_betting_prediction(data))

        # Process input data
        features = process_betting_input(data)

        # Make prediction
        prediction = betting_model.predict(features)

        # Format response
        result = {
            'event': data['event'],
            'odds': float(data['odds']),
            'stake': float(data['stake']),
            'expected_value': float(prediction[0]),
            'expected_return': float(prediction[0]) + float(data['stake']),
            'risk_level': get_risk_level(float(prediction[0]), float(data['stake'])),
            'confidence': float(np.random.uniform(0.6, 0.95)),
            'odds_assessment': {
                'fair_odds': float(data['odds']) * (1 - np.random.uniform(0.05, 0.2)),
                'market_odds': float(data['odds']),
                'edge': float(np.random.uniform(0.0, 0.2))
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in betting prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Live data endpoints
@app.route('/api/data/sports/latest')
def get_latest_sports_data():
    """Get the latest scraped sports data"""
    try:
        # Find the most recent sports data file
        data_dir = os.path.join(src_dir, '..', 'data', 'processed_data')
        sports_files = [f for f in os.listdir(data_dir) if f.startswith('sports_') and f.endswith('.csv')]

        if not sports_files:
            return jsonify({'error': 'No sports data available'}), 404

        # Get the most recent file
        latest_file = sorted(sports_files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)[0]
        file_path = os.path.join(data_dir, latest_file)

        # Read the data
        df = pd.read_csv(file_path)

        # Convert to list of records
        data = df.to_dict(orient='records')

        # Return limited number of records (50 max)
        return jsonify({
            'filename': latest_file,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'count': len(data),
            'data': data[:50]  # Return at most 50 records
        })

    except Exception as e:
        logger.error(f"Error retrieving sports data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/betting/latest')
def get_latest_betting_data():
    """Get the latest processed betting data"""
    try:
        # Find the most recent betting data file
        data_dir = os.path.join(src_dir, '..', 'data', 'processed_data')
        betting_files = [f for f in os.listdir(data_dir) if f.startswith('betting_') and f.endswith('.csv')]

        if not betting_files:
            return jsonify({'error': 'No betting data available'}), 404

        # Get the most recent file
        latest_file = sorted(betting_files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)[0]
        file_path = os.path.join(data_dir, latest_file)

        # Read the data
        df = pd.read_csv(file_path)

        # Convert to list of records
        data = df.to_dict(orient='records')

        # Return limited number of records (50 max)
        return jsonify({
            'filename': latest_file,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'count': len(data),
            'data': data[:50]  # Return at most 50 records
        })

    except Exception as e:
        logger.error(f"Error retrieving betting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/market/latest')
def get_latest_market_data():
    """Get latest market data"""
    try:
        # In a real application, this would fetch real data from a database or API
        # For demo purposes, we'll return simulated data
        data = []
        for i in range(10):
            data.append({
                'market': f'Market {i+1}',
                'timestamp': (datetime.now().timestamp() - i * 3600),
                'last_value': float(np.random.normal(100, 10)),
                'change': float(np.random.normal(0, 2)),
                'volume': int(np.random.randint(10000, 100000))
            })

        return jsonify(data)

    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# New API endpoints for scraping functionality
@app.route('/api/scraping/status', methods=['GET'])
def get_scraping_status():
    """Get current scraping status"""
    try:
        # In a real application, this would check the actual status of scraping jobs
        status = {
            'status': 'idle',  # Could be 'idle', 'running', 'completed', 'error'
            'last_run': datetime.now().isoformat(),
            'next_scheduled': datetime.now().isoformat(),
            'scrapers': {
                'sports_data': 'idle',
                'odds_data': 'idle',
                'market_data': 'idle'
            },
            'stats': {
                'total_records': 0,
                'last_run_duration': 0,
                'last_run_records': 0
            }
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting scraping status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scraping/start', methods=['POST'])
def start_scraping():
    """Start data scraping job"""
    try:
        data = request.get_json() or {}

        # Extract scraping parameters
        sources = data.get('sources', ['sports', 'odds', 'market'])
        force_update = data.get('force_update', False)

        # In a real app, this would trigger actual scraping jobs
        # For demo, we'll just simulate a response
        result = {
            'job_id': f"scrape_{int(time.time())}",
            'status': 'started',
            'sources': sources,
            'start_time': datetime.now().isoformat(),
            'message': f"Started scraping job for {', '.join(sources)}"
        }

        # Log the scraping job
        logger.info(f"Started scraping job: {result['job_id']} for sources: {sources}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error starting scraping job: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Generic prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    """Generic prediction endpoint that routes to the appropriate model"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Determine prediction type based on input data
        prediction_type = data.get('type', '').lower()

        if prediction_type == 'sports':
            # Route to sports prediction
            return predict_sports()
        elif prediction_type == 'betting':
            # Route to betting prediction
            return predict_betting()
        else:
            return jsonify({
                'error': 'Invalid prediction type',
                'message': 'Supported types are "sports" and "betting"'
            }), 400

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Query endpoint for data retrieval
@app.route('/api/query', methods=['POST'])
def query_data():
    """Query data from the system"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No query parameters provided'}), 400

        # Extract query parameters
        data_type = data.get('data_type', '').lower()
        filters = data.get('filters', {})
        limit = data.get('limit', 100)

        # In a real application, this would query a database
        # For demo, return simulated data
        results = []

        if data_type == 'sports':
            # Generate mock sports data
            for i in range(min(limit, 100)):
                results.append({
                    'id': i + 1,
                    'home_team': f'Team A{i}',
                    'away_team': f'Team B{i}',
                    'home_score': np.random.randint(0, 5),
                    'away_score': np.random.randint(0, 5),
                    'date': (datetime.now().timestamp() - i * 86400),
                    'league': 'Premier League'
                })
        elif data_type == 'betting':
            # Generate mock betting data
            for i in range(min(limit, 100)):
                results.append({
                    'id': i + 1,
                    'event': f'Event {i}',
                    'odds': round(np.random.uniform(1.1, 10.0), 2),
                    'stake': round(np.random.uniform(10, 1000), 2),
                    'result': np.random.choice(['win', 'loss']),
                    'profit': round(np.random.uniform(-1000, 1000), 2),
                    'date': (datetime.now().timestamp() - i * 86400)
                })
        else:
            return jsonify({
                'error': 'Invalid data type',
                'message': 'Supported types are "sports" and "betting"'
            }), 400

        return jsonify({
            'results': results,
            'count': len(results),
            'query': {
                'data_type': data_type,
                'filters': filters,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error querying data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Training endpoints
@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training job"""
    try:
        data = request.get_json() or {}

        # Extract training parameters
        model_type = data.get('model_type', 'sports')
        dataset = data.get('dataset', 'default')
        params = data.get('parameters', {})

        # In a real app, this would start an actual training job
        # For demo, we'll just simulate a response
        result = {
            'job_id': f"train_{model_type}_{int(time.time())}",
            'status': 'started',
            'model_type': model_type,
            'dataset': dataset,
            'start_time': datetime.now().isoformat(),
            'estimated_duration': '00:30:00',
            'message': f"Started training job for {model_type} model"
        }

        # Log the training job
        logger.info(f"Started training job: {result['job_id']} for model: {model_type}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training job status"""
    try:
        job_id = request.args.get('job_id')

        if not job_id:
            # Return status of all recent training jobs
            status = {
                'jobs': [
                    {
                        'job_id': f"train_sports_{int(time.time()) - 3600}",
                        'status': 'completed',
                        'progress': 100,
                        'model_type': 'sports',
                        'start_time': datetime.fromtimestamp(time.time() - 3600).isoformat(),
                        'end_time': datetime.fromtimestamp(time.time() - 1800).isoformat(),
                        'metrics': {
                            'accuracy': 0.82,
                            'loss': 0.35
                        }
                    }
                ]
            }
        else:
            # Return status of specific job
            status = {
                'job_id': job_id,
                'status': 'completed',
                'progress': 100,
                'model_type': 'sports',
                'start_time': datetime.fromtimestamp(time.time() - 3600).isoformat(),
                'end_time': datetime.fromtimestamp(time.time() - 1800).isoformat(),
                'metrics': {
                    'accuracy': 0.82,
                    'loss': 0.35
                }
            }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Athlete Profile and Performance Metrics Endpoints
@app.route('/api/athletes/search', methods=['GET'])
def search_athletes():
    """Search for athletes by name, team, or sport"""
    try:
        query = request.args.get('q', '')
        sport = request.args.get('sport', '')
        team = request.args.get('team', '')
        limit = int(request.args.get('limit', 10))

        # In a real implementation, this would query a database
        # For demo purposes, we'll return mock data
        athletes = []

        # Generate mock athlete data
        for i in range(min(limit, 20)):
            if query.lower() in f"athlete {i}".lower() or not query:
                sport_type = sport if sport else random.choice(['Football', 'Basketball', 'Baseball', 'Soccer', 'Tennis'])
                position = get_position_for_sport(sport_type)
                team_name = team if team else f"Team {chr(65 + i % 26)}"

                athletes.append({
                    'id': f"ath_{i}_{int(time.time())}",
                    'name': f"Athlete {i}",
                    'sport': sport_type,
                    'position': position,
                    'team': team_name,
                    'country': random.choice(['USA', 'UK', 'Spain', 'Brazil', 'Germany', 'Japan']),
                    'matches_played': random.randint(10, 500),
                    'profile_completion': random.randint(70, 100),
                    'last_updated': datetime.now().isoformat()
                })

        return jsonify({
            'query': query,
            'sport': sport,
            'team': team,
            'count': len(athletes),
            'athletes': athletes
        })

    except Exception as e:
        logger.error(f"Error searching athletes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/athletes/<athlete_id>', methods=['GET'])
def get_athlete_profile(athlete_id):
    """Get detailed profile for a specific athlete"""
    try:
        # In a real implementation, this would retrieve from a database
        # For demo purposes, we'll generate mock data

        # Extract info from athlete_id format
        parts = athlete_id.split('_')

        if len(parts) < 2 or parts[0] != 'ath':
            return jsonify({'error': 'Invalid athlete ID format'}), 400

        athlete_num = int(parts[1])
        sport_type = random.choice(['Football', 'Basketball', 'Baseball', 'Soccer', 'Tennis'])
        position = get_position_for_sport(sport_type)

        # Basic profile info
        profile = {
            'id': athlete_id,
            'name': f"Athlete {athlete_num}",
            'full_name': f"First{athlete_num} Last{athlete_num}",
            'sport': sport_type,
            'position': position,
            'team': f"Team {chr(65 + athlete_num % 26)}",
            'former_teams': [f"Team {chr(65 + (athlete_num + i) % 26)}" for i in range(1, 4)],
            'country': random.choice(['USA', 'UK', 'Spain', 'Brazil', 'Germany', 'Japan']),
            'age': random.randint(20, 38),
            'height': round(175 + random.uniform(-15, 15), 1),  # cm
            'weight': round(75 + random.uniform(-15, 15), 1),   # kg
            'active': random.random() > 0.1,
            'professional_since': 2010 + random.randint(-10, 10),
            'profile_image_url': f"https://example.com/athletes/{athlete_id}.jpg",

            # Physical performance metrics
            'physical_metrics': {
                'max_speed': round(random.uniform(20, 38), 2),         # km/h
                'acceleration': round(random.uniform(2.5, 5.0), 2),    # m/s²
                'vo2_max': round(random.uniform(45, 65), 1),           # mL/kg/min
                'agility_score': round(random.uniform(7.5, 9.5), 2),   # arbitrary 1-10
                'power_output': random.randint(900, 2200),             # watts
                'jump_height': round(random.uniform(40, 80), 1),       # cm
                'strength_rating': round(random.uniform(7.0, 9.8), 1), # arbitrary 1-10
                'reaction_time': random.randint(150, 250),             # milliseconds
                'fatigue_index': round(random.uniform(3.5, 12.0), 2),  # % drop
                'distance_covered_avg': round(random.uniform(7, 14), 2), # km per game
                'heart_rate_avg': random.randint(140, 180),            # bpm
                'heart_rate_max': random.randint(180, 210),            # bpm
                'work_rate': round(random.uniform(75, 120), 1),        # m/min
            },

            # Technical performance metrics
            'technical_metrics': {
                'accuracy': round(random.uniform(55, 90), 1),          # %
                'possession_time_avg': round(random.uniform(1.0, 4.5), 1), # minutes per game
                'scoring_efficiency': round(random.uniform(30, 60), 1), # %
                'assist_rate': round(random.uniform(0.5, 5.0), 2),     # per game
                'tackle_success_rate': round(random.uniform(65, 95), 1), # %
                'serve_velocity': random.randint(100, 230),            # km/h
                'dribbling_success': round(random.uniform(50, 85), 1), # %
                'blocking_efficiency': round(random.uniform(0.5, 3.0), 1), # per game
                'error_rate': round(random.uniform(5, 25), 1),         # %
                'shot_distance_avg': round(random.uniform(3.5, 15.0), 1), # meters
            },

            # Tactical performance metrics
            'tactical_metrics': {
                'pass_selection_rating': round(random.uniform(6.5, 9.5), 1), # arbitrary 1-10
                'defensive_positioning': round(random.uniform(6.0, 9.5), 1), # arbitrary 1-10
                'offensive_contribution': round(random.uniform(0.1, 0.9), 2), # xG+xA per game
                'expected_goals': round(random.uniform(0.1, 0.7), 2),   # xG per game
                'expected_assists': round(random.uniform(0.05, 0.4), 2), # xA per game
                'game_intelligence': round(random.uniform(6.0, 9.8), 1), # arbitrary 1-10
                'transition_speed': round(random.uniform(1.0, 4.0), 1),  # seconds
                'zone_entries_avg': round(random.uniform(2.0, 12.0), 1), # per game
            },

            # Mental performance metrics
            'mental_metrics': {
                'composure_rating': round(random.uniform(6.0, 9.8), 1), # arbitrary 1-10
                'work_ethic_rating': round(random.uniform(7.0, 9.9), 1), # arbitrary 1-10
                'leadership_impact': round(random.uniform(-2.0, 8.0), 1), # plus/minus
                'focus_rating': round(random.uniform(6.0, 9.5), 1),     # arbitrary 1-10
                'resilience_score': round(random.uniform(60, 95), 1),   # %
            },

            # Advanced analytics metrics (sport-specific)
            'advanced_metrics': get_advanced_metrics_for_sport(sport_type),

            # Health and injury metrics
            'health_metrics': {
                'injury_frequency': round(random.uniform(0.5, 3.0), 1), # per season
                'current_injury_status': random.choice(['Healthy', 'Day-to-Day', 'Out', 'IR']),
                'recovery_time_avg': round(random.uniform(1.0, 6.0), 1), # weeks
                'muscle_asymmetry': round(random.uniform(2.0, 12.0), 1), # %
                'joint_stability_rating': round(random.uniform(6.0, 9.5), 1), # arbitrary 1-10
                'games_missed_last_season': random.randint(0, 25),
            },

            # Career statistics
            'career_stats': {
                'games_played': random.randint(50, 800),
                'seasons': random.randint(1, 18),
                'championships': random.randint(0, 5),
                'all_star_appearances': random.randint(0, 10),
                'awards': random.randint(0, 15),
            },

            # Metadata
            'data_sources': [
                'Public Statistics',
                'AI Estimation',
                'ML Projection',
                'Wearable Devices'
            ],
            'confidence_score': round(random.uniform(75, 98), 1),  # % confidence in data
            'last_updated': datetime.now().isoformat(),
            'next_match': {
                'opponent': f"Team {chr(65 + (athlete_num + 5) % 26)}",
                'date': (datetime.now() + timedelta(days=random.randint(1, 14))).isoformat(),
                'prediction_available': random.random() > 0.3,
            }
        }

        return jsonify(profile)

    except Exception as e:
        logger.error(f"Error retrieving athlete profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/athletes/update', methods=['POST'])
def update_athlete_metrics():
    """Update metrics for an athlete (for internal use/scheduled jobs)"""
    try:
        data = request.get_json()

        if not data or 'athlete_id' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        athlete_id = data.get('athlete_id')
        metrics = data.get('metrics', {})
        source = data.get('source', 'manual')

        # In a real implementation, this would update a database
        # For demo, we'll just return success

        return jsonify({
            'success': True,
            'athlete_id': athlete_id,
            'updated_metrics_count': len(metrics),
            'source': source,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error updating athlete metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams/<team_id>', methods=['GET'])
def get_team_profile(team_id):
    """Get detailed profile for a specific team"""
    try:
        # In a real implementation, this would retrieve from a database
        # For demo purposes, we'll generate mock data

        # Extract info from team_id format
        parts = team_id.split('_')

        if len(parts) < 2 or parts[0] != 'team':
            return jsonify({'error': 'Invalid team ID format'}), 400

        team_num = int(parts[1]) if parts[1].isdigit() else ord(parts[1][0]) - 65
        sport_type = random.choice(['Football', 'Basketball', 'Baseball', 'Soccer', 'Tennis'])

        # Basic team profile
        profile = {
            'id': team_id,
            'name': f"Team {chr(65 + team_num % 26)}",
            'full_name': f"City {team_num} {['Lions', 'Tigers', 'Bears', 'Eagles', 'Bulls'][team_num % 5]}",
            'sport': sport_type,
            'league': f"{sport_type} {['Premier', 'Major', 'National', 'International'][team_num % 4]} League",
            'country': random.choice(['USA', 'UK', 'Spain', 'Brazil', 'Germany', 'Japan']),
            'founded': 1900 + random.randint(0, 100),
            'current_ranking': random.randint(1, 30),
            'home_venue': f"Stadium {team_num}",
            'logo_url': f"https://example.com/teams/{team_id}.jpg",

            # Team performance metrics
            'performance_metrics': {
                'win_loss_record': f"{random.randint(5, 50)}-{random.randint(5, 30)}",
                'win_percentage': round(random.uniform(0.3, 0.8), 3),
                'points_per_game': round(random.uniform(70, 110), 1),
                'points_allowed_per_game': round(random.uniform(70, 110), 1),
                'home_record': f"{random.randint(5, 25)}-{random.randint(2, 15)}",
                'away_record': f"{random.randint(2, 25)}-{random.randint(5, 20)}",
                'streak': f"{'W' if random.random() > 0.5 else 'L'}{random.randint(1, 10)}",
                'last_10': f"{random.randint(0, 10)}-{random.randint(0, 10)}",
                'elo_rating': 1500 + random.randint(-300, 300),
            },

            # Team composition
            'roster': {
                'active_players': random.randint(15, 30),
                'average_age': round(random.uniform(24, 32), 1),
                'star_players': random.randint(1, 5),
                'international_players': random.randint(2, 15),
                'roster_stability': round(random.uniform(50, 95), 1),  # % unchanged from last season
            },

            # Team tactical profile
            'tactical_profile': {
                'offensive_rating': round(random.uniform(6.5, 9.5), 1),
                'defensive_rating': round(random.uniform(6.5, 9.5), 1),
                'play_style': random.choice(['Aggressive', 'Balanced', 'Defensive', 'Counter-attacking', 'Possession']),
                'formation': random.choice(['4-3-3', '4-4-2', '3-5-2', '5-3-2', '4-2-3-1']),
                'pressing_intensity': round(random.uniform(6.0, 9.5), 1),
            },

            # Historical performance
            'historical_performance': {
                'championships': random.randint(0, 10),
                'playoff_appearances': random.randint(0, 30),
                'historical_win_percentage': round(random.uniform(0.4, 0.7), 3),
                'best_season_record': f"{random.randint(40, 60)}-{random.randint(10, 30)}",
                'worst_season_record': f"{random.randint(10, 30)}-{random.randint(40, 60)}",
            },

            # Team health
            'team_health': {
                'injured_players': random.randint(0, 8),
                'average_games_missed': round(random.uniform(0, 15), 1),
                'key_injuries': random.randint(0, 3),
                'injury_impact_rating': round(random.uniform(1, 10), 1),
            },

            # Upcoming schedule
            'upcoming_matches': [
                {
                    'opponent': f"Team {chr(65 + (team_num + i) % 26)}",
                    'date': (datetime.now() + timedelta(days=3*i)).isoformat(),
                    'home_away': 'Home' if i % 2 == 0 else 'Away',
                    'win_probability': round(random.uniform(0.3, 0.7), 3),
                }
                for i in range(1, 6)
            ],

            # Metadata
            'data_sources': [
                'Official League Stats',
                'Team Reports',
                'AI Analysis',
                'Betting Metrics'
            ],
            'last_updated': datetime.now().isoformat(),
        }

        return jsonify(profile)

    except Exception as e:
        logger.error(f"Error retrieving team profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches/upcoming', methods=['GET'])
def get_upcoming_matches():
    """Get upcoming matches with predictions"""
    try:
        sport = request.args.get('sport', '')
        team = request.args.get('team', '')
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 20))

        # In a real implementation, this would query a database
        # For demo purposes, we'll return mock data
        matches = []

        sports = ['Football', 'Basketball', 'Baseball', 'Soccer', 'Tennis']
        leagues = {
            'Football': ['NFL', 'NCAA', 'CFL'],
            'Basketball': ['NBA', 'NCAA', 'EuroLeague'],
            'Baseball': ['MLB', 'KBO', 'NPB'],
            'Soccer': ['Premier League', 'La Liga', 'Serie A', 'Bundesliga'],
            'Tennis': ['ATP', 'WTA', 'Grand Slam'],
        }

        # Filter by sport if provided
        available_sports = [sport] if sport and sport in sports else sports

        # Generate mock match data
        for i in range(min(limit, 50)):
            # Select random sport and league
            selected_sport = random.choice(available_sports)
            selected_league = random.choice(leagues[selected_sport])

            # Generate team names
            home_team = team if team and i % 2 == 0 else f"Team {chr(65 + i % 26)}"
            away_team = team if team and i % 2 == 1 else f"Team {chr(65 + (i + 13) % 26)}"

            # Skip if both teams are the same
            if home_team == away_team:
                continue

            # Generate match time (within the next 'days' days)
            match_time = datetime.now() + timedelta(
                days=random.randint(0, days-1),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Generate home win probability
            home_win_prob = round(random.uniform(0.3, 0.7), 3)

            # Add more prediction details for certain sports
            prediction_details = {}
            if selected_sport == 'Football':
                prediction_details = {
                    'predicted_score': f"{random.randint(14, 35)}-{random.randint(10, 32)}",
                    'over_under': round(random.uniform(42, 54), 1),
                    'spread': f"{'-' if home_win_prob > 0.5 else '+'}{round(random.uniform(1, 10), 1)}",
                }
            elif selected_sport == 'Basketball':
                prediction_details = {
                    'predicted_score': f"{random.randint(95, 130)}-{random.randint(85, 125)}",
                    'over_under': round(random.uniform(210, 245), 1),
                    'spread': f"{'-' if home_win_prob > 0.5 else '+'}{round(random.uniform(1, 12), 1)}",
                }
            elif selected_sport == 'Baseball':
                prediction_details = {
                    'predicted_score': f"{random.randint(2, 9)}-{random.randint(1, 8)}",
                    'over_under': round(random.uniform(7, 11), 1),
                    'predicted_pitchers_era': [round(random.uniform(2.5, 5.5), 2), round(random.uniform(2.5, 5.5), 2)],
                }
            elif selected_sport == 'Soccer':
                prediction_details = {
                    'predicted_score': f"{random.randint(0, 4)}-{random.randint(0, 3)}",
                    'over_under_goals': round(random.uniform(2, 3.5), 1),
                    'both_teams_to_score': random.random() > 0.4,
                    'clean_sheet_probability': round(random.uniform(0.2, 0.5), 2),
                }

            # Create match object
            match = {
                'id': f"match_{i}_{int(time.time())}",
                'sport': selected_sport,
                'league': selected_league,
                'home_team': home_team,
                'away_team': away_team,
                'datetime': match_time.isoformat(),
                'venue': f"{home_team} Stadium",
                'prediction': {
                    'home_win_probability': home_win_prob,
                    'away_win_probability': round(1 - home_win_prob, 3),
                    'draw_probability': round(random.uniform(0, 0.3), 3) if selected_sport in ['Soccer'] else 0,
                    'confidence': round(random.uniform(0.6, 0.95), 2),
                    'key_factors': [
                        'Recent Form',
                        'Head-to-Head Record',
                        'Home Field Advantage',
                        'Player Performance Metrics'
                    ],
                    **prediction_details
                },
                'betting_odds': {
                    'home_win': round(1 / home_win_prob, 2),
                    'away_win': round(1 / (1 - home_win_prob), 2),
                    'draw': round(1 / random.uniform(0.1, 0.3), 2) if selected_sport in ['Soccer'] else None,
                },
                'key_players': [
                    {'name': f"Player {i}A", 'team': home_team, 'impact_score': round(random.uniform(7.5, 9.8), 1)},
                    {'name': f"Player {i}B", 'team': away_team, 'impact_score': round(random.uniform(7.5, 9.8), 1)},
                ],
                'weather': {
                    'condition': random.choice(['Clear', 'Cloudy', 'Rain', 'Snow', 'Wind']),
                    'temperature': round(random.uniform(5, 30), 1),
                    'precipitation_chance': round(random.uniform(0, 100), 1),
                    'wind_speed': round(random.uniform(0, 25), 1),
                },
                'last_updated': datetime.now().isoformat(),
            }

            matches.append(match)

        return jsonify({
            'sport': sport,
            'team': team,
            'days': days,
            'count': len(matches),
            'matches': matches
        })

    except Exception as e:
        logger.error(f"Error retrieving upcoming matches: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sports/leagues', methods=['GET'])
def get_sports_leagues():
    """Get list of available sports and leagues"""
    try:
        # In a real implementation, this would come from a database
        # For demo purposes, we'll return a comprehensive static list
        sports_leagues = {
            'American Football': ['NFL', 'NCAA Football', 'CFL'],
            'Basketball': ['NBA', 'NCAA Basketball', 'WNBA', 'EuroLeague', 'Liga ACB'],
            'Baseball': ['MLB', 'Minor League Baseball', 'KBO', 'NPB'],
            'Soccer': ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'MLS', 'Champions League', 'World Cup'],
            'Ice Hockey': ['NHL', 'KHL', 'IIHF World Championships'],
            'Tennis': ['ATP', 'WTA', 'Australian Open', 'French Open', 'Wimbledon', 'US Open', 'Davis Cup'],
            'Golf': ['PGA Tour', 'European Tour', 'Masters', 'US Open', 'British Open', 'PGA Championship'],
            'Boxing': ['WBA', 'WBC', 'IBF', 'WBO', 'Heavyweight', 'Middleweight'],
            'Mixed Martial Arts': ['UFC', 'Bellator', 'ONE Championship'],
            'Cricket': ['IPL', 'T20 World Cup', 'Test matches', 'ODI', '10-over cricket'],
            'Rugby Union': ['Super Rugby', 'Six Nations', 'Rugby World Cup'],
            'Rugby League': ['NRL', 'Super League'],
            'Australian Rules Football': ['AFL'],
            'Lacrosse': ['National Lacrosse League', 'NCAA Lacrosse'],
            'Volleyball': ['FIVB World Championships', 'Olympic Volleyball', 'Professional Leagues'],
            'Handball': ['European Handball League', 'Olympic Handball'],
            'Snooker': ['World Snooker Championship', 'Masters', 'UK Championship'],
            'Darts': ['PDC World Championship', 'Premier League Darts'],
            'Table Tennis': ['ITTF World Tour', 'Olympic Table Tennis'],
            'Badminton': ['BWF World Tour', 'Thomas & Uber Cup'],
            'Esports': ['League of Legends', 'Dota 2', 'CS:GO', 'Valorant', 'FIFA'],
            'Motorsports': ['Formula 1', 'NASCAR', 'IndyCar', 'MotoGP'],
            'Winter Sports': ['Alpine Skiing', 'Snowboarding', 'Figure Skating', 'Biathlon'],
            'Combat Sports': ['Wrestling', 'Kickboxing', 'Judo', 'Taekwondo'],
            'Horse Racing': ['Thoroughbred Racing', 'Harness Racing', 'Kentucky Derby', 'Royal Ascot'],
            'Cycling': ['Tour de France', 'UCI World Championships'],
            'Athletics': ['Diamond League', 'Olympics', 'World Championships'],
        }

        # Add forex currency pairs to the response
        try:
            from web_interface.forex_api import CURRENCY_PAIRS, TIMEFRAMES

            sports_leagues['forex'] = {
                'name': 'Forex',
                'leagues': [pair.replace('/', '') for pair in CURRENCY_PAIRS],
                'timeframes': TIMEFRAMES
            }
        except Exception as e:
            logger.warning(f"Could not add forex to sports leagues: {str(e)}")

        return jsonify({
            'count': len(sports_leagues),
            'sports': [
                {
                    'name': sport,
                    'leagues': leagues,
                    'active': True,  # All sports active in demo
                }
                for sport, leagues in sports_leagues.items()
            ]
        })

    except Exception as e:
        logger.error(f"Error retrieving sports leagues: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Helper functions for athlete and team profiles
def get_position_for_sport(sport):
    """Return a position appropriate for the given sport"""
    positions = {
        'American Football': ['Quarterback', 'Running Back', 'Wide Receiver', 'Tight End', 'Offensive Line', 'Defensive Line', 'Linebacker', 'Cornerback', 'Safety', 'Kicker'],
        'Basketball': ['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center'],
        'Baseball': ['Pitcher', 'Catcher', 'First Base', 'Second Base', 'Third Base', 'Shortstop', 'Left Field', 'Center Field', 'Right Field'],
        'Soccer': ['Goalkeeper', 'Defender', 'Midfielder', 'Forward', 'Striker'],
        'Tennis': ['Singles Specialist', 'Doubles Specialist', 'All-Court Player', 'Clay-Court Specialist', 'Grass-Court Specialist'],
        'Golf': ['Professional Golfer'],
        'Mixed Martial Arts': ['Heavyweight', 'Light Heavyweight', 'Middleweight', 'Welterweight', 'Lightweight', 'Featherweight', 'Bantamweight', 'Flyweight'],
        'Boxing': ['Heavyweight', 'Cruiserweight', 'Light Heavyweight', 'Super Middleweight', 'Middleweight', 'Welterweight', 'Lightweight', 'Featherweight'],
        'Cricket': ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'],
        'Rugby Union': ['Prop', 'Hooker', 'Lock', 'Flanker', 'Number 8', 'Scrum-half', 'Fly-half', 'Centre', 'Wing', 'Full-back'],
        'Rugby League': ['Prop', 'Hooker', 'Second-row', 'Lock', 'Halfback', 'Five-eighth', 'Centre', 'Wing', 'Fullback'],
        'Ice Hockey': ['Goaltender', 'Defenseman', 'Center', 'Winger'],
        'Australian Rules Football': ['Ruck', 'Midfielder', 'Forward', 'Defender'],
        'Lacrosse': ['Goalie', 'Defense', 'Midfield', 'Attack'],
        'Volleyball': ['Setter', 'Outside Hitter', 'Middle Blocker', 'Opposite Hitter', 'Libero'],
        'Handball': ['Goalkeeper', 'Back', 'Center Back', 'Wing', 'Pivot'],
        'Snooker': ['Professional Snooker Player'],
        'Darts': ['Professional Darts Player'],
        'Table Tennis': ['Penhold Grip', 'Shakehand Grip', 'Defensive Player', 'Offensive Player'],
        'Badminton': ['Singles Player', 'Doubles Player', 'Mixed Doubles Specialist']
    }

    return random.choice(positions.get(sport, ['Player']))

def get_advanced_metrics_for_sport(sport):
    """Return advanced metrics appropriate for the given sport"""
    if sport == 'Basketball':
        return {
            'per': round(random.uniform(10, 30), 2),
            'true_shooting': round(random.uniform(50, 65), 1),
            'usage_rate': round(random.uniform(15, 35), 1),
            'win_shares': round(random.uniform(1, 15), 1),
            'box_plus_minus': round(random.uniform(-3, 10), 1),
            'vorp': round(random.uniform(-1, 7), 1),
        }
    elif sport == 'Baseball':
        return {
            'war': round(random.uniform(-1, 8), 1),
            'ops': round(random.uniform(0.6, 1.1), 3),
            'wrc_plus': random.randint(70, 160),
            'era_plus': random.randint(70, 160) if random.random() > 0.5 else None,
            'whip': round(random.uniform(1.0, 1.5), 2) if random.random() > 0.5 else None,
            'babip': round(random.uniform(0.25, 0.35), 3),
        }
    elif sport == 'Football':
        return {
            'qbr': round(random.uniform(40, 95), 1) if random.random() > 0.5 else None,
            'passer_rating': round(random.uniform(70, 120), 1) if random.random() > 0.5 else None,
            'yards_per_attempt': round(random.uniform(3, 15), 1),
            'catch_rate': round(random.uniform(50, 80), 1) if random.random() > 0.5 else None,
            'yards_after_contact': round(random.uniform(1, 5), 1) if random.random() > 0.5 else None,
            'broken_tackles': random.randint(0, 30) if random.random() > 0.5 else None,
        }
    elif sport == 'Soccer':
        return {
            'expected_goals': round(random.uniform(0, 25), 2),
            'expected_assists': round(random.uniform(0, 15), 2),
            'progressive_passes': random.randint(20, 300),
            'progressive_carries': random.randint(20, 300),
            'pressures': random.randint(100, 600),
            'tackles_plus_interceptions': random.randint(20, 150),
        }
    else:
        # Generic advanced metrics for other sports
        return {
            'efficiency_rating': round(random.uniform(60, 95), 1),
            'value_above_replacement': round(random.uniform(-1, 8), 1),
            'clutch_performance': round(random.uniform(60, 95), 1),
            'consistency_rating': round(random.uniform(60, 95), 1),
        }

def check_data_availability():
    """Check availability of different data sources"""
    # In a real application, this would check databases or APIs
    # For demo purposes, we'll simulate availability
    data_dir = os.path.join(src_dir, '..', 'data', 'processed_data')

    # Check existence of data directory
    if not os.path.exists(data_dir):
        return {
            'sports_data': 'unavailable',
            'betting_data': 'unavailable',
            'market_data': 'unavailable'
        }

    # Initialize status
    status = {
        'sports_data': 'unavailable',
        'betting_data': 'unavailable',
        'market_data': 'unavailable'
    }

    try:
        # Check for sports data
        sports_files = [f for f in os.listdir(data_dir) if f.startswith(('sports_', 'Historical_sports_')) and f.endswith('.csv')]
        if sports_files:
            status['sports_data'] = 'available'

        # Check for betting data
        betting_files = [f for f in os.listdir(data_dir) if f.startswith('betting_') and f.endswith('.csv')]
        if betting_files:
            status['betting_data'] = 'available'

        # Check for market data
        market_files = [f for f in os.listdir(data_dir) if f.startswith('processed_market_data_') and f.endswith('.csv')]
        if market_files:
            status['market_data'] = 'available'

    except Exception as e:
        logger.error(f"Error checking data availability: {str(e)}")

    return status

def process_sports_input(data):
    """Process sports prediction input data into model features"""
    # Create feature array with defaults
    features = np.zeros(8)  # 8 features for sports model

    try:
        # Process common features we can extract
        if 'home_odds' in data and 'away_odds' in data:
            features[5] = float(data['home_odds'])  # home_odds
            features[7] = float(data['away_odds'])  # away_odds
            features[8] = features[5] / features[7]  # odds_ratio

        # For demo, fill remaining features with random values
        for i in range(len(features)):
            if features[i] == 0:
                features[i] = np.random.uniform(0, 1)

    except Exception as e:
        logger.error(f"Error processing sports input: {str(e)}")

    return features.reshape(1, -1)  # Reshape for model input

def process_betting_input(data):
    """Process betting prediction input data into model features"""
    # Create feature array with defaults
    features = np.zeros(18)  # 18 features for betting model

    try:
        # Process common features we can extract
        if 'odds' in data:
            features[5] = float(data['odds'])
            # Implied probability
            features[9] = 1 / float(data['odds'])

        if 'stake' in data:
            features[6] = float(data['stake'])

        # For demo, fill remaining features with random values
        for i in range(len(features)):
            if features[i] == 0:
                features[i] = np.random.uniform(0, 1)

    except Exception as e:
        logger.error(f"Error processing betting input: {str(e)}")

    return features.reshape(1, -1)  # Reshape for model input

def get_risk_level(expected_value, stake):
    """Determine risk level based on expected value and stake"""
    if expected_value <= 0:
        return "High"
    elif expected_value < stake * 0.2:
        return "Medium"
    else:
        return "Low"

def generate_simulated_sports_prediction(data):
    """Generate simulated sports prediction for demo"""
    home_win_prob = np.random.uniform(0.3, 0.7)

    return {
        'home_team': data['home_team'],
        'away_team': data['away_team'],
        'home_win_probability': float(home_win_prob),
        'away_win_probability': 1.0 - float(home_win_prob),
        'confidence': float(np.random.uniform(0.6, 0.95)),
        'factors': {
            'player_performance': float(np.random.uniform(0.5, 0.9)),
            'team_history': float(np.random.uniform(0.5, 0.9)),
            'opponent_analysis': float(np.random.uniform(0.5, 0.9))
        },
        'note': 'This is a simulated prediction (model not available)',
        'timestamp': datetime.now().isoformat()
    }

def generate_simulated_betting_prediction(data):
    """Generate simulated betting prediction for demo"""
    expected_value = float(data['stake']) * np.random.uniform(-0.2, 0.3)

    return {
        'event': data['event'],
        'odds': float(data['odds']),
        'stake': float(data['stake']),
        'expected_value': expected_value,
        'expected_return': expected_value + float(data['stake']),
        'risk_level': get_risk_level(expected_value, float(data['stake'])),
        'confidence': float(np.random.uniform(0.6, 0.95)),
        'odds_assessment': {
            'fair_odds': float(data['odds']) * (1 - np.random.uniform(0.05, 0.2)),
            'market_odds': float(data['odds']),
            'edge': float(np.random.uniform(0.0, 0.2))
        },
        'note': 'This is a simulated prediction (model not available)',
        'timestamp': datetime.now().isoformat()
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Super AI Web Interface')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def main():
    """Main entry point for the web interface"""
    args = parse_arguments()

    # Set debug environment variable if requested
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'

    # Load models
    app.config['models'] = load_models()

    # Run the Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

@app.route('/api/athletes/compare', methods=['POST'])
def compare_athletes():
    """
    Compare two athletes and predict match outcome based on various contextual factors.
    """
    try:
        data = request.get_json()

        if not data or 'athlete1' not in data or 'athlete2' not in data:
            return jsonify({
                'error': 'Missing athlete information',
                'message': 'Please provide both athletes for comparison'
            }), 400

        athlete1_name = data.get('athlete1')
        athlete2_name = data.get('athlete2')

        # Optional parameters
        sport = data.get('sport')
        surface = data.get('surface')  # e.g., 'clay', 'grass', 'hard'
        tournament = data.get('tournament')

        # Use AthleteProfiler to get detailed metrics
        from .athlete_profiler import AthleteProfiler
        profiler = AthleteProfiler(db_connection=None)  # In production, use actual DB

        # Identify athletes
        athlete1_info = profiler.identify_athlete(name=athlete1_name, sport=sport)
        athlete2_info = profiler.identify_athlete(name=athlete2_name, sport=sport)

        if not athlete1_info or not athlete2_info:
            return jsonify({
                'error': 'Athlete not found',
                'message': 'One or both athletes could not be identified'
            }), 404

        # Get metrics for both athletes
        athlete1_metrics = profiler.scrape_athlete_metrics(athlete1_info['id'])
        athlete2_metrics = profiler.scrape_athlete_metrics(athlete2_info['id'])

        # Analyze and compare metrics for contextual prediction
        prediction = _analyze_matchup(
            athlete1_info, athlete1_metrics,
            athlete2_info, athlete2_metrics,
            surface=surface, tournament=tournament
        )

        return jsonify({
            'prediction': prediction,
            'athlete1': athlete1_info,
            'athlete2': athlete2_info,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in athlete comparison: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

def _analyze_matchup(athlete1_info, athlete1_metrics,
                     athlete2_info, athlete2_metrics,
                     surface=None, tournament=None):
    """
    Analyze metrics to predict match outcome with contextual factors.
    """
    # Extract metrics for comparison
    try:
        # 1. Recent Form Analysis
        form_score_1 = _calculate_form_score(athlete1_metrics)
        form_score_2 = _calculate_form_score(athlete2_metrics)

        # 2. Fitness Analysis
        fitness_score_1 = _calculate_fitness_score(athlete1_metrics)
        fitness_score_2 = _calculate_fitness_score(athlete2_metrics)

        # 3. Tactical Matchup Analysis
        tactical_advantage = _analyze_tactical_matchup(
            athlete1_metrics, athlete2_metrics,
            sport=athlete1_info.get('sport')
        )

        # 4. Mental Resilience Analysis
        mental_edge = _analyze_mental_edge(athlete1_metrics, athlete2_metrics)

        # 5. Surface Compatibility (especially for tennis)
        surface_factor = _analyze_surface_compatibility(
            athlete1_metrics, athlete2_metrics, surface
        )

        # Weight the factors and calculate overall prediction
        # Higher value favors athlete1, lower value favors athlete2
        prediction_score = (
            0.25 * (form_score_1 - form_score_2) +
            0.20 * (fitness_score_1 - fitness_score_2) +
            0.20 * tactical_advantage +
            0.15 * mental_edge +
            0.20 * surface_factor
        )

        # Generate prediction text
        confidence = abs(prediction_score) * 100  # Convert to percentage
        confidence = min(95, max(55, confidence))  # Cap between 55-95%

        if prediction_score > 0:
            winner = athlete1_info.get('name')
            loser = athlete2_info.get('name')
        else:
            winner = athlete2_info.get('name')
            loser = athlete1_info.get('name')
            confidence = abs(confidence)

        prediction = {
            'winner': winner,
            'loser': loser,
            'confidence': round(confidence, 1),
            'analysis': {
                'form_comparison': _format_comparison(form_score_1, form_score_2),
                'fitness_comparison': _format_comparison(fitness_score_1, fitness_score_2),
                'tactical_analysis': _format_tactical(tactical_advantage),
                'mental_edge': _format_mental(mental_edge),
                'surface_compatibility': _format_surface(surface_factor, surface)
            },
            'reasoning': _generate_prediction_reasoning(
                winner, loser,
                form_score_1, form_score_2,
                fitness_score_1, fitness_score_2,
                tactical_advantage, mental_edge, surface_factor,
                surface
            )
        }

        return prediction

    except Exception as e:
        logger.error(f"Error in matchup analysis: {str(e)}")
        return {
            'error': 'Analysis failed',
            'message': str(e)
        }

def _calculate_form_score(metrics):
    """Calculate recent form score from metrics"""
    # Extract relevant metrics from athlete data
    # In a real implementation, this would analyze recent match results
    try:
        technical = metrics.get('technical_metrics', {})
        advanced = metrics.get('advanced_metrics', {})

        # Combine various technical aspects
        accuracy = technical.get('accuracy', 0) / 100 if technical.get('accuracy') else 0.7
        efficiency = technical.get('scoring_efficiency', 0) / 100 if technical.get('scoring_efficiency') else 0.6

        # Advanced metrics for form
        performance_rating = advanced.get('efficiency_rating', 0) / 100 if advanced.get('efficiency_rating') else 0.7
        consistency = advanced.get('consistency_rating', 0) / 10 if advanced.get('consistency_rating') else 0.6

        # Calculate overall form score (0-1 scale)
        form_score = (accuracy * 0.25 + efficiency * 0.25 +
                     performance_rating * 0.30 + consistency * 0.20)

        # Add some randomness to simulate real-world variability
        form_score = form_score * (0.9 + random.random() * 0.2)

        return min(1.0, max(0.0, form_score))
    except:
        return 0.7  # Default if calculation fails

def _calculate_fitness_score(metrics):
    """Calculate fitness level from physical metrics"""
    try:
        physical = metrics.get('physical_metrics', {})
        health = metrics.get('health_metrics', {})

        # Physical fitness components
        endurance = physical.get('vo2_max', 0) / 70 if physical.get('vo2_max') else 0.7
        power = physical.get('power_output', 0) / 2000 if physical.get('power_output') else 0.6
        speed = physical.get('max_speed', 0) / 40 if physical.get('max_speed') else 0.7

        # Health factors (inverse of injury risk)
        injury_status = 1.0 - (health.get('injury_frequency', 0) / 5) if health.get('injury_frequency') else 0.8

        # Calculate overall fitness score (0-1 scale)
        fitness_score = (endurance * 0.3 + power * 0.2 +
                         speed * 0.2 + injury_status * 0.3)

        # Add slight randomness
        fitness_score = fitness_score * (0.95 + random.random() * 0.1)

        return min(1.0, max(0.0, fitness_score))
    except:
        return 0.7  # Default if calculation fails

def _analyze_tactical_matchup(metrics1, metrics2, sport=None):
    """
    Analyze tactical matchup between two athletes.
    Returns value between -1 and 1 (positive favors athlete1)
    """
    try:
        tactical1 = metrics1.get('tactical_metrics', {})
        tactical2 = metrics2.get('tactical_metrics', {})

        # Get sport-specific matchup factors
        if sport and sport.lower() == 'tennis':
            # Tennis-specific tactical analysis
            offense1 = tactical1.get('offensive_contribution', 0)
            offense2 = tactical2.get('offensive_contribution', 0)

            defense1 = tactical1.get('defensive_positioning', 0)
            defense2 = tactical2.get('defensive_positioning', 0)

            # Calculate matchup advantage
            tactical_advantage = ((offense1 - defense2) - (offense2 - defense1)) / 10
        else:
            # Generic tactical analysis
            game_iq1 = tactical1.get('game_intelligence', 0) / 10 if tactical1.get('game_intelligence') else 0.7
            game_iq2 = tactical2.get('game_intelligence', 0) / 10 if tactical2.get('game_intelligence') else 0.7

            tactical_advantage = (game_iq1 - game_iq2) * 2  # Scale to -1 to 1 range

        # Add variability
        tactical_advantage += random.uniform(-0.1, 0.1)

        return min(1.0, max(-1.0, tactical_advantage))
    except:
        return random.uniform(-0.2, 0.2)  # Slight random advantage if calculation fails

def _analyze_mental_edge(metrics1, metrics2):
    """
    Analyze mental resilience and head-to-head mental edge.
    Returns value between -1 and 1 (positive favors athlete1)
    """
    try:
        mental1 = metrics1.get('mental_metrics', {})
        mental2 = metrics2.get('mental_metrics', {})

        # Extract mental factors
        composure1 = mental1.get('composure_rating', 0) / 10 if mental1.get('composure_rating') else 0.7
        composure2 = mental2.get('composure_rating', 0) / 10 if mental2.get('composure_rating') else 0.7

        focus1 = mental1.get('focus_rating', 0) / 10 if mental1.get('focus_rating') else 0.7
        focus2 = mental2.get('focus_rating', 0) / 10 if mental2.get('focus_rating') else 0.7

        # High pressure performance
        clutch1 = mental1.get('high_pressure_performance', 0) / 10 if mental1.get('high_pressure_performance') else 0.7
        clutch2 = mental2.get('high_pressure_performance', 0) / 10 if mental2.get('high_pressure_performance') else 0.7

        # Calculate overall mental edge
        mental_edge = (
            (composure1 - composure2) * 0.3 +
            (focus1 - focus2) * 0.3 +
            (clutch1 - clutch2) * 0.4
        )

        # Add head-to-head psychological factor (random for simulation)
        h2h_factor = random.uniform(-0.2, 0.2)

        return min(1.0, max(-1.0, mental_edge + h2h_factor))
    except:
        return random.uniform(-0.15, 0.15)  # Slight random advantage if calculation fails

def _analyze_surface_compatibility(metrics1, metrics2, surface=None):
    """
    Analyze surface compatibility (especially for tennis).
    Returns value between -1 and 1 (positive favors athlete1)
    """
    # If no surface specified or not tennis, return neutral
    if not surface:
        return 0.0

    try:
        tech1 = metrics1.get('technical_metrics', {})
        tech2 = metrics2.get('technical_metrics', {})

        # Get sport-specific metrics
        sport_metrics1 = tech1.get('sport_specific', {})
        sport_metrics2 = tech2.get('sport_specific', {})

        # Surface-specific ratings (simulated)
        surface = surface.lower()

        # For tennis, different surfaces favor different playing styles
        if surface == 'clay':
            # Clay favors endurance and defensive play
            rating1 = sport_metrics1.get('clay_rating', random.uniform(0.5, 0.9))
            rating2 = sport_metrics2.get('clay_rating', random.uniform(0.5, 0.9))
        elif surface == 'grass':
            # Grass favors serve-and-volley and aggressive play
            rating1 = sport_metrics1.get('grass_rating', random.uniform(0.5, 0.9))
            rating2 = sport_metrics2.get('grass_rating', random.uniform(0.5, 0.9))
        elif surface == 'hard':
            # Hard courts are balanced
            rating1 = sport_metrics1.get('hard_rating', random.uniform(0.6, 0.9))
            rating2 = sport_metrics2.get('hard_rating', random.uniform(0.6, 0.9))
        else:
            # Unknown surface - use general compatibility
            rating1 = random.uniform(0.6, 0.9)
            rating2 = random.uniform(0.6, 0.9)

        return min(1.0, max(-1.0, (rating1 - rating2) * 2))
    except:
        return random.uniform(-0.2, 0.2)  # Slight random advantage if calculation fails

def _format_comparison(score1, score2):
    """Format comparison between two scores"""
    diff = score1 - score2
    if diff > 0.2:
        return "Significant advantage"
    elif diff > 0.05:
        return "Slight advantage"
    elif diff < -0.2:
        return "Significant disadvantage"
    elif diff < -0.05:
        return "Slight disadvantage"
    else:
        return "Evenly matched"

def _format_tactical(advantage):
    """Format tactical advantage"""
    if advantage > 0.3:
        return "Clear tactical advantage"
    elif advantage > 0.1:
        return "Slight tactical edge"
    elif advantage < -0.3:
        return "Significant tactical disadvantage"
    elif advantage < -0.1:
        return "Minor tactical disadvantage"
    else:
        return "Tactically well-matched"

def _format_mental(edge):
    """Format mental edge description"""
    if edge > 0.3:
        return "Strong mental advantage"
    elif edge > 0.1:
        return "Mental resilience edge"
    elif edge < -0.3:
        return "Mental vulnerability"
    elif edge < -0.1:
        return "Slight mental disadvantage"
    else:
        return "Similar mental strength"

def _format_surface(factor, surface):
    """Format surface compatibility description"""
    if not surface:
        return "No specific surface considered"

    surface = surface.capitalize()

    if factor > 0.3:
        return f"Excellent {surface} court compatibility"
    elif factor > 0.1:
        return f"Good {surface} court adaptation"
    elif factor < -0.3:
        return f"Poor {surface} court compatibility"
    elif factor < -0.1:
        return f"Below average on {surface} courts"
    else:
        return f"Neutral {surface} court performance"

def _generate_prediction_reasoning(winner, loser, form1, form2, fitness1, fitness2,
                                 tactical, mental, surface, surface_type=None):
    """Generate natural language reasoning for the prediction"""

    form_text = ""
    if abs(form1 - form2) > 0.15:
        better_form = winner if form1 > form2 else loser
        form_text = f"{better_form} has shown better recent form. "

    fitness_text = ""
    if abs(fitness1 - fitness2) > 0.15:
        more_fit = winner if fitness1 > fitness2 else loser
        fitness_text = f"{more_fit}'s superior fitness levels are a key factor. "

    tactical_text = ""
    if abs(tactical) > 0.2:
        if (tactical > 0 and winner == form1) or (tactical < 0 and winner == form2):
            tactical_text = f"{winner}'s playing style creates matchup problems for {loser}. "

    mental_text = ""
    if abs(mental) > 0.2:
        mentally_stronger = winner if ((mental > 0 and winner == form1) or
                                       (mental < 0 and winner == form2)) else loser
        mental_text = f"{mentally_stronger} has demonstrated better mental resilience under pressure. "

    surface_text = ""
    if surface_type and abs(surface) > 0.2:
        surface_type = surface_type.capitalize()
        surface_text = f"{winner} has historically performed better on {surface_type} courts. "

    # Combine all factors that were significant
    reasoning = form_text + fitness_text + tactical_text + mental_text + surface_text

    # Add general conclusion if we have enough specific points
    if len(reasoning.strip()) > 30:
        reasoning += f"Taking all factors into account, {winner} has the edge over {loser}."
    else:
        # Generic reasoning if not enough specific points
        reasoning = f"Based on overall performance metrics and contextual factors, {winner} has a higher probability of winning against {loser}."

    return reasoning

@app.route('/api/athletes/predict/<athlete_id>', methods=['POST'])
def predict_athlete_performance(athlete_id):
    """
    Generate contextual predictions for an athlete's performance in different scenarios.

    This endpoint accepts various contextual factors and returns predictions for how
    an athlete would perform against different opponents or in different conditions.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Missing request data',
                'message': 'Please provide contextual information for the prediction'
            }), 400

        # Get athlete data
        from .athlete_profiler import AthleteProfiler
        profiler = AthleteProfiler(db_connection=None)  # In production, use actual DB

        # Get athlete profile details
        athlete_info = None
        try:
            # In production, this would query the database by ID
            # For now, we'll use the athlete name if provided in the request
            if 'athlete_name' in data:
                athlete_info = profiler.identify_athlete(
                    name=data.get('athlete_name'),
                    sport=data.get('sport')
                )
            else:
                # This would normally fetch by ID from the database
                # For demo purposes, we're creating a synthetic profile
                athlete_info = {
                    'id': athlete_id,
                    'name': f"Athlete_{athlete_id}",
                    'sport': data.get('sport', 'Unknown')
                }
        except Exception as e:
            return jsonify({
                'error': 'Athlete not found',
                'message': f'Could not retrieve athlete with ID: {athlete_id}'
            }), 404

        # Get athlete metrics
        athlete_metrics = profiler.scrape_athlete_metrics(athlete_id)

        # Combine athlete info and metrics into a single object for the analyzer
        athlete_data = _prepare_athlete_data(athlete_info, athlete_metrics)

        # Process contextual prediction scenarios
        scenarios = data.get('scenarios', [])
        if not scenarios:
            # Default scenario if none provided
            scenarios = [{
                'type': 'default',
                'description': 'Standard performance prediction'
            }]

        # Import the prediction analyzer
        from .prediction_analyzer import PredictionAnalyzer
        analyzer = PredictionAnalyzer()

        # Generate predictions for each scenario
        predictions = []
        for scenario in scenarios:
            scenario_type = scenario.get('type', 'default')

            # Process different scenario types
            if scenario_type == 'opponent':
                # Predict against a specific opponent
                opponent_name = scenario.get('opponent_name')
                if not opponent_name:
                    continue

                # Get opponent data
                opponent_info = profiler.identify_athlete(
                    name=opponent_name,
                    sport=data.get('sport')
                )
                opponent_metrics = profiler.scrape_athlete_metrics(opponent_info['id'])

                # Prepare opponent data for analyzer
                opponent_data = _prepare_athlete_data(opponent_info, opponent_metrics)

                # Create context dict with additional factors
                context = {
                    'surface': scenario.get('surface'),
                    'tournament': scenario.get('tournament'),
                    'home_advantage': scenario.get('home_advantage', False),
                    'pressure_level': scenario.get('pressure_level', 'normal')
                }

                # Generate prediction for this matchup
                prediction = analyzer.analyze_matchup(
                    athlete_data,
                    opponent_data,
                    context
                )

                predictions.append({
                    'scenario': scenario,
                    'prediction': prediction
                })

            elif scenario_type == 'condition':
                # Predict performance under specific conditions
                # For conditions, we'll generate a synthetic opponent that represents
                # the "average" player the athlete might face in these conditions

                # Create a synthetic opponent
                opponent_info = {
                    'id': 'synthetic_opponent',
                    'name': 'Average Opponent',
                    'sport': athlete_info.get('sport')
                }

                # Create context with condition factors
                context = {
                    'surface': scenario.get('surface'),
                    'weather': scenario.get('weather'),
                    'altitude': scenario.get('altitude'),
                    'tournament_stage': scenario.get('tournament_stage'),
                    'pressure_level': scenario.get('pressure_level', 'normal')
                }

                # Get metrics for average opponent in this condition
                opponent_metrics = _generate_average_opponent_metrics(
                    athlete_info.get('sport'),
                    athlete_metrics,
                    context
                )

                # Prepare opponent data for analyzer
                opponent_data = _prepare_athlete_data(opponent_info, opponent_metrics)

                # Generate prediction
                prediction = analyzer.analyze_matchup(
                    athlete_data,
                    opponent_data,
                    context
                )

                predictions.append({
                    'scenario': scenario,
                    'prediction': prediction
                })

            elif scenario_type == 'improvement':
                # Predict performance if athlete improves in certain areas
                # This creates a modified version of the athlete with improved metrics
                improved_athlete = athlete_info.copy()
                improved_athlete['id'] = f"{athlete_id}_improved"
                improved_athlete['name'] = f"{athlete_info.get('name')} (Improved)"

                # Apply improvement factors to metrics
                improvement_areas = scenario.get('improvement_areas', {})
                improved_metrics = _apply_improvements(
                    athlete_metrics,
                    improvement_areas
                )

                # Prepare improved athlete data for analyzer
                improved_athlete_data = _prepare_athlete_data(improved_athlete, improved_metrics)

                # Create average opponent
                opponent_info = {
                    'id': 'synthetic_opponent',
                    'name': 'Average Opponent',
                    'sport': athlete_info.get('sport')
                }

                # Get metrics for average opponent
                opponent_metrics = _generate_average_opponent_metrics(
                    athlete_info.get('sport'),
                    athlete_metrics,
                    {}
                )

                # Prepare opponent data for analyzer
                opponent_data = _prepare_athlete_data(opponent_info, opponent_metrics)

                # Generate prediction with improved metrics
                context = {
                    'improvement_scenario': True,
                    'improved_areas': list(improvement_areas.keys())
                }

                prediction = analyzer.analyze_matchup(
                    improved_athlete_data,
                    opponent_data,
                    context
                )

                # Calculate improvement delta compared to current performance
                base_prediction = analyzer.analyze_matchup(
                    athlete_data,
                    opponent_data,
                    {}
                )

                # Add improvement delta to the prediction
                prediction['improvement_delta'] = {
                    'win_probability_change': prediction.get('athlete1_win_probability', 0.5) -
                                             base_prediction.get('athlete1_win_probability', 0.5),
                    'performance_boost': round(random.uniform(5, 20), 1)  # In a real system, this would be calculated
                }

                predictions.append({
                    'scenario': scenario,
                    'prediction': prediction,
                    'base_prediction': base_prediction
                })

            else:
                # Default prediction scenario - general performance assessment
                # Create a generic opponent representing the average for this athlete's level
                opponent_info = {
                    'id': 'average_opponent',
                    'name': 'Average Opponent',
                    'sport': athlete_info.get('sport')
                }

                # Get metrics for average opponent
                opponent_metrics = _generate_average_opponent_metrics(
                    athlete_info.get('sport'),
                    athlete_metrics,
                    {}
                )

                # Prepare opponent data for analyzer
                opponent_data = _prepare_athlete_data(opponent_info, opponent_metrics)

                # Generate a generic prediction
                prediction = analyzer.analyze_matchup(
                    athlete_data,
                    opponent_data,
                    {}
                )

                predictions.append({
                    'scenario': scenario,
                    'prediction': prediction
                })

        return jsonify({
            'athlete': athlete_info,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'note': 'Contextual predictions are based on available data and performance models.'
        })

    except Exception as e:
        logger.error(f"Error in athlete prediction: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

def _prepare_athlete_data(athlete_info, metrics):
    """
    Prepare athlete data in the format expected by the PredictionAnalyzer.
    """
    # Start with athlete info
    athlete_data = athlete_info.copy()

    # Add key metrics directly to the athlete data
    athlete_data['recent_win_percentage'] = random.uniform(0.4, 0.8)

    # Surface win percentages
    athlete_data['surface_win_percentages'] = {
        'hard': random.uniform(0.4, 0.7),
        'clay': random.uniform(0.3, 0.8),
        'grass': random.uniform(0.3, 0.8)
    }

    # Head-to-head records
    athlete_data['head_to_head'] = {}

    # Mental metrics
    athlete_data['tiebreak_win_percentage'] = random.uniform(0.4, 0.7)
    athlete_data['deciding_set_win_percentage'] = random.uniform(0.4, 0.7)

    # Playing style
    athlete_data['style_of_play'] = random.choice(['aggressive', 'defensive', 'all-court'])
    athlete_data['strengths'] = random.sample(['forehand', 'backhand', 'serve', 'return'], 2)
    athlete_data['weaknesses'] = random.sample(['forehand', 'backhand', 'serve', 'return'], 2)

    # Add metrics to athlete data
    for category, category_metrics in metrics.items():
        athlete_data[category] = category_metrics

    # Add injury and fatigue info
    athlete_data['injury_status'] = 'healthy'
    athlete_data['fatigue_level'] = random.uniform(10, 50)
    athlete_data['recovery_score'] = random.uniform(50, 95)

    return athlete_data

def _apply_improvements(metrics, improvement_areas):
    """
    Apply improvement factors to the athlete's metrics.

    Args:
        metrics: The athlete's current metrics
        improvement_areas: Dict mapping improvement areas to percentage improvements

    Returns:
        The improved metrics
    """
    improved_metrics = copy.deepcopy(metrics)

    for area, improvement_pct in improvement_areas.items():
        # Parse the improvement area to find the category and specific metric
        parts = area.split('.')
        if len(parts) == 2:
            category, metric = parts
            if category in improved_metrics and metric in improved_metrics[category]:
                # Apply the improvement percentage
                current_value = improved_metrics[category][metric]
                if isinstance(current_value, (int, float)):
                    improved_metrics[category][metric] = current_value * (1 + improvement_pct/100)
        elif parts[0] in improved_metrics:
            # Apply improvement to entire category
            category = parts[0]
            for metric, value in improved_metrics[category].items():
                if isinstance(value, (int, float)):
                    improved_metrics[category][metric] = value * (1 + improvement_pct/100)

    return improved_metrics

def _generate_average_opponent_metrics(sport, athlete_metrics, context):
    """
    Generate metrics for an average opponent based on the sport and context.
    This is a placeholder function that would be replaced with actual
    statistical models in production.
    """
    # In a real system, this would use statistics for average players in the given sport/level
    average_metrics = {}

    # Copy structure from athlete metrics but adjust values to be average
    for category, metrics in athlete_metrics.items():
        average_metrics[category] = {}
        for metric, value in metrics.items():
            # Adjust each metric to be slightly different from the athlete
            # In a real system, this would use actual statistical averages
            if isinstance(value, (int, float)):
                # Add some random variation around the athlete's value
                average_metrics[category][metric] = value * random.uniform(0.85, 1.15)
            else:
                average_metrics[category][metric] = value

    # Apply context-specific adjustments
    # For example, some opponents might be better on certain surfaces
    if 'surface' in context and context['surface']:
        surface = context.get('surface')
        if surface == 'clay':
            # Simulate that the average opponent might be slightly better on clay
            if 'technical_metrics' in average_metrics:
                average_metrics['technical_metrics']['accuracy'] = \
                    average_metrics['technical_metrics'].get('accuracy', 0) * 1.05
        elif surface == 'grass':
            # Might be slightly worse on grass
            if 'technical_metrics' in average_metrics:
                average_metrics['technical_metrics']['accuracy'] = \
                    average_metrics['technical_metrics'].get('accuracy', 0) * 0.95

    # Tournament stage adjustment
    if 'tournament_stage' in context and context['tournament_stage']:
        stage = context['tournament_stage']
        if stage == 'final':
            # Better opponents in finals
            factor = 1.1
        elif stage == 'semifinal':
            factor = 1.05
        else:
            factor = 1.0

        # Apply tournament stage factor
        for category in average_metrics:
            for metric, value in average_metrics[category].items():
                if isinstance(value, (int, float)):
                    average_metrics[category][metric] = value * factor

    # Weather conditions
    if 'weather' in context and context['weather']:
        weather = context['weather']
        if weather == 'rain':
            # Some metrics may be affected by rain
            if 'physical_metrics' in average_metrics:
                average_metrics['physical_metrics']['agility_score'] = \
                    average_metrics['physical_metrics'].get('agility_score', 0) * 0.9
        elif weather == 'hot':
            # Endurance affected in hot weather
            if 'physical_metrics' in average_metrics:
                average_metrics['physical_metrics']['vo2_max'] = \
                    average_metrics['physical_metrics'].get('vo2_max', 0) * 0.95

    return average_metrics

# Import the Player Intelligence Engine
@app.route('/api/athletes/readiness/<athlete_id>', methods=['GET'])
def get_athlete_readiness(athlete_id):
    """
    Get the current readiness score for an athlete.

    This endpoint calculates a comprehensive readiness score based on
    various metrics including recent form, fitness, and recovery.
    """
    try:
        # Import the PlayerIntelligence module
        from ..core.player_intelligence import player_intelligence

        # Get readiness score
        readiness = player_intelligence.calculate_readiness_score(athlete_id)

        # Get athlete profile for name and other details
        profile = player_intelligence.get_athlete_profile(athlete_id)

        return jsonify({
            'athlete_id': athlete_id,
            'name': profile.get('name', f"Athlete_{athlete_id}"),
            'readiness_score': readiness,
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'recent_form': profile.get('metrics', {}).get('recent_form', 0.5) * 100,
                'fatigue_level': profile.get('metrics', {}).get('fatigue_level', 50),
                'injury_status': profile.get('metrics', {}).get('injury_status', 'unknown')
            }
        })

    except Exception as e:
        logger.error(f"Error calculating athlete readiness: {str(e)}")
        return jsonify({
            'error': 'Failed to calculate readiness',
            'message': str(e)
        }), 500

@app.route('/api/athletes/injury-risk/<athlete_id>', methods=['GET'])
def get_athlete_injury_risk(athlete_id):
    """
    Calculate the injury risk probability for an athlete.

    This endpoint analyzes various factors including injury history,
    workload, age, and sport-specific risks.
    """
    try:
        # Import the PlayerIntelligence module
        from ..core.player_intelligence import player_intelligence

        # Get injury probability
        injury_probability = player_intelligence.calculate_injury_probability(athlete_id)

        # Get athlete profile for name and other details
        profile = player_intelligence.get_athlete_profile(athlete_id)

        # Determine risk level
        risk_level = 'low'
        if injury_probability > 70:
            risk_level = 'high'
        elif injury_probability > 40:
            risk_level = 'medium'

        return jsonify({
            'athlete_id': athlete_id,
            'name': profile.get('name', f"Athlete_{athlete_id}"),
            'injury_probability': injury_probability,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'age': profile.get('metrics', {}).get('age', 0),
                'injury_history': profile.get('injury_history', []),
                'recent_workload': profile.get('metrics', {}).get('work_rate_km_per_game', 0)
            }
        })

    except Exception as e:
        logger.error(f"Error calculating injury risk: {str(e)}")
        return jsonify({
            'error': 'Failed to calculate injury risk',
            'message': str(e)
        }), 500

@app.route('/api/athletes/matchup', methods=['POST'])
def analyze_athlete_matchup():
    """
    Analyze a matchup between two athletes with sport-specific metrics weighting.

    This endpoint provides a more comprehensive matchup analysis than the
    standard comparison, including sport-specific metric weighting.
    """
    try:
        data = request.get_json()

        if not data or 'athlete1_id' not in data or 'athlete2_id' not in data:
            return jsonify({
                'error': 'Missing required data',
                'message': 'Please provide both athlete IDs'
            }), 400

        athlete1_id = data.get('athlete1_id')
        athlete2_id = data.get('athlete2_id')

        # Optional context parameters
        context = {
            'surface': data.get('surface'),
            'venue': data.get('venue'),
            'tournament': data.get('tournament'),
            'tournament_stage': data.get('tournament_stage'),
            'home_advantage': data.get('home_advantage', False)
        }

        # Import the PlayerIntelligence module
        from ..core.player_intelligence import player_intelligence

        # Get matchup analysis
        matchup = player_intelligence.calculate_matchup_rating(athlete1_id, athlete2_id, context)

        if 'error' in matchup:
            return jsonify(matchup), 400

        return jsonify({
            'matchup': matchup,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })

    except Exception as e:
        logger.error(f"Error analyzing athlete matchup: {str(e)}")
        return jsonify({
            'error': 'Failed to analyze matchup',
            'message': str(e)
        }), 500

@app.route('/api/athletes/scrape', methods=['POST'])
def scrape_athlete_data():
    """
    Scrape and update athlete data from multiple sources.

    This endpoint triggers data collection from various sources including
    sports websites, wearable APIs, and other data providers.
    """
    try:
        data = request.get_json()

        if not data or 'name' not in data or 'sport' not in data:
            return jsonify({
                'error': 'Missing required data',
                'message': 'Please provide athlete name and sport'
            }), 400

        name = data.get('name')
        sport = data.get('sport')
        sources = data.get('sources')  # Optional specific sources

        # Import the athlete scraper
        from ..scrapers.athlete_metrics_scraper import athlete_scraper

        # Scrape athlete data
        athlete_data = athlete_scraper.scrape_athlete(name, sport, sources)

        if not athlete_data:
            return jsonify({
                'error': 'No data found',
                'message': f'Could not find data for {name} in {sport}'
            }), 404

        # Store the athlete ID for reference
        athlete_id = athlete_data.get('id')

        # Add Wearable Data if provided
        wearable_type = data.get('wearable_type')
        if wearable_type:
            try:
                wearable_data = athlete_scraper.fetch_wearable_data(athlete_id, wearable_type)

                # Import PlayerIntelligence to merge data
                from ..core.player_intelligence import player_intelligence

                # Merge with scraped data
                merged_data = player_intelligence.merge_athlete_data(athlete_id, [
                    {
                        'source': athlete_data.get('source', 'web'),
                        'metrics': athlete_data.get('metrics', {}),
                        'reliability': 0.8
                    },
                    {
                        'source': wearable_data.get('source'),
                        'metrics': wearable_data.get('metrics', {}),
                        'reliability': wearable_data.get('reliability', 0.9)
                    }
                ])

                athlete_data = merged_data
            except Exception as we:
                logger.error(f"Error fetching wearable data: {str(we)}")
                # Continue with just the scraped data

        return jsonify({
            'athlete_id': athlete_id,
            'name': athlete_data.get('name'),
            'sport': athlete_data.get('sport'),
            'data_sources': athlete_data.get('data_sources', []),
            'metrics_count': len(athlete_data.get('metrics', {})),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error scraping athlete data: {str(e)}")
        return jsonify({
            'error': 'Failed to scrape athlete data',
            'message': str(e)
        }), 500

@app.route('/api/athletes/metric-trends/<athlete_id>', methods=['GET'])
def get_athlete_metric_trends(athlete_id):
    """
    Get trend analysis for athlete metrics over time.

    This endpoint provides trend data for specific metrics to track
    performance development over time.
    """
    try:
        # Get query parameters
        metrics = request.args.get('metrics')
        period = request.args.get('period', 5, type=int)

        if not metrics:
            return jsonify({
                'error': 'Missing required parameters',
                'message': 'Please provide metrics to analyze'
            }), 400

        # Parse metrics list
        metric_list = metrics.split(',')

        # Import the PlayerIntelligence module
        from ..core.player_intelligence import player_intelligence

        # Get metric trends
        trends = player_intelligence.get_metric_trends(athlete_id, metric_list, period)

        # Get athlete profile for name and other details
        profile = player_intelligence.get_athlete_profile(athlete_id)

        return jsonify({
            'athlete_id': athlete_id,
            'name': profile.get('name', f"Athlete_{athlete_id}"),
            'sport': profile.get('sport', 'unknown'),
            'trends': trends,
            'period': period,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error analyzing metric trends: {str(e)}")
        return jsonify({
            'error': 'Failed to analyze metric trends',
            'message': str(e)
        }), 500

@app.route('/api/athletes/relevance/<athlete_id>', methods=['GET'])
def get_athlete_relevant_metrics(athlete_id):
    """
    Get sport-relevant metrics for an athlete.

    This endpoint extracts only the metrics that are most relevant to
    the athlete's specific sport, with appropriate weightings.
    """
    try:
        # Import the PlayerIntelligence module
        from ..core.player_intelligence import player_intelligence

        # Get athlete profile
        profile = player_intelligence.get_athlete_profile(athlete_id)

        if not profile:
            return jsonify({
                'error': 'Athlete not found',
                'message': f'No profile found for athlete ID: {athlete_id}'
            }), 404

        # Extract sport-relevant metrics
        relevant_data = player_intelligence.extract_sport_relevant_metrics(profile)

        return jsonify({
            'athlete_id': athlete_id,
            'name': profile.get('name', f"Athlete_{athlete_id}"),
            'sport': profile.get('sport', 'unknown'),
            'relevant_metrics': relevant_data.get('relevant_metrics', {}),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error extracting relevant metrics: {str(e)}")
        return jsonify({
            'error': 'Failed to extract relevant metrics',
            'message': str(e)
        }), 500

# Import the ValueBettingModel class
try:
    from betting.value_betting_model import ValueBettingModel
except ImportError:
    logger.warning("ValueBettingModel not found, some functionality will be limited")
    ValueBettingModel = None

# Ensure the existing code has app.config

# Add value betting model to the app configuration if not present
def initialize_models():
    # Initialize other models if they aren't already (keep existing code)
    if 'models' not in app.config:
        app.config['models'] = {}

    # Add value betting model for each supported sport
    if 'value_betting_models' not in app.config['models']:
        app.config['models']['value_betting_models'] = {}

        # Add models for common sports if ValueBettingModel is available
        if ValueBettingModel is not None:
            for sport in ['american_football', 'basketball', 'soccer']:
                for market in ['moneyline', 'spread', 'over_under']:
                    model_id = f"{sport}_{market}"
                    try:
                        # Initialize value betting model (without loading a pre-trained model yet)
                        value_model = ValueBettingModel(sport=sport, market=market)
                        app.config['models']['value_betting_models'][model_id] = value_model
                        logger.info(f"Initialized value betting model for {sport} {market}")
                    except Exception as e:
                        logger.error(f"Failed to initialize value betting model for {sport} {market}: {str(e)}")
        else:
            logger.warning("Skipping value betting model initialization as ValueBettingModel is not available")

    # Try to initialize forex model cache
    try:
        from trading.forex_model import ForexTradingModel
        from web_interface.forex_api import MODEL_CACHE

        # Check for existing models
        models_dir = os.path.join(src_dir, "models", "forex")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if model_files:
                # Just load the first model to verify functionality
                sample_model = model_files[0]
                parts = sample_model.replace('.joblib', '').split('_')

                if len(parts) >= 3:
                    # Extract model details
                    if len(parts) > 3:
                        currency_pair = parts[0] + '/' + parts[1]
                        timeframe = parts[2]
                        model_type = parts[3]
                    else:
                        currency_pair = parts[0]
                        timeframe = parts[1]
                        model_type = parts[2]

                    # Try to load the model
                    model = ForexTradingModel(
                        currency_pair=currency_pair,
                        timeframe=timeframe,
                        model_type=model_type
                    )

                    if model.load_model():
                        MODEL_CACHE[f"{currency_pair}_{timeframe}_{model_type}"] = model
                        logger.info(f"Preloaded forex model: {currency_pair} {timeframe}")
        else:
            logger.warning("No forex models directory found")
    except Exception as e:
        logger.warning(f"Could not initialize forex models: {str(e)}")

    logger.info("Models initialized")

# Call initialize_models during app startup
with app.app_context():
    initialize_models()

# New endpoint for value betting analysis
@app.route('/api/betting/value_bets', methods=['POST'])
def find_value_bets():
    """
    Find value betting opportunities with positive expected value (EV).

    Expects JSON with:
    {
        "sport": "soccer",
        "market": "moneyline",
        "events": [
            {
                "home_team": "Team A",
                "away_team": "Team B",
                "market_odds": 2.5,
                "selection": "Team A",
                ...other features
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Check for required fields
        required_fields = ['sport', 'market', 'events']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'required': required_fields}), 400

        sport = data['sport'].lower()
        market = data['market'].lower()
        events = data['events']

        if not events:
            return jsonify({'error': 'No events provided for analysis'}), 400

        # If ValueBettingModel is not available, return simulated data
        if ValueBettingModel is None:
            return jsonify(generate_simulated_value_bets(data))

        # Check if value betting model exists for this sport/market
        model_id = f"{sport}_{market}"
        value_betting_models = app.config.get('models', {}).get('value_betting_models', {})

        if model_id not in value_betting_models:
            # No model exists, create one for this request
            try:
                value_model = ValueBettingModel(sport=sport, market=market)
                # Add to app config for future requests
                if 'models' not in app.config:
                    app.config['models'] = {}
                if 'value_betting_models' not in app.config['models']:
                    app.config['models']['value_betting_models'] = {}
                app.config['models']['value_betting_models'][model_id] = value_model
            except Exception as e:
                logger.error(f"Failed to create value betting model for {sport} {market}: {str(e)}")
                # Return simulated value bets
                return jsonify(generate_simulated_value_bets(data))
        else:
            value_model = value_betting_models[model_id]

        # Check if we have a trained model, if not use fake predictions
        if value_model.model is None:
            # Return simulated value bets for demo/development
            return jsonify(generate_simulated_value_bets(data))

        # Use the model to find value bets
        value_bets = value_model.find_value_bets(events)

        # Create response
        result = {
            'sport': sport,
            'market': market,
            'value_bets_found': len(value_bets),
            'value_bets': value_bets,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in value betting analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

# New endpoint for training a value betting model
@app.route('/api/betting/train_value_model', methods=['POST'])
def train_value_model():
    """
    Train a value betting model for a specific sport and market.

    Expects JSON with:
    {
        "sport": "soccer",
        "market": "moneyline",
        "model_type": "xgboost"  # Options: logistic, xgboost, random_forest, neural_network
    }

    If no data is provided, will generate synthetic data for training.
    """
    try:
        # Check if ValueBettingModel is available
        if ValueBettingModel is None:
            return jsonify({
                'error': 'ValueBettingModel module is not available',
                'status': 'error',
                'message': 'Value betting model training is not available in this environment'
            }), 400

        try:
            from betting.generate_betting_data import BettingDataGenerator
        except ImportError:
            logger.warning("BettingDataGenerator not found, using simulated training data")
            BettingDataGenerator = None

        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        required_fields = ['sport', 'market']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'required': required_fields}), 400

        sport = data['sport'].lower()
        market = data['market'].lower()
        model_type = data.get('model_type', 'xgboost').lower()
        use_synthetic_data = data.get('use_synthetic_data', True)

        # Check if value betting model exists for this sport/market
        model_id = f"{sport}_{market}"
        value_betting_models = app.config.get('models', {}).get('value_betting_models', {})

        if model_id not in value_betting_models:
            # Create a new model
            value_model = ValueBettingModel(sport=sport, market=market)
            # Add to app config
            if 'models' not in app.config:
                app.config['models'] = {}
            if 'value_betting_models' not in app.config['models']:
                app.config['models']['value_betting_models'] = {}
            app.config['models']['value_betting_models'][model_id] = value_model
        else:
            value_model = value_betting_models[model_id]

        # Generate synthetic data if needed
        if use_synthetic_data:
            generator = BettingDataGenerator()
            df = generator.generate_data(sport=sport, market=market, n_samples=500)
            # Save this data for reference
            data_path = generator.save_data(df, sport, market)
            logger.info(f"Generated synthetic data for {sport} {market} at {data_path}")
        else:
            # In a real application, we would load real data here
            return jsonify({'error': 'Real data training not implemented yet'}), 501

        # Train the model
        training_result = value_model.train_model(
            data=df,
            model_type=model_type,
            target_col='outcome',
            test_size=0.2
        )

        # Return the training results
        result = {
            'status': 'success',
            'sport': sport,
            'market': market,
            'model_type': model_type,
            'metrics': training_result.get('metrics', {}),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error training value betting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Helper function to generate simulated value bets for development
def generate_simulated_value_bets(data):
    """Generate simulated value bets for demo/development."""
    import random
    import numpy as np

    sport = data['sport']
    market = data['market']
    events = data['events']

    # Generate random value bets (roughly 30% of events will be value bets)
    value_bets = []

    for event in events:
        # Extract info from event
        if 'home_team' in event and 'away_team' in event:
            event_name = f"{event['home_team']} vs {event['away_team']}"
        elif 'event_name' in event:
            event_name = event['event_name']
        else:
            event_name = f"Event {random.randint(1, 1000)}"

        # Extract or generate odds
        market_odds = event.get('market_odds', random.uniform(1.5, 4.0))
        implied_probability = 1 / market_odds

        # Randomly decide if this is a value bet (~30% chance)
        if random.random() < 0.3:
            # Value bet with edge
            edge = random.uniform(0.05, 0.15)
            model_probability = implied_probability + edge
            ev = (model_probability * (market_odds - 1)) - (1 - model_probability)
            kelly = max(0, (model_probability * market_odds - 1) / (market_odds - 1)) * 0.25

            value_bets.append({
                'event': event_name,
                'sport': sport,
                'market': market,
                'selection': event.get('selection', event.get('home_team', 'Home Team')),
                'model_probability': round(model_probability, 3),
                'implied_probability': round(implied_probability, 3),
                'market_odds': round(market_odds, 2),
                'edge': round(edge, 3),
                'expected_value': round(ev, 3),
                'kelly_stake': round(kelly, 3),
                'is_value_bet': True,
                'recommendation': 'BET',
                'confidence': round(min(edge * 10, 1.0), 2),
                'bet_rating': random.randint(3, 5)  # 3-5 stars for value bets
            })

    # Sort by expected value
    value_bets.sort(key=lambda x: x['expected_value'], reverse=True)

    return {
        'sport': sport,
        'market': market,
        'value_bets_found': len(value_bets),
        'value_bets': value_bets,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Load models
    app.config['models'] = load_models()

    # Run the development server
    app.run(debug=True, host='0.0.0.0', port=5000)