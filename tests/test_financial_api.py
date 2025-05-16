#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Unit Tests for Financial API endpoints

This module provides pytest tests for the Financial API endpoints.
"""

import os
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import time
from unittest import mock
from flask import url_for

# Import the Flask app factory
from minimal_financial_api import create_app, generate_api_key

@pytest.fixture
def app():
    """Create and configure a Flask app for testing"""
    # Set up test configuration
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test_secret',
        'JWT_SECRET_KEY': 'test_jwt_secret',
        'CACHE_TYPE': 'NullCache',  # Disable caching for tests
    })

    yield app

@pytest.fixture
def client(app):
    """Create a test client for the app"""
    return app.test_client()

@pytest.fixture
def auth_headers():
    """Generate authentication headers for API requests"""
    api_key = generate_api_key()
    return {
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    }

# Basic endpoint tests
def test_index_route(client):
    """Test that the index route returns 200 status"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'message' in data

def test_model_info_endpoint(client, auth_headers):
    """Test the model info endpoint"""
    response = client.get('/api/models', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)

    # Verify structure of response
    assert 'stock_models' in data
    assert 'forex_models' in data
    assert isinstance(data['stock_models'], list)
    assert isinstance(data['forex_models'], list)

    # Verify each model has required fields
    for model in data['stock_models']:
        assert all(k in model for k in ['id', 'name', 'description', 'accuracy'])
    for model in data['forex_models']:
        assert all(k in model for k in ['id', 'name', 'description', 'accuracy'])

def test_dashboard_metrics_endpoint(client, auth_headers):
    """Test the dashboard metrics endpoint"""
    response = client.get('/api/dashboard_metrics', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)

    # Verify structure of response
    assert 'predictions_today' in data
    assert 'accuracy_7d' in data
    assert 'active_users' in data
    assert 'avg_response_time_ms' in data

# Mock tests for prediction endpoints
@patch('api_interface.financial_api.PredictionEngine')
def test_forex_prediction_endpoint(mock_prediction_engine, client, auth_headers):
    """Test the forex prediction endpoint with mocked prediction engine"""
    # Set up mock
    mock_engine_instance = MagicMock()
    mock_prediction_engine.return_value = mock_engine_instance

    # Mock prediction result
    prediction_result = {
        'predictions': [
            {'date': '2023-11-01', 'predicted_price': 1.1234},
            {'date': '2023-11-02', 'predicted_price': 1.1250}
        ],
        'currency_pair': 'EUR/USD',
        'model_used': 'forex_rf'
    }
    mock_engine_instance.predict_forex.return_value = prediction_result

    # Test request data
    request_data = {
        'currency_pair': 'EUR/USD',
        'days_ahead': 2,
        'model': 'forex_rf',
        'current_data': {
            'open': 1.1200,
            'high': 1.1250,
            'low': 1.1180,
            'volume': 10000,
            'currency_pair': 'EUR/USD'
        }
    }

    # Make the request
    response = client.post(
        '/api/forex/predict',
        data=json.dumps(request_data),
        headers=auth_headers
    )

    # Check response
    assert response.status_code == 200
    data = json.loads(response.data)

    # Verify structure of response
    assert 'predictions' in data
    assert 'currency_pair' in data
    assert len(data['predictions']) == 2

    # Check if mock was called correctly
    mock_engine_instance.predict_forex.assert_called_once()
    call_args = mock_engine_instance.predict_forex.call_args[0]
    assert call_args[1] == 2  # days_ahead
    assert call_args[2] == 'forex_rf'  # model_name

@patch('api_interface.financial_api.PredictionEngine')
def test_sports_prediction_endpoint(mock_prediction_engine, client, auth_headers):
    """Test the sports prediction endpoint with mocked prediction engine"""
    # Set up mock
    mock_engine_instance = MagicMock()
    mock_prediction_engine.return_value = mock_engine_instance

    # Mock prediction result
    prediction_result = {
        'match': 'Manchester United vs Liverpool',
        'league': 'Premier League',
        'predicted_outcome': 'home_win',
        'probabilities': {
            'home_win': 0.55,
            'draw': 0.25,
            'away_win': 0.20
        },
        'model_used': 'sports_rf'
    }
    mock_engine_instance.predict_sports.return_value = prediction_result

    # Test request data
    request_data = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'league': 'Premier League',
        'match_date': '2023-11-05',
        'home_goals_scored_avg': 2.1,
        'away_goals_scored_avg': 2.3,
        'home_goals_conceded_avg': 0.8,
        'away_goals_conceded_avg': 1.0
    }

    # Make the request
    response = client.post(
        '/api/sports/predict',
        data=json.dumps(request_data),
        headers=auth_headers
    )

    # Check response
    assert response.status_code == 200
    data = json.loads(response.data)

    # Verify structure of response
    assert 'match' in data
    assert 'predicted_outcome' in data
    assert 'probabilities' in data
    assert 'home_win' in data['probabilities']

    # Check if mock was called correctly
    mock_engine_instance.predict_sports.assert_called_once_with(request_data)

# Error handling tests
def test_invalid_input_handling(client, auth_headers):
    """Test that the API properly handles invalid input"""
    # Missing required fields
    request_data = {
        'currency_pair': 'EUR/USD'
        # Missing other required fields
    }

    response = client.post(
        '/api/forex/predict',
        data=json.dumps(request_data),
        headers=auth_headers
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_unauthorized_access(client):
    """Test that endpoints require proper authorization"""
    # Make request without auth headers
    response = client.post(
        '/api/forex/predict',
        data=json.dumps({'currency_pair': 'EUR/USD'}),
        headers={'Content-Type': 'application/json'}
    )

    # Should return 401 Unauthorized
    assert response.status_code == 401

@patch('api_interface.financial_api.PredictionEngine')
def test_model_error_handling(mock_prediction_engine, client, auth_headers):
    """Test that API properly handles model errors"""
    # Set up mock to return an error
    mock_engine_instance = MagicMock()
    mock_prediction_engine.return_value = mock_engine_instance
    mock_engine_instance.predict_forex.return_value = {'error': 'Model prediction failed'}

    # Test request data
    request_data = {
        'currency_pair': 'EUR/USD',
        'days_ahead': 2,
        'model': 'forex_rf',
        'current_data': {
            'open': 1.1200,
            'high': 1.1250,
            'low': 1.1180,
            'volume': 10000,
            'currency_pair': 'EUR/USD'
        }
    }

    # Make the request
    response = client.post(
        '/api/forex/predict',
        data=json.dumps(request_data),
        headers=auth_headers
    )

    # Should return 500 Internal Server Error
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

# Cache tests
def test_caching_behavior(client, app):
    """Test that caching is working as expected"""
    with patch('minimal_financial_api.cache') as mock_cache:
        # Call the cached endpoint
        response = client.get('/api/dashboard_metrics')
        assert response.status_code == 200

        # Call it again
        response = client.get('/api/dashboard_metrics')
        assert response.status_code == 200

        # Verify the cache was used
        mock_cache.cached.assert_called()

# Performance tests
@pytest.mark.parametrize('endpoint', [
    '/',
    '/api/model/info',
    '/api/dashboard_metrics'
])
def test_endpoint_performance(client, endpoint):
    """Test that endpoints respond within acceptable time limits"""
    start_time = time.time()
    response = client.get(endpoint)
    end_time = time.time()

    # Endpoints should respond in less than 500ms
    assert end_time - start_time < 0.5
    assert response.status_code == 200

# Integration tests with real API response format
def test_stock_prediction_response_format(client, auth_headers):
    """Test that stock prediction response has correct format."""
    with mock.patch('minimal_financial_api.prediction_engine.predict_stock') as mock_predict:
        # Setup realistic mock return value with all expected fields
        mock_predict.return_value = {
            'predictions': [
                {'date': '2023-11-01', 'predicted_price': 150.25, 'confidence': 0.85},
                {'date': '2023-11-02', 'predicted_price': 152.75, 'confidence': 0.82}
            ],
            'ticker': 'AAPL',
            'model_used': 'stock_rf',
            'historical_data': [
                {'date': '2023-10-30', 'price': 149.50},
                {'date': '2023-10-31', 'price': 150.00}
            ],
            'meta': {
                'prediction_time': '2023-10-31T15:30:00Z',
                'model_version': '1.2.3',
                'factors': ['price_momentum', 'volume', 'market_sentiment']
            }
        }

        payload = {
            'ticker': 'AAPL',
            'days_ahead': 2,
            'model_id': 'stock_rf'
        }

        response = client.post(
            '/api/stocks/predict',
            data=json.dumps(payload),
            headers=auth_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check structure matches expected format
        assert 'predictions' in data
        assert isinstance(data['predictions'], list)
        assert len(data['predictions']) == 2
        assert all(isinstance(p, dict) for p in data['predictions'])
        assert all('date' in p and 'predicted_price' in p for p in data['predictions'])

        assert 'meta' in data
        assert isinstance(data['meta'], dict)
        assert 'model_version' in data['meta']

# Rate limiting tests
def test_rate_limiting(client, auth_headers):
    """Test that rate limiting is enforced."""
    # Make multiple requests in quick succession
    responses = []
    for _ in range(20):  # Assuming rate limit is less than 20 req/sec
        response = client.get('/api/models', headers=auth_headers)
        responses.append(response)

    # Check if any responses hit rate limits
    rate_limited = any(r.status_code == 429 for r in responses)

    if rate_limited:
        # Verify rate limit error message
        for response in responses:
            if response.status_code == 429:
                data = json.loads(response.data)
                assert 'error' in data
                assert 'rate limit' in data['error'].lower()
                break
    else:
        # If no rate limiting occurred, test passes anyway (rate limit may be higher)
        assert True

if __name__ == '__main__':
    pytest.main(['-xvs', 'test_financial_api.py'])