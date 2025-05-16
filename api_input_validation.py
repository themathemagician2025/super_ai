#!/usr/bin/env python3
"""
API Input Validation Models

This module contains Pydantic models for validating API input data.
"""

from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass
from typing import List, Dict, Optional, Union
from datetime import datetime

@dataclass
class FinancialDataQuery:
    """Validation model for financial data query parameters"""
    currency_pair: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = Field(50, ge=1, le=1000)

    @validator('currency_pair')
    def validate_currency_pair(cls, v):
        if v and '/' not in v:
            raise ValueError("Currency pair must be in format 'BASE/QUOTE', e.g. 'EUR/USD'")
        return v

@dataclass
class PredictionRequest:
    """Validation model for prediction requests"""
    data_points: List[Dict[str, Union[str, float, int]]]
    model_type: str
    confidence_threshold: Optional[float] = Field(0.7, ge=0.1, le=1.0)

    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ['forex', 'sports', 'betting', 'combined']
        if v.lower() not in allowed_models:
            raise ValueError(f"Model type must be one of: {', '.join(allowed_models)}")
        return v.lower()

@dataclass
class FileUploadMetadata:
    """Validation model for file upload metadata"""
    file_type: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None

    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = ['forex_data', 'sports_data', 'training_data']
        if v not in allowed_types:
            raise ValueError(f"File type must be one of: {', '.join(allowed_types)}")
        return v

class SportsPredictionRequest(BaseModel):
    """Validation model for sports prediction requests"""
    home_team: str
    away_team: str
    league: str
    sport: str = Field(..., description="Sport type")
    match_date: Optional[str] = None
    include_players: Optional[bool] = Field(False, description="Include player-level analysis")

    @validator('sport')
    def validate_sport(cls, v):
        allowed_sports = [
            'football', 'american_football', 'basketball', 'baseball', 'hockey', 'ice_hockey',
            'soccer', 'tennis', 'golf', 'boxing', 'mma', 'cricket', 'rugby_union', 'rugby_league',
            'australian_rules', 'afl', 'lacrosse', 'volleyball', 'handball', 'snooker', 'darts',
            'table_tennis', 'badminton'
        ]
        if v.lower().replace(' ', '_') not in allowed_sports:
            raise ValueError(f"Sport must be one of the supported types")
        return v.lower().replace(' ', '_')

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
