# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Database models for the Super AI Web Interface.

This module defines SQLAlchemy models for storing athlete profiles,
performance metrics, teams, matches, and other data required by the system.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, JSON, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

# Association tables for many-to-many relationships
athlete_team_association = Table(
    'athlete_team', Base.metadata,
    Column('athlete_id', String(50), ForeignKey('athletes.id')),
    Column('team_id', String(50), ForeignKey('teams.id')),
    Column('start_date', DateTime, nullable=True),
    Column('end_date', DateTime, nullable=True),
    Column('is_current', Boolean, default=False)
)

match_athlete_association = Table(
    'match_athlete', Base.metadata,
    Column('match_id', String(50), ForeignKey('matches.id')),
    Column('athlete_id', String(50), ForeignKey('athletes.id')),
    Column('playing_status', String(20), default='active'),  # active, reserve, injured, etc.
    Column('minutes_played', Integer, nullable=True)
)

class Athlete(Base):
    """Model to store athlete information and basic profile"""
    __tablename__ = 'athletes'

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=True)
    sport = Column(String(50), nullable=False)
    position = Column(String(50), nullable=True)
    country = Column(String(50), nullable=True)
    age = Column(Integer, nullable=True)
    height = Column(Float, nullable=True)  # cm
    weight = Column(Float, nullable=True)  # kg
    active = Column(Boolean, default=True)
    professional_since = Column(Integer, nullable=True)
    profile_image_url = Column(String(500), nullable=True)

    # Relationships
    teams = relationship("Team", secondary=athlete_team_association, backref="athletes")
    matches = relationship("Match", secondary=match_athlete_association, backref="athletes")
    physical_metrics = relationship("PhysicalMetrics", backref="athlete", uselist=False)
    technical_metrics = relationship("TechnicalMetrics", backref="athlete", uselist=False)
    tactical_metrics = relationship("TacticalMetrics", backref="athlete", uselist=False)
    mental_metrics = relationship("MentalMetrics", backref="athlete", uselist=False)
    health_metrics = relationship("HealthMetrics", backref="athlete", uselist=False)
    advanced_metrics = relationship("AdvancedMetrics", backref="athlete", uselist=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_sources = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Athlete {self.name} ({self.sport})>"

class PhysicalMetrics(Base):
    """Model to store athlete physical performance metrics"""
    __tablename__ = 'physical_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Physical metrics
    max_speed = Column(Float, nullable=True)  # km/h
    acceleration = Column(Float, nullable=True)  # m/s²
    vo2_max = Column(Float, nullable=True)  # mL/kg/min
    agility_score = Column(Float, nullable=True)  # arbitrary 1-10
    power_output = Column(Float, nullable=True)  # watts
    jump_height = Column(Float, nullable=True)  # cm
    strength_rating = Column(Float, nullable=True)  # arbitrary 1-10
    reaction_time = Column(Integer, nullable=True)  # milliseconds
    fatigue_index = Column(Float, nullable=True)  # % drop
    distance_covered_avg = Column(Float, nullable=True)  # km per game
    heart_rate_avg = Column(Integer, nullable=True)  # bpm
    heart_rate_max = Column(Integer, nullable=True)  # bpm
    work_rate = Column(Float, nullable=True)  # m/min

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<PhysicalMetrics for Athlete {self.athlete_id}>"

class TechnicalMetrics(Base):
    """Model to store athlete technical performance metrics"""
    __tablename__ = 'technical_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Technical metrics
    accuracy = Column(Float, nullable=True)  # %
    possession_time_avg = Column(Float, nullable=True)  # minutes per game
    scoring_efficiency = Column(Float, nullable=True)  # %
    assist_rate = Column(Float, nullable=True)  # per game
    tackle_success_rate = Column(Float, nullable=True)  # %
    serve_velocity = Column(Float, nullable=True)  # km/h
    dribbling_success = Column(Float, nullable=True)  # %
    blocking_efficiency = Column(Float, nullable=True)  # per game
    error_rate = Column(Float, nullable=True)  # %
    shot_distance_avg = Column(Float, nullable=True)  # meters

    # Sport-specific fields stored as JSON
    sport_specific = Column(JSON, nullable=True)

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TechnicalMetrics for Athlete {self.athlete_id}>"

class TacticalMetrics(Base):
    """Model to store athlete tactical performance metrics"""
    __tablename__ = 'tactical_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Tactical metrics
    pass_selection_rating = Column(Float, nullable=True)  # arbitrary 1-10
    defensive_positioning = Column(Float, nullable=True)  # arbitrary 1-10
    offensive_contribution = Column(Float, nullable=True)  # xG+xA per game
    expected_goals = Column(Float, nullable=True)  # xG per game
    expected_assists = Column(Float, nullable=True)  # xA per game
    game_intelligence = Column(Float, nullable=True)  # arbitrary 1-10
    transition_speed = Column(Float, nullable=True)  # seconds
    zone_entries_avg = Column(Float, nullable=True)  # per game

    # Heat map data stored as JSON
    heat_map_data = Column(JSON, nullable=True)  # Spatial data

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TacticalMetrics for Athlete {self.athlete_id}>"

class MentalMetrics(Base):
    """Model to store athlete mental performance metrics"""
    __tablename__ = 'mental_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Mental metrics
    composure_rating = Column(Float, nullable=True)  # arbitrary 1-10
    work_ethic_rating = Column(Float, nullable=True)  # arbitrary 1-10
    leadership_impact = Column(Float, nullable=True)  # plus/minus
    focus_rating = Column(Float, nullable=True)  # arbitrary 1-10
    resilience_score = Column(Float, nullable=True)  # %

    # Additional context
    high_pressure_performance = Column(Float, nullable=True)  # performance in critical moments

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MentalMetrics for Athlete {self.athlete_id}>"

class HealthMetrics(Base):
    """Model to store athlete health and injury metrics"""
    __tablename__ = 'health_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Health metrics
    injury_frequency = Column(Float, nullable=True)  # per season
    current_injury_status = Column(String(20), nullable=True)  # Healthy, Day-to-Day, Out, IR, etc.
    recovery_time_avg = Column(Float, nullable=True)  # weeks
    muscle_asymmetry = Column(Float, nullable=True)  # %
    joint_stability_rating = Column(Float, nullable=True)  # arbitrary 1-10
    games_missed_last_season = Column(Integer, nullable=True)

    # Injury history stored as JSON
    injury_history = Column(JSON, nullable=True)

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<HealthMetrics for Athlete {self.athlete_id}>"

class AdvancedMetrics(Base):
    """Model to store athlete advanced analytics metrics (sport-specific)"""
    __tablename__ = 'advanced_metrics'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)

    # Common advanced metrics across sports
    efficiency_rating = Column(Float, nullable=True)
    value_above_replacement = Column(Float, nullable=True)
    clutch_performance = Column(Float, nullable=True)
    consistency_rating = Column(Float, nullable=True)

    # Sport-specific advanced metrics stored as JSON
    sport_metrics = Column(JSON, nullable=True)

    # Metadata
    measured_at = Column(DateTime, nullable=True)  # When these metrics were last measured
    source = Column(String(50), nullable=True)  # e.g., 'scraped', 'estimated', 'projected'
    confidence = Column(Float, nullable=True)  # How confident are we in these metrics (0-1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AdvancedMetrics for Athlete {self.athlete_id}>"

class Team(Base):
    """Model to store team information and profile"""
    __tablename__ = 'teams'

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=True)
    sport = Column(String(50), nullable=False)
    league = Column(String(100), nullable=True)
    country = Column(String(50), nullable=True)
    founded = Column(Integer, nullable=True)
    current_ranking = Column(Integer, nullable=True)
    home_venue = Column(String(100), nullable=True)
    logo_url = Column(String(500), nullable=True)

    # Team stats and profile as JSON
    performance_metrics = Column(JSON, nullable=True)
    roster_stats = Column(JSON, nullable=True)
    tactical_profile = Column(JSON, nullable=True)
    historical_performance = Column(JSON, nullable=True)
    team_health = Column(JSON, nullable=True)

    # Relationships are defined via backref from athlete_team_association

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_sources = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<Team {self.name} ({self.sport})>"

class Match(Base):
    """Model to store match information and predictions"""
    __tablename__ = 'matches'

    id = Column(String(50), primary_key=True)
    sport = Column(String(50), nullable=False)
    league = Column(String(100), nullable=True)

    # Teams
    home_team_id = Column(String(50), ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(String(50), ForeignKey('teams.id'), nullable=False)
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    # Match details
    match_datetime = Column(DateTime, nullable=False)
    venue = Column(String(100), nullable=True)
    status = Column(String(20), default='scheduled')  # scheduled, live, completed, cancelled

    # Results (if completed)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Prediction data stored as JSON
    prediction = Column(JSON, nullable=True)
    betting_odds = Column(JSON, nullable=True)
    key_players = Column(JSON, nullable=True)
    weather = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Match {self.home_team_id} vs {self.away_team_id} ({self.match_datetime})>"

class DataSource(Base):
    """Model to store information about data sources for athlete metrics"""
    __tablename__ = 'data_sources'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    source_type = Column(String(50), nullable=False)  # scraped, estimated, projected, etc.
    url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    reliability_score = Column(Float, nullable=True)  # 0-1 reliability score
    last_updated = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DataSource {self.name} ({self.source_type})>"

class AthleteIdentification(Base):
    """Model to store athlete identification and name variants"""
    __tablename__ = 'athlete_identification'

    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(50), ForeignKey('athletes.id'), nullable=False)
    name_variant = Column(String(200), nullable=False)
    source = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)  # 0-1 confidence score

    # Relationship
    athlete = relationship("Athlete", backref="name_variants")

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AthleteIdentification {self.name_variant} -> {self.athlete_id}>"
