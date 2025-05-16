#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Player Intelligence Engine Test Script

This script demonstrates the capabilities of the Player Intelligence Engine,
including athlete readiness calculation, matchup analysis, and trend detection.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the Player Intelligence Engine
from src.core.player_intelligence import player_intelligence

def test_athlete_readiness():
    """Test athlete readiness score calculation."""
    print("\n===== ATHLETE READINESS SCORES =====")

    # Test with different athletes
    athletes = ["lionelmessi_soccer", "cristianoronaldo_soccer", "lebronjames_basketball"]

    for athlete_id in athletes:
        # Get athlete profile
        profile = player_intelligence.get_athlete_profile(athlete_id)

        if not profile:
            print(f"Profile for {athlete_id} not found")
            continue

        # Calculate readiness score
        readiness = player_intelligence.calculate_readiness_score(athlete_id)

        print(f"\n{profile.get('name')} ({profile.get('sport')}):")
        print(f"  Readiness Score: {readiness}/100")
        print(f"  Current Status: {profile.get('metrics', {}).get('injury_status', 'unknown')}")
        print(f"  Fatigue Level: {profile.get('metrics', {}).get('fatigue_level', 'N/A')}")

        # Add some sport-specific insights
        sport = profile.get('sport', '').lower()
        if sport == 'soccer':
            print(f"  Recent Form: {profile.get('metrics', {}).get('recent_form', 0) * 100:.1f}%")
            print(f"  Work Rate: {profile.get('metrics', {}).get('work_rate_km_per_game', 0)} km/game")
        elif sport == 'basketball':
            print(f"  PER: {profile.get('metrics', {}).get('PER', 0)}")
            print(f"  Minutes per Game: {profile.get('metrics', {}).get('minutes_per_game', 0)}")

def test_injury_probability():
    """Test injury probability calculation."""
    print("\n===== INJURY PROBABILITY =====")

    # Test with different athletes
    athletes = ["lionelmessi_soccer", "cristianoronaldo_soccer", "lebronjames_basketball"]

    for athlete_id in athletes:
        # Get athlete profile
        profile = player_intelligence.get_athlete_profile(athlete_id)

        if not profile:
            print(f"Profile for {athlete_id} not found")
            continue

        # Calculate injury probability
        injury_prob = player_intelligence.calculate_injury_probability(athlete_id)

        # Determine risk level
        risk_level = "Low"
        if injury_prob > 70:
            risk_level = "High"
        elif injury_prob > 40:
            risk_level = "Medium"

        print(f"\n{profile.get('name')} ({profile.get('sport')}):")
        print(f"  Injury Probability: {injury_prob}%")
        print(f"  Risk Level: {risk_level}")

        # List recent injuries
        injuries = profile.get('injury_history', [])
        if injuries:
            recent_injuries = sorted(injuries, key=lambda x: x.get('date', ''), reverse=True)[:2]
            print("  Recent Injuries:")
            for injury in recent_injuries:
                print(f"    - {injury.get('type')} ({injury.get('date')}): {injury.get('severity')}, {injury.get('days_out')} days out")

def test_matchup_analysis():
    """Test matchup analysis between athletes."""
    print("\n===== MATCHUP ANALYSIS =====")

    # Test soccer matchup
    print("\n• Soccer Matchup Analysis:")
    soccer_matchup = player_intelligence.calculate_matchup_rating(
        "lionelmessi_soccer",
        "cristianoronaldo_soccer",
        {"surface": "grass"}
    )

    if soccer_matchup:
        print(f"  {soccer_matchup.get('athlete1', {}).get('name')} vs {soccer_matchup.get('athlete2', {}).get('name')}")
        print(f"  Advantage: {soccer_matchup.get('advantage')} (Score: {soccer_matchup.get('advantage_score')})")

        # Print key metric comparisons
        print("  Key Metric Comparisons:")
        for metric, comparison in list(soccer_matchup.get('metric_comparisons', {}).items())[:5]:
            adv = comparison.get('advantage', 0)
            better = "1" if adv > 0 else "2" if adv < 0 else "Tie"
            print(f"    - {metric}: {comparison.get('athlete1_value')} vs {comparison.get('athlete2_value')} (Advantage: {better})")

        # Print context factors
        context_factors = soccer_matchup.get('context_factors', [])
        if context_factors:
            print("  Context Factors:")
            for factor in context_factors:
                print(f"    - {factor.get('factor')}: {factor.get('description')}")

    # Test cross-sport matchup (should return error)
    print("\n• Cross-Sport Matchup Analysis:")
    cross_sport_matchup = player_intelligence.calculate_matchup_rating(
        "lionelmessi_soccer",
        "lebronjames_basketball"
    )

    if 'error' in cross_sport_matchup:
        print(f"  Error: {cross_sport_matchup.get('error')}")
        print(f"  {cross_sport_matchup.get('athlete1_sport')} vs {cross_sport_matchup.get('athlete2_sport')}")

def test_metric_trends():
    """Test metric trend analysis."""
    print("\n===== METRIC TRENDS =====")

    # Test with Messi's data
    athlete_id = "lionelmessi_soccer"
    profile = player_intelligence.get_athlete_profile(athlete_id)

    if not profile:
        print(f"Profile for {athlete_id} not found")
        return

    # Define metrics to analyze
    metrics = ["speed_kmh", "work_rate_km_per_game", "recent_form", "fatigue_level"]

    # Get trends
    trends = player_intelligence.get_metric_trends(athlete_id, metrics)

    print(f"\n{profile.get('name')} Metric Trends:")

    for metric, trend_data in trends.items():
        percent_change = trend_data.get('percent_change', 0)
        direction = "improved" if percent_change > 0 else "declined" if percent_change < 0 else "unchanged"

        print(f"  • {metric.replace('_', ' ').title()}: {abs(percent_change):.1f}% {direction}")
        print(f"    Values: {trend_data.get('values')}")

    # Plot trends if there's data
    if trends and any(len(t.get('values', [])) > 0 for t in trends.values()):
        plt.figure(figsize=(10, 6))

        for i, (metric, trend_data) in enumerate(trends.items()):
            values = trend_data.get('values', [])
            if values:
                plt.subplot(2, 2, i+1)
                plt.plot(values, marker='o')
                plt.title(metric.replace('_', ' ').title())
                plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(project_root, "output", "metric_trends.png"))
        print("\n  Trend plot saved to output/metric_trends.png")

def test_sport_relevant_metrics():
    """Test extraction of sport-relevant metrics."""
    print("\n===== SPORT-RELEVANT METRICS =====")

    # Test with different sports
    athletes = [
        ("lionelmessi_soccer", "Soccer"),
        ("lebronjames_basketball", "Basketball")
    ]

    for athlete_id, sport in athletes:
        profile = player_intelligence.get_athlete_profile(athlete_id)

        if not profile:
            print(f"Profile for {athlete_id} not found")
            continue

        # Extract relevant metrics
        relevant_data = player_intelligence.extract_sport_relevant_metrics(profile)

        print(f"\n{profile.get('name')} ({sport}) Relevant Metrics:")

        # Print top metrics
        relevant_metrics = relevant_data.get('relevant_metrics', {})
        if relevant_metrics:
            for metric, value in relevant_metrics.items():
                print(f"  • {metric}: {value}")
        else:
            print("  No relevant metrics found")

def main():
    """Main test function."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("\nPlayer Intelligence Engine Test")
    print("===============================")

    # Run tests
    test_athlete_readiness()
    test_injury_probability()
    test_matchup_analysis()
    test_metric_trends()
    test_sport_relevant_metrics()

    print("\nTests completed!")

if __name__ == "__main__":
    main()