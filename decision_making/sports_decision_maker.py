# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..prediction.sports.football_predictor import FootballPredictor

logger = logging.getLogger(__name__)

class SportsBettingDecisionMaker:
    """Decision maker for sports betting using AI predictions."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.football_predictor = FootballPredictor(config_path)
        self.confidence_threshold = 0.65  # Minimum confidence for betting decisions
        self.risk_tolerance = 0.3  # Risk tolerance factor (0-1)
        self.max_stake_ratio = 0.1  # Maximum stake as ratio of bankroll
        
        # Load configuration if provided
        if config_path:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
                self.risk_tolerance = config.get('risk_tolerance', self.risk_tolerance)
                self.max_stake_ratio = config.get('max_stake_ratio', self.max_stake_ratio)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def initialize_predictor(self, historical_data_path: str):
        """Initialize the football predictor with historical data."""
        self.football_predictor.load_historical_data(historical_data_path)
    
    def analyze_match(self, home_team: str, away_team: str) -> Dict:
        """Analyze a match and make betting decisions."""
        # Get match prediction
        prediction = self.football_predictor.predict_match(home_team, away_team)
        
        # Get team form analysis
        home_form = self.football_predictor.get_team_form_analysis(home_team)
        away_form = self.football_predictor.get_team_form_analysis(away_team)
        
        # Get head-to-head history
        h2h_history = self.football_predictor.get_head_to_head(home_team, away_team)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(prediction, home_form, away_form, h2h_history)
        
        # Make betting decisions
        betting_decisions = self._make_betting_decisions(prediction, confidence_scores)
        
        return {
            'match_prediction': prediction,
            'confidence_scores': confidence_scores,
            'betting_decisions': betting_decisions,
            'team_analysis': {
                'home_team': home_form,
                'away_team': away_form
            },
            'head_to_head': h2h_history
        }
    
    def _calculate_confidence_scores(
        self,
        prediction: Dict,
        home_form: Dict,
        away_form: Dict,
        h2h_history: List[Dict]
    ) -> Dict[str, float]:
        """Calculate confidence scores for different betting options."""
        
        # Calculate basic confidence from probabilities
        home_win_confidence = prediction['home_win_probability']
        away_win_confidence = prediction['away_win_probability']
        draw_confidence = prediction['draw_probability']
        
        # Adjust confidence based on recent form
        if 'recent_form' in home_form and 'recent_form' in away_form:
            home_form_factor = sum(home_form['recent_form']) / len(home_form['recent_form']) if home_form['recent_form'] else 0
            away_form_factor = sum(away_form['recent_form']) / len(away_form['recent_form']) if away_form['recent_form'] else 0
            
            home_win_confidence *= (1 + home_form_factor * 0.1)
            away_win_confidence *= (1 + away_form_factor * 0.1)
        
        # Adjust confidence based on head-to-head history
        h2h_home_wins = sum(1 for m in h2h_history if m['home_team'] == home_form and m['home_score'] > m['away_score'])
        h2h_away_wins = sum(1 for m in h2h_history if m['away_team'] == away_form and m['away_score'] > m['home_score'])
        
        if h2h_history:
            h2h_factor = 0.05  # Impact factor of head-to-head history
            home_win_confidence *= (1 + (h2h_home_wins / len(h2h_history)) * h2h_factor)
            away_win_confidence *= (1 + (h2h_away_wins / len(h2h_history)) * h2h_factor)
        
        # Normalize confidence scores
        total_confidence = home_win_confidence + away_win_confidence + draw_confidence
        if total_confidence > 0:
            home_win_confidence /= total_confidence
            away_win_confidence /= total_confidence
            draw_confidence /= total_confidence
        
        return {
            'home_win': home_win_confidence,
            'away_win': away_win_confidence,
            'draw': draw_confidence
        }
    
    def _make_betting_decisions(
        self,
        prediction: Dict,
        confidence_scores: Dict[str, float]
    ) -> Dict:
        """Make betting decisions based on predictions and confidence scores."""
        decisions = {
            'recommended_bets': [],
            'stake_recommendations': {},
            'risk_assessment': {}
        }
        
        # Assess each betting option
        betting_options = [
            ('home_win', confidence_scores['home_win'], prediction['home_win_probability']),
            ('away_win', confidence_scores['away_win'], prediction['away_win_probability']),
            ('draw', confidence_scores['draw'], prediction['draw_probability'])
        ]
        
        for bet_type, confidence, probability in betting_options:
            # Calculate risk score (0-1)
            risk_score = 1 - confidence
            
            # Determine if confidence meets threshold
            if confidence >= self.confidence_threshold:
                decisions['recommended_bets'].append(bet_type)
                
                # Calculate recommended stake based on confidence and risk tolerance
                stake_ratio = min(
                    self.max_stake_ratio,
                    confidence * self.risk_tolerance
                )
                
                decisions['stake_recommendations'][bet_type] = {
                    'stake_ratio': stake_ratio,
                    'confidence': confidence
                }
                
                decisions['risk_assessment'][bet_type] = {
                    'risk_score': risk_score,
                    'probability': probability
                }
        
        return decisions
    
    def get_decision_explanation(self, analysis_result: Dict) -> str:
        """Generate a human-readable explanation of the betting decisions."""
        explanation = []
        
        # Add match prediction summary
        pred = analysis_result['match_prediction']
        explanation.append(f"Match Prediction Summary:")
        explanation.append(f"- Home win probability: {pred['home_win_probability']:.2%}")
        explanation.append(f"- Draw probability: {pred['draw_probability']:.2%}")
        explanation.append(f"- Away win probability: {pred['away_win_probability']:.2%}")
        explanation.append(f"- Predicted score: {pred['predicted_home_goals']}-{pred['predicted_away_goals']}")
        
        # Add betting recommendations
        decisions = analysis_result['betting_decisions']
        if decisions['recommended_bets']:
            explanation.append("\nRecommended Bets:")
            for bet in decisions['recommended_bets']:
                stake_info = decisions['stake_recommendations'][bet]
                risk_info = decisions['risk_assessment'][bet]
                explanation.append(
                    f"- {bet.replace('_', ' ').title()}: "
                    f"Stake {stake_info['stake_ratio']:.1%} of bankroll, "
                    f"Confidence {stake_info['confidence']:.1%}, "
                    f"Risk Score {risk_info['risk_score']:.2f}"
                )
        else:
            explanation.append("\nNo bets recommended for this match.")
        
        # Add form analysis
        home_form = analysis_result['team_analysis']['home_team']
        away_form = analysis_result['team_analysis']['away_team']
        
        explanation.append("\nTeam Form Analysis:")
        explanation.append(f"Home team:")
        explanation.append(f"- Win rate: {home_form['overall_win_rate']:.2%}")
        explanation.append(f"- Recent form: {home_form['recent_form']}")
        explanation.append(f"- Goals per game: {home_form['goals_per_game']:.2f}")
        
        explanation.append(f"\nAway team:")
        explanation.append(f"- Win rate: {away_form['overall_win_rate']:.2%}")
        explanation.append(f"- Recent form: {away_form['recent_form']}")
        explanation.append(f"- Goals per game: {away_form['goals_per_game']:.2f}")
        
        return "\n".join(explanation) 