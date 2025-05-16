# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Risk Management Module

This module provides tools for:
1. Tracking prediction strategies and their performance
2. Calculating risk exposure based on confidence levels and historical accuracy
3. Implementing risk mitigation measures for high-risk scenarios
4. Generating risk reports and visualizations
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class RiskAssessment:
    """Risk assessment data structure"""
    risk_level: RiskLevel
    confidence: float
    exposure: float
    factors: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'risk_level': self.risk_level.name,
            'confidence': self.confidence,
            'exposure': self.exposure,
            'factors': self.factors,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskAssessment':
        """Create instance from dictionary"""
        return cls(
            risk_level=RiskLevel[data['risk_level']],
            confidence=data['confidence'],
            exposure=data['exposure'],
            factors=data['factors'],
            recommendations=data['recommendations'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class StrategyMetrics:
    """Metrics for evaluating prediction strategies"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_loss: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'profit_loss': self.profit_loss,
            'roi': self.roi,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyMetrics':
        """Create instance from dictionary"""
        return cls(**data)

class StrategyTracker:
    """Tracks and evaluates prediction strategies"""

    def __init__(self, strategy_name: str, config: Dict = None):
        """
        Initialize strategy tracker

        Args:
            strategy_name: Name of the strategy to track
            config: Configuration dictionary
        """
        self.strategy_name = strategy_name
        self.config = config or {}

        # Prediction history
        self.predictions = []
        self.outcomes = []
        self.timestamps = []
        self.confidences = []
        self.metadata = []

        # Performance metrics
        self.metrics = None
        self.last_updated = None

        # Risk tracking
        self.risk_assessments = []

        # Data paths
        self.data_dir = Path(self.config.get('data_dir', 'data/strategies'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data if available
        self._load_history()

        logger.info(f"Initialized StrategyTracker for {strategy_name}")

    def record_prediction(self, prediction: Any, confidence: float,
                         metadata: Dict = None) -> int:
        """
        Record a new prediction

        Args:
            prediction: The prediction value
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional information about the prediction

        Returns:
            Prediction ID (index)
        """
        pred_id = len(self.predictions)

        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(datetime.now())
        self.outcomes.append(None)  # Outcome not known yet
        self.metadata.append(metadata or {})

        logger.info(f"Recorded prediction {pred_id} for strategy {self.strategy_name}")

        # Save after each prediction
        self._save_history()

        return pred_id

    def record_outcome(self, pred_id: int, outcome: Any) -> bool:
        """
        Record the actual outcome for a prediction

        Args:
            pred_id: Prediction ID to update
            outcome: The actual outcome

        Returns:
            Success status
        """
        if pred_id < 0 or pred_id >= len(self.predictions):
            logger.error(f"Invalid prediction ID: {pred_id}")
            return False

        self.outcomes[pred_id] = outcome

        # Update metrics when outcome is recorded
        self._calculate_metrics()
        self._save_history()

        logger.info(f"Recorded outcome for prediction {pred_id} in strategy {self.strategy_name}")
        return True

    def assign_risk_assessment(self, pred_id: int, assessment: RiskAssessment) -> bool:
        """
        Assign risk assessment to a prediction

        Args:
            pred_id: Prediction ID to update
            assessment: Risk assessment

        Returns:
            Success status
        """
        if pred_id < 0 or pred_id >= len(self.predictions):
            logger.error(f"Invalid prediction ID: {pred_id}")
            return False

        # Ensure metadata dict has risk_assessment key
        if 'risk_assessment' not in self.metadata[pred_id]:
            self.metadata[pred_id]['risk_assessment'] = []

        # Add assessment
        self.metadata[pred_id]['risk_assessment'].append(assessment.to_dict())

        # Also keep track of all assessments
        self.risk_assessments.append((pred_id, assessment))

        self._save_history()

        logger.info(f"Assigned risk assessment to prediction {pred_id}")
        return True

    def get_metrics(self) -> Optional[StrategyMetrics]:
        """Get current performance metrics"""
        if not self.metrics:
            self._calculate_metrics()
        return self.metrics

    def get_prediction_history(self) -> pd.DataFrame:
        """Get prediction history as DataFrame"""
        data = {
            'prediction': self.predictions,
            'outcome': self.outcomes,
            'confidence': self.confidences,
            'timestamp': self.timestamps
        }

        # Add any consistent metadata fields as columns
        meta_keys = set()
        for meta in self.metadata:
            meta_keys.update(meta.keys())

        for key in meta_keys:
            data[f'meta_{key}'] = [meta.get(key) for meta in self.metadata]

        return pd.DataFrame(data)

    def generate_performance_chart(self, output_path: Optional[str] = None) -> str:
        """
        Generate performance chart visualization

        Args:
            output_path: Path to save the chart (or auto-generate if None)

        Returns:
            Path to the saved chart
        """
        if output_path is None:
            chart_dir = self.data_dir / "charts"
            chart_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(chart_dir / f"{self.strategy_name}_performance_{timestamp}.png")

        # Get data
        df = self.get_prediction_history()
        df = df.dropna(subset=['outcome'])  # Only include predictions with outcomes

        # Calculate cumulative performance
        df['correct'] = df.apply(lambda row: row['prediction'] == row['outcome'], axis=1)
        df['cumulative_accuracy'] = df['correct'].cumsum() / (df.index + 1)

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot accuracy over time
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['cumulative_accuracy'], 'b-', label='Accuracy')
        plt.axhline(y=df['cumulative_accuracy'].iloc[-1], color='r', linestyle='--',
                   label=f'Final Accuracy: {df["cumulative_accuracy"].iloc[-1]:.2f}')
        plt.title(f"Strategy Performance: {self.strategy_name}")
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot confidence distribution
        plt.subplot(2, 1, 2)
        correct_conf = df[df['correct']]['confidence']
        incorrect_conf = df[~df['correct']]['confidence']

        plt.hist([correct_conf, incorrect_conf], bins=10, alpha=0.7,
                label=['Correct Predictions', 'Incorrect Predictions'])
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()

        # Save and return
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _calculate_metrics(self) -> StrategyMetrics:
        """Calculate performance metrics from prediction history"""
        df = self.get_prediction_history()
        df = df.dropna(subset=['outcome'])  # Only include predictions with outcomes

        if len(df) == 0:
            logger.warning(f"No completed predictions for strategy {self.strategy_name}")
            return None

        # Basic accuracy
        df['correct'] = df.apply(lambda row: row['prediction'] == row['outcome'], axis=1)
        accuracy = df['correct'].mean()

        # Precision, recall, F1 (simplified - would depend on prediction type)
        # For this implementation, we'll use the same value as accuracy
        precision = recall = accuracy
        f1_score = accuracy

        # Win rate
        win_rate = accuracy

        # Placeholder for financial metrics (would be calculated differently based on prediction domain)
        profit_loss = 0.0
        roi = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0

        # Create metrics
        self.metrics = StrategyMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            profit_loss=profit_loss,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate
        )

        self.last_updated = datetime.now()
        return self.metrics

    def _save_history(self) -> bool:
        """Save prediction history to file"""
        data = {
            'strategy_name': self.strategy_name,
            'predictions': self.predictions,
            'outcomes': self.outcomes,
            'confidences': self.confidences,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'metadata': self.metadata,
            'last_updated': datetime.now().isoformat()
        }

        # Add metrics if available
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()

        # Save to file
        file_path = self.data_dir / f"{self.strategy_name}_history.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save strategy history: {str(e)}")
            return False

    def _load_history(self) -> bool:
        """Load prediction history from file"""
        file_path = self.data_dir / f"{self.strategy_name}_history.json"

        if not file_path.exists():
            logger.info(f"No history file found for strategy {self.strategy_name}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.predictions = data.get('predictions', [])
            self.outcomes = data.get('outcomes', [])
            self.confidences = data.get('confidences', [])
            self.timestamps = [datetime.fromisoformat(ts) for ts in data.get('timestamps', [])]
            self.metadata = data.get('metadata', [])

            # Load metrics if available
            if 'metrics' in data:
                self.metrics = StrategyMetrics.from_dict(data['metrics'])

            logger.info(f"Loaded history for strategy {self.strategy_name}: {len(self.predictions)} predictions")
            return True
        except Exception as e:
            logger.error(f"Failed to load strategy history: {str(e)}")
            return False

class RiskManager:
    """Main risk management system"""

    def __init__(self, config: Dict = None):
        """
        Initialize risk manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Strategy trackers
        self.strategy_trackers = {}

        # Risk thresholds
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75
        })

        # Risk mitigation rules
        self.mitigation_rules = self.config.get('mitigation_rules', {})

        # Data paths
        self.data_dir = Path(self.config.get('data_dir', 'data/risk'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing strategies
        self._load_strategies()

        logger.info("Initialized RiskManager")

    def get_strategy_tracker(self, strategy_name: str, create_if_missing: bool = True) -> Optional[StrategyTracker]:
        """
        Get a strategy tracker by name

        Args:
            strategy_name: Name of the strategy
            create_if_missing: Whether to create a new tracker if not found

        Returns:
            StrategyTracker instance or None if not found and not created
        """
        if strategy_name in self.strategy_trackers:
            return self.strategy_trackers[strategy_name]

        if create_if_missing:
            tracker = StrategyTracker(strategy_name, {'data_dir': str(self.data_dir / 'strategies')})
            self.strategy_trackers[strategy_name] = tracker
            return tracker

        return None

    def assess_risk(self, strategy_name: str, prediction: Any, confidence: float,
                   context_data: Dict = None) -> RiskAssessment:
        """
        Assess risk for a prediction

        Args:
            strategy_name: Name of the prediction strategy
            prediction: The prediction value
            confidence: Confidence level (0.0 to 1.0)
            context_data: Additional data for risk calculation

        Returns:
            Risk assessment
        """
        tracker = self.get_strategy_tracker(strategy_name)
        metrics = tracker.get_metrics()

        # Default risk factors
        risk_factors = {
            'confidence': 1.0 - confidence,  # Lower confidence means higher risk
            'strategy_history': 0.5,  # Default value
            'market_volatility': 0.5,  # Default value
            'data_quality': 0.5  # Default value
        }

        # Update risk factors based on metrics and context
        if metrics:
            # Historical accuracy affects risk
            accuracy_factor = 1.0 - metrics.accuracy
            risk_factors['strategy_history'] = accuracy_factor

        # Update with context-specific factors
        if context_data:
            if 'market_volatility' in context_data:
                risk_factors['market_volatility'] = context_data['market_volatility']

            if 'data_quality' in context_data:
                risk_factors['data_quality'] = context_data['data_quality']

        # Calculate overall risk score (weighted average of factors)
        weights = self.config.get('risk_factor_weights', {
            'confidence': 0.4,
            'strategy_history': 0.3,
            'market_volatility': 0.2,
            'data_quality': 0.1
        })

        risk_score = sum(risk_factors[factor] * weights.get(factor, 0.0)
                        for factor in risk_factors)

        # Determine risk level based on thresholds
        risk_level = RiskLevel.LOW
        if risk_score >= self.risk_thresholds['high']:
            risk_level = RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = RiskLevel.MEDIUM

        # Calculate exposure (how much we stand to lose)
        # This would typically be a financial value, but we'll use a proxy score
        exposure = risk_score * (1.0 + (1.0 - confidence))

        # Generate recommendations based on risk level
        recommendations = self._generate_recommendations(risk_level, risk_factors, context_data)

        # Create assessment
        assessment = RiskAssessment(
            risk_level=risk_level,
            confidence=confidence,
            exposure=exposure,
            factors=risk_factors,
            recommendations=recommendations
        )

        logger.info(f"Risk assessment for {strategy_name}: {risk_level.name} (score: {risk_score:.2f})")
        return assessment

    def track_prediction_with_risk(self, strategy_name: str, prediction: Any,
                                 confidence: float, context_data: Dict = None) -> Tuple[int, RiskAssessment]:
        """
        Record a prediction and assess its risk

        Args:
            strategy_name: Name of the prediction strategy
            prediction: The prediction value
            confidence: Confidence level (0.0 to 1.0)
            context_data: Additional data for risk calculation

        Returns:
            Tuple of (prediction ID, risk assessment)
        """
        # Get tracker
        tracker = self.get_strategy_tracker(strategy_name)

        # Assess risk
        assessment = self.assess_risk(strategy_name, prediction, confidence, context_data)

        # Record prediction with metadata including risk assessment
        metadata = context_data or {}
        metadata['risk_assessment'] = [assessment.to_dict()]

        pred_id = tracker.record_prediction(prediction, confidence, metadata)

        # Record risk assessment explicitly
        tracker.assign_risk_assessment(pred_id, assessment)

        return pred_id, assessment

    def update_prediction_outcome(self, strategy_name: str, pred_id: int,
                                outcome: Any) -> bool:
        """
        Update a prediction with its actual outcome

        Args:
            strategy_name: Name of the prediction strategy
            pred_id: Prediction ID to update
            outcome: The actual outcome

        Returns:
            Success status
        """
        tracker = self.get_strategy_tracker(strategy_name, create_if_missing=False)
        if not tracker:
            logger.error(f"Strategy not found: {strategy_name}")
            return False

        return tracker.record_outcome(pred_id, outcome)

    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """Get metrics for a strategy"""
        tracker = self.get_strategy_tracker(strategy_name, create_if_missing=False)
        if not tracker:
            logger.error(f"Strategy not found: {strategy_name}")
            return None

        return tracker.get_metrics()

    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return {
            name: tracker.get_metrics()
            for name, tracker in self.strategy_trackers.items()
            if tracker.get_metrics()
        }

    def generate_risk_report(self, timeframe_days: int = 30) -> Dict:
        """
        Generate a comprehensive risk report

        Args:
            timeframe_days: Number of days to include in the report

        Returns:
            Dictionary with report data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'timeframe_days': timeframe_days,
            'strategies': {},
            'overall_risk_profile': {
                'low_risk_count': 0,
                'medium_risk_count': 0,
                'high_risk_count': 0,
                'extreme_risk_count': 0,
                'avg_risk_score': 0.0
            }
        }

        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        risk_scores = []

        # Analyze each strategy
        for name, tracker in self.strategy_trackers.items():
            df = tracker.get_prediction_history()

            # Filter by timeframe
            df = df[df['timestamp'] >= cutoff_date]

            if len(df) == 0:
                continue

            # Get risk assessments
            risk_levels = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 0,
                RiskLevel.HIGH: 0,
                RiskLevel.EXTREME: 0
            }

            # Track correct predictions by risk level
            correct_by_risk = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 0,
                RiskLevel.HIGH: 0,
                RiskLevel.EXTREME: 0
            }

            # Count assessed predictions and outcomes by risk level
            assessed_count = 0
            outcome_count = 0

            for i, row in df.iterrows():
                meta = tracker.metadata[i]
                if 'risk_assessment' in meta and meta['risk_assessment']:
                    # Get the latest assessment
                    assessment_dict = meta['risk_assessment'][-1]
                    risk_level = RiskLevel[assessment_dict['risk_level']]
                    risk_levels[risk_level] += 1

                    # Track risk scores for average
                    risk_factor_avg = sum(assessment_dict['factors'].values()) / len(assessment_dict['factors'])
                    risk_scores.append(risk_factor_avg)

                    assessed_count += 1

                    # Check if outcome is known
                    if pd.notna(row['outcome']):
                        outcome_count += 1
                        if row['prediction'] == row['outcome']:
                            correct_by_risk[risk_level] += 1

            # Calculate accuracy by risk level
            accuracy_by_risk = {}
            for level in RiskLevel:
                count = risk_levels[level]
                correct = correct_by_risk[level]
                accuracy_by_risk[level.name] = correct / count if count > 0 else None

            # Strategy summary
            strategy_summary = {
                'prediction_count': len(df),
                'assessed_count': assessed_count,
                'outcome_count': outcome_count,
                'risk_distribution': {level.name: count for level, count in risk_levels.items()},
                'accuracy_by_risk': accuracy_by_risk
            }

            # Add metrics if available
            metrics = tracker.get_metrics()
            if metrics:
                strategy_summary['metrics'] = metrics.to_dict()

            report['strategies'][name] = strategy_summary

            # Update overall counts
            report['overall_risk_profile']['low_risk_count'] += risk_levels[RiskLevel.LOW]
            report['overall_risk_profile']['medium_risk_count'] += risk_levels[RiskLevel.MEDIUM]
            report['overall_risk_profile']['high_risk_count'] += risk_levels[RiskLevel.HIGH]
            report['overall_risk_profile']['extreme_risk_count'] += risk_levels[RiskLevel.EXTREME]

        # Calculate average risk score
        if risk_scores:
            report['overall_risk_profile']['avg_risk_score'] = sum(risk_scores) / len(risk_scores)

        # Save report
        self._save_risk_report(report)

        return report

    def _generate_recommendations(self, risk_level: RiskLevel, risk_factors: Dict,
                               context_data: Dict = None) -> List[str]:
        """Generate recommendations based on risk level and factors"""
        recommendations = []

        # Basic recommendations by risk level
        if risk_level == RiskLevel.LOW:
            recommendations.append("Standard monitoring recommended.")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Enhanced monitoring recommended.")
            recommendations.append("Consider reducing exposure by 15-25%.")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Close monitoring required.")
            recommendations.append("Reduce exposure by 30-50%.")
            recommendations.append("Implement hedging strategies.")
        elif risk_level == RiskLevel.EXTREME:
            recommendations.append("Immediate action required.")
            recommendations.append("Minimize exposure - reduce by 75-100%.")
            recommendations.append("Implement full hedging or position exit.")

        # Factor-specific recommendations
        if risk_factors.get('confidence', 0) > 0.6:
            recommendations.append("Low confidence prediction - seek additional confirmation.")

        if risk_factors.get('strategy_history', 0) > 0.5:
            recommendations.append("Strategy has underperformed historically - consider model refinement.")

        if risk_factors.get('market_volatility', 0) > 0.7:
            recommendations.append("High market volatility - implement volatility-based position sizing.")

        if risk_factors.get('data_quality', 0) > 0.5:
            recommendations.append("Data quality concerns - verify data sources and preprocessing.")

        # Add custom rules from config
        if risk_level.name in self.mitigation_rules:
            custom_rules = self.mitigation_rules[risk_level.name]
            if isinstance(custom_rules, list):
                recommendations.extend(custom_rules)

        return recommendations

    def _save_risk_report(self, report: Dict) -> bool:
        """Save risk report to file"""
        report_dir = self.data_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"risk_report_{timestamp}.json"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Risk report saved to {report_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save risk report: {str(e)}")
            return False

    def _load_strategies(self) -> None:
        """Load existing strategy trackers"""
        strategies_dir = self.data_dir / "strategies"
        if not strategies_dir.exists():
            return

        for file_path in strategies_dir.glob("*_history.json"):
            try:
                strategy_name = file_path.stem.replace("_history", "")
                tracker = StrategyTracker(strategy_name, {'data_dir': str(self.data_dir / 'strategies')})
                self.strategy_trackers[strategy_name] = tracker
            except Exception as e:
                logger.error(f"Failed to load strategy from {file_path}: {str(e)}")

        logger.info(f"Loaded {len(self.strategy_trackers)} strategy trackers")

class StrategyLogger:
    """Logs prediction strategy execution and performance"""

    def __init__(self, strategy_name: str, config: Dict = None):
        """
        Initialize strategy logger

        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary
        """
        self.strategy_name = strategy_name
        self.config = config or {}

        # Log storage
        self.log_dir = Path(self.config.get('log_dir', 'logs/strategies'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Strategy log file
        self.log_file = self.log_dir / f"{strategy_name}.log"

        # Set up file handler for this strategy
        self.logger = logging.getLogger(f"strategy.{strategy_name}")

        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(self.log_file)
                  for h in self.logger.handlers):
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Performance log
        self.performance_log = []

        # Risk manager integration
        self.risk_manager = None
        if self.config.get('use_risk_manager', True):
            self.risk_manager = RiskManager(self.config.get('risk_config'))

        self.logger.info(f"Initialized StrategyLogger for {strategy_name}")

    def log_strategy_start(self, params: Dict = None) -> None:
        """Log strategy execution start"""
        self.logger.info(f"Strategy execution started with params: {params}")

    def log_strategy_end(self, result: Dict = None) -> None:
        """Log strategy execution end"""
        self.logger.info(f"Strategy execution completed with result: {result}")

    def log_prediction(self, prediction: Any, confidence: float,
                      context: Dict = None) -> int:
        """
        Log a prediction and assess risk if enabled

        Args:
            prediction: The prediction value
            confidence: Confidence level (0.0 to 1.0)
            context: Additional context for the prediction

        Returns:
            Prediction ID for later outcome logging
        """
        self.logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f})")

        if context:
            self.logger.info(f"Prediction context: {context}")

        pred_id = -1

        # If risk manager is enabled, track prediction with risk assessment
        if self.risk_manager:
            pred_id, assessment = self.risk_manager.track_prediction_with_risk(
                self.strategy_name, prediction, confidence, context
            )

            # Log risk assessment
            self.logger.info(f"Risk assessment: {assessment.risk_level.name} "
                           f"(exposure: {assessment.exposure:.2f})")

            if assessment.recommendations:
                self.logger.info(f"Risk recommendations: {'; '.join(assessment.recommendations)}")

        return pred_id

    def log_outcome(self, pred_id: int, actual_outcome: Any,
                   performance_metrics: Dict = None) -> bool:
        """
        Log the actual outcome of a prediction

        Args:
            pred_id: Prediction ID returned from log_prediction
            actual_outcome: The actual outcome
            performance_metrics: Additional performance metrics

        Returns:
            Success status
        """
        self.logger.info(f"Outcome for prediction {pred_id}: {actual_outcome}")

        if performance_metrics:
            self.logger.info(f"Performance metrics: {performance_metrics}")

        # Record in performance log
        perf_entry = {
            'timestamp': datetime.now().isoformat(),
            'pred_id': pred_id,
            'outcome': actual_outcome,
            'metrics': performance_metrics
        }
        self.performance_log.append(perf_entry)

        # Save performance log periodically
        if len(self.performance_log) % 10 == 0:
            self._save_performance_log()

        # Update risk manager if available
        if self.risk_manager and pred_id >= 0:
            return self.risk_manager.update_prediction_outcome(
                self.strategy_name, pred_id, actual_outcome
            )

        return True

    def log_error(self, error_message: str, error_type: str = None,
                 stack_trace: str = None) -> None:
        """Log an error during strategy execution"""
        self.logger.error(f"Error: {error_message}")

        if error_type:
            self.logger.error(f"Error type: {error_type}")

        if stack_trace:
            self.logger.error(f"Stack trace: {stack_trace}")

    def log_warning(self, warning_message: str) -> None:
        """Log a warning during strategy execution"""
        self.logger.warning(warning_message)

    def get_strategy_logs(self, max_entries: int = 100) -> List[str]:
        """
        Get recent log entries for the strategy

        Args:
            max_entries: Maximum number of entries to return

        Returns:
            List of log entries
        """
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Return most recent entries
            return lines[-max_entries:] if len(lines) > max_entries else lines
        except Exception as e:
            self.logger.error(f"Failed to read log file: {str(e)}")
            return []

    def get_performance_metrics(self) -> Dict:
        """Get aggregated performance metrics"""
        if not self.performance_log:
            return {}

        # Get metrics from risk manager if available
        if self.risk_manager:
            metrics = self.risk_manager.get_strategy_metrics(self.strategy_name)
            if metrics:
                return metrics.to_dict()

        # Otherwise calculate basic metrics from performance log
        total = len(self.performance_log)
        correct = sum(1 for entry in self.performance_log
                     if entry.get('metrics', {}).get('correct', False))

        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0.0
        }

    def _save_performance_log(self) -> bool:
        """Save performance log to file"""
        perf_file = self.log_dir / f"{self.strategy_name}_performance.json"

        try:
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_log, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save performance log: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create risk manager
    risk_manager = RiskManager({
        'risk_thresholds': {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        },
        'risk_factor_weights': {
            'confidence': 0.4,
            'strategy_history': 0.3,
            'market_volatility': 0.2,
            'data_quality': 0.1
        },
        'mitigation_rules': {
            'HIGH': [
                "Implement stop-loss at 2% below entry price",
                "Reduce position size by 40%"
            ]
        }
    })

    # Create strategy logger
    logger = StrategyLogger("forex_trend_following", {"use_risk_manager": True})

    # Log prediction and outcome
    logger.log_strategy_start({"timeframe": "4h", "pairs": ["EUR/USD", "GBP/JPY"]})

    pred_id = logger.log_prediction("BUY", 0.75, {
        "market_volatility": 0.4,
        "signal_strength": 0.8,
        "trend_duration": "3 days"
    })

    # Later, log the outcome
    logger.log_outcome(pred_id, "BUY", {
        "correct": True,
        "profit": 120,
        "roi": 0.08
    })

    logger.log_strategy_end({"total_trades": 1, "profitable_trades": 1})
