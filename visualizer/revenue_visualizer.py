# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Revenue Visualization Module

This module provides tools for visualizing:
1. Prediction performance metrics
2. Revenue and profitability trends
3. ROI and risk-adjusted returns
4. Strategy comparison dashboards
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os

logger = logging.getLogger(__name__)

class RevenueVisualizer:
    """Visualizes financial performance and prediction metrics"""

    def __init__(self, config: Dict = None):
        """
        Initialize revenue visualizer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Output directories
        self.output_dir = Path(self.config.get('output_dir', 'visualizations/revenue'))
        self.data_dir = Path(self.config.get('data_dir', 'data/performance'))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use(self.config.get('plot_style', 'seaborn-v0_8-darkgrid'))
        self.figsize = self.config.get('figsize', (12, 8))
        self.dpi = self.config.get('dpi', 100)

        # Color schemes
        self.colors = self.config.get('colors', {
            'profit': '#2ecc71',
            'loss': '#e74c3c',
            'neutral': '#3498db',
            'prediction': '#f39c12',
            'actual': '#9b59b6'
        })

        logger.info("Initialized RevenueVisualizer")

    def load_performance_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load performance data from file

        Args:
            file_path: Path to the performance data file

        Returns:
            DataFrame with performance data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Performance data file not found: {file_path}")
            return pd.DataFrame()

        try:
            ext = file_path.suffix.lower()

            if ext == '.csv':
                return pd.read_csv(file_path, parse_dates=['timestamp'])
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert to DataFrame
                df = pd.DataFrame(data)

                # Parse timestamps
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                return df
            else:
                logger.error(f"Unsupported file format: {ext}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to load performance data: {str(e)}")
            return pd.DataFrame()

    def plot_revenue_over_time(self, data: pd.DataFrame,
                              output_file: Optional[str] = None) -> str:
        """
        Plot revenue/profit over time

        Args:
            data: DataFrame with performance data
            output_file: Output file path (or auto-generate if None)

        Returns:
            Path to the saved visualization
        """
        if data.empty:
            logger.error("Cannot plot empty dataset")
            return ""

        # Check required columns
        required_cols = ['timestamp', 'profit']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Data missing required columns: {required_cols}")
            return ""

        # Prepare output file name
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"revenue_over_time_{timestamp}.png")

        # Sort by timestamp
        data = data.sort_values('timestamp')

        # Calculate cumulative profit
        if 'cumulative_profit' not in data.columns:
            data['cumulative_profit'] = data['profit'].cumsum()

        # Create plot
        plt.figure(figsize=self.figsize, dpi=self.dpi)

        # Plot cumulative profit
        plt.plot(data['timestamp'], data['cumulative_profit'],
                 color=self.colors['profit'], linewidth=2)

        # Add individual profit/loss points
        profit_points = data[data['profit'] > 0]
        loss_points = data[data['profit'] < 0]

        plt.scatter(profit_points['timestamp'], profit_points['cumulative_profit'],
                   color=self.colors['profit'], alpha=0.6)
        plt.scatter(loss_points['timestamp'], loss_points['cumulative_profit'],
                   color=self.colors['loss'], alpha=0.6)

        # Add trend line
        if len(data) > 1:
            z = np.polyfit(range(len(data)), data['cumulative_profit'], 1)
            p = np.poly1d(z)
            plt.plot(data['timestamp'], p(range(len(data))),
                    color=self.colors['neutral'], linestyle='--',
                    label=f"Trend: {z[0]:.2f} per prediction")

        # Add labels and title
        plt.title("Cumulative Profit Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Profit")
        plt.grid(True, alpha=0.3)

        # Add stats
        total_profit = data['profit'].sum()
        win_rate = (data['profit'] > 0).mean() * 100

        plt.figtext(0.15, 0.02, f"Total Profit: {total_profit:.2f}", fontsize=12)
        plt.figtext(0.5, 0.02, f"Win Rate: {win_rate:.1f}%", fontsize=12)

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Revenue over time visualization saved to {output_file}")
        return output_file

    def plot_prediction_accuracy(self, data: pd.DataFrame,
                               output_file: Optional[str] = None) -> str:
        """
        Plot prediction accuracy metrics

        Args:
            data: DataFrame with prediction results
            output_file: Output file path (or auto-generate if None)

        Returns:
            Path to the saved visualization
        """
        if data.empty:
            logger.error("Cannot plot empty dataset")
            return ""

        # Check required columns
        required_cols = ['timestamp', 'correct']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Data missing required columns: {required_cols}")
            return ""

        # Prepare output file name
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"prediction_accuracy_{timestamp}.png")

        # Sort by timestamp
        data = data.sort_values('timestamp')

        # Calculate rolling accuracy
        window_size = min(10, len(data))
        data['rolling_accuracy'] = data['correct'].rolling(window=window_size).mean()

        # Create plot
        plt.figure(figsize=self.figsize, dpi=self.dpi)

        # Plot rolling accuracy
        plt.plot(data['timestamp'], data['rolling_accuracy'],
                 color=self.colors['neutral'], linewidth=2,
                 label=f"{window_size}-prediction Rolling Accuracy")

        # Plot overall accuracy line
        overall_accuracy = data['correct'].mean()
        plt.axhline(y=overall_accuracy, color=self.colors['prediction'],
                   linestyle='--', label=f"Overall Accuracy: {overall_accuracy:.2f}")

        # Add individual prediction points
        correct_points = data[data['correct']]
        incorrect_points = data[~data['correct']]

        plt.scatter(correct_points['timestamp'], [1] * len(correct_points),
                   color=self.colors['profit'], alpha=0.6, label="Correct")
        plt.scatter(incorrect_points['timestamp'], [0] * len(incorrect_points),
                   color=self.colors['loss'], alpha=0.6, label="Incorrect")

        # Add labels and title
        plt.title("Prediction Accuracy Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Accuracy")
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)

        # Add stats
        total_predictions = len(data)
        correct_predictions = data['correct'].sum()
        accuracy = correct_predictions / total_predictions

        plt.figtext(0.15, 0.02, f"Total Predictions: {total_predictions}", fontsize=12)
        plt.figtext(0.5, 0.02, f"Accuracy: {accuracy:.1%}", fontsize=12)

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Prediction accuracy visualization saved to {output_file}")
        return output_file

    def plot_roi_distribution(self, data: pd.DataFrame,
                             output_file: Optional[str] = None) -> str:
        """
        Plot distribution of ROI values

        Args:
            data: DataFrame with ROI data
            output_file: Output file path (or auto-generate if None)

        Returns:
            Path to the saved visualization
        """
        if data.empty:
            logger.error("Cannot plot empty dataset")
            return ""

        # Check required columns
        if 'roi' not in data.columns:
            logger.error("Data missing required 'roi' column")
            return ""

        # Prepare output file name
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"roi_distribution_{timestamp}.png")

        # Create plot
        plt.figure(figsize=self.figsize, dpi=self.dpi)

        # Plot ROI distribution
        sns.histplot(data['roi'], bins=20, kde=True, color=self.colors['neutral'])

        # Add mean and median lines
        roi_mean = data['roi'].mean()
        roi_median = data['roi'].median()

        plt.axvline(x=roi_mean, color=self.colors['prediction'],
                   linestyle='--', label=f"Mean ROI: {roi_mean:.2%}")
        plt.axvline(x=roi_median, color=self.colors['actual'],
                   linestyle=':', label=f"Median ROI: {roi_median:.2%}")

        # Add zero line
        plt.axvline(x=0, color='black', alpha=0.5)

        # Add labels and title
        plt.title("ROI Distribution", fontsize=16)
        plt.xlabel("Return on Investment (ROI)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add stats
        positive_roi = (data['roi'] > 0).mean() * 100
        plt.figtext(0.15, 0.02, f"Positive ROI: {positive_roi:.1f}%", fontsize=12)

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        logger.info(f"ROI distribution visualization saved to {output_file}")
        return output_file

    def plot_strategy_comparison(self, strategies_data: Dict[str, pd.DataFrame],
                                output_file: Optional[str] = None) -> str:
        """
        Plot comparison of multiple strategies

        Args:
            strategies_data: Dictionary mapping strategy names to their data
            output_file: Output file path (or auto-generate if None)

        Returns:
            Path to the saved visualization
        """
        if not strategies_data:
            logger.error("No strategy data provided")
            return ""

        # Prepare output file name
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"strategy_comparison_{timestamp}.png")

        # Create plot
        plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.5), dpi=self.dpi)

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), dpi=self.dpi)

        # Plot cumulative profit for each strategy
        for strategy_name, data in strategies_data.items():
            if data.empty or 'profit' not in data.columns or 'timestamp' not in data.columns:
                logger.warning(f"Skipping invalid data for strategy: {strategy_name}")
                continue

            # Sort and calculate cumulative profit
            data = data.sort_values('timestamp')
            data['cumulative_profit'] = data['profit'].cumsum()

            # Plot on the first axis
            axes[0].plot(data['timestamp'], data['cumulative_profit'], label=strategy_name)

        axes[0].set_title("Cumulative Profit Comparison", fontsize=14)
        axes[0].set_xlabel("")  # Hide x-label for top plots
        axes[0].set_ylabel("Profit")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot accuracy comparison (bar chart)
        strategy_names = []
        accuracies = []
        roi_values = []
        sharpe_values = []

        for strategy_name, data in strategies_data.items():
            if data.empty:
                continue

            # Calculate accuracy
            if 'correct' in data.columns:
                accuracy = data['correct'].mean()
                accuracies.append(accuracy)
                strategy_names.append(strategy_name)

                # Calculate ROI if available
                if 'roi' in data.columns:
                    roi = data['roi'].mean()
                    roi_values.append(roi)
                else:
                    roi_values.append(0)

                # Calculate Sharpe ratio if available
                if 'roi' in data.columns:
                    returns = data['roi'].values
                    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                    sharpe_values.append(sharpe)
                else:
                    sharpe_values.append(0)

        # Plot accuracy bars
        if strategy_names:
            axes[1].bar(strategy_names, accuracies, color=self.colors['neutral'])
            axes[1].set_title("Prediction Accuracy by Strategy", fontsize=14)
            axes[1].set_xlabel("")  # Hide x-label for middle plot
            axes[1].set_ylabel("Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)

            # Add accuracy values on top of bars
            for i, v in enumerate(accuracies):
                axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center')

            # Plot ROI and Sharpe ratio
            x = np.arange(len(strategy_names))
            width = 0.35

            axes[2].bar(x - width/2, roi_values, width, color=self.colors['profit'], label='ROI')
            axes[2].bar(x + width/2, sharpe_values, width, color=self.colors['prediction'], label='Sharpe Ratio')

            axes[2].set_title("ROI and Risk-Adjusted Returns by Strategy", fontsize=14)
            axes[2].set_xlabel("Strategy")
            axes[2].set_ylabel("Value")
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(strategy_names)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Strategy comparison visualization saved to {output_file}")
        return output_file

    def create_performance_dashboard(self, data: pd.DataFrame,
                                    strategy_name: str = "Strategy Performance",
                                    output_file: Optional[str] = None) -> str:
        """
        Create a comprehensive performance dashboard

        Args:
            data: DataFrame with prediction and performance data
            strategy_name: Name of the strategy
            output_file: Output file path (or auto-generate if None)

        Returns:
            Path to the saved dashboard
        """
        if data.empty:
            logger.error("Cannot create dashboard with empty dataset")
            return ""

        # Prepare output file name
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"dashboard_{strategy_name.replace(' ', '_')}_{timestamp}.png")

        # Create plot with multiple panels
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5), dpi=self.dpi)
        plt.suptitle(f"{strategy_name} Dashboard", fontsize=20)

        # Panel 1: Cumulative profit over time
        if 'profit' in data.columns and 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            data['cumulative_profit'] = data['profit'].cumsum()

            axes[0, 0].plot(data['timestamp'], data['cumulative_profit'],
                          color=self.colors['profit'], linewidth=2)

            # Add trend line
            if len(data) > 1:
                z = np.polyfit(range(len(data)), data['cumulative_profit'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(data['timestamp'], p(range(len(data))),
                              color='black', linestyle='--')

            axes[0, 0].set_title("Cumulative Profit", fontsize=14)
            axes[0, 0].set_xlabel("")  # Hide x-label
            axes[0, 0].set_ylabel("Profit")
            axes[0, 0].grid(True, alpha=0.3)

            # Add total profit text
            total_profit = data['profit'].sum()
            axes[0, 0].text(0.05, 0.95, f"Total Profit: {total_profit:.2f}",
                          transform=axes[0, 0].transAxes, fontsize=12,
                          verticalalignment='top')

        # Panel 2: Monthly returns
        if 'profit' in data.columns and 'timestamp' in data.columns:
            # Calculate monthly returns
            data['month'] = data['timestamp'].dt.to_period('M')
            monthly_returns = data.groupby('month')['profit'].sum()

            # Convert period index to datetime for plotting
            months = [pd.to_datetime(str(m)) for m in monthly_returns.index]

            # Color bars based on positive/negative returns
            colors = [self.colors['profit'] if r > 0 else self.colors['loss']
                     for r in monthly_returns.values]

            axes[0, 1].bar(months, monthly_returns.values, color=colors)
            axes[0, 1].set_title("Monthly Returns", fontsize=14)
            axes[0, 1].set_xlabel("")  # Hide x-label
            axes[0, 1].set_ylabel("Profit")
            axes[0, 1].grid(True, alpha=0.3)

            # Format x-axis as dates
            import matplotlib.dates as mdates
            axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

        # Panel 3: Prediction accuracy over time
        if 'correct' in data.columns and 'timestamp' in data.columns:
            # Calculate rolling accuracy
            window_size = min(10, len(data))
            data['rolling_accuracy'] = data['correct'].rolling(window=window_size).mean()

            axes[1, 0].plot(data['timestamp'], data['rolling_accuracy'],
                          color=self.colors['neutral'], linewidth=2)

            # Add overall accuracy line
            overall_accuracy = data['correct'].mean()
            axes[1, 0].axhline(y=overall_accuracy, color=self.colors['prediction'],
                             linestyle='--')

            axes[1, 0].set_title("Prediction Accuracy", fontsize=14)
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)

            # Add accuracy text
            axes[1, 0].text(0.05, 0.95, f"Overall Accuracy: {overall_accuracy:.2f}",
                          transform=axes[1, 0].transAxes, fontsize=12,
                          verticalalignment='top')

        # Panel 4: ROI distribution
        if 'roi' in data.columns:
            sns.histplot(data['roi'], bins=20, kde=True, ax=axes[1, 1],
                        color=self.colors['neutral'])

            # Add mean and median lines
            roi_mean = data['roi'].mean()
            roi_median = data['roi'].median()

            axes[1, 1].axvline(x=roi_mean, color=self.colors['prediction'],
                             linestyle='--')
            axes[1, 1].axvline(x=roi_median, color=self.colors['actual'],
                             linestyle=':')

            # Add zero line
            axes[1, 1].axvline(x=0, color='black', alpha=0.5)

            axes[1, 1].set_title("ROI Distribution", fontsize=14)
            axes[1, 1].set_xlabel("ROI")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True, alpha=0.3)

            # Add ROI stats
            positive_roi = (data['roi'] > 0).mean() * 100
            axes[1, 1].text(0.05, 0.95, f"Mean ROI: {roi_mean:.2%}\nPositive ROI: {positive_roi:.1f}%",
                          transform=axes[1, 1].transAxes, fontsize=12,
                          verticalalignment='top')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(output_file)
        plt.close()

        logger.info(f"Performance dashboard saved to {output_file}")
        return output_file

    def export_metrics_report(self, data: pd.DataFrame,
                             strategy_name: str = "Strategy Report") -> str:
        """
        Export a comprehensive metrics report as CSV

        Args:
            data: DataFrame with prediction and performance data
            strategy_name: Name of the strategy

        Returns:
            Path to the saved report
        """
        if data.empty:
            logger.error("Cannot create report with empty dataset")
            return ""

        # Prepare output file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = str(self.data_dir / f"report_{strategy_name.replace(' ', '_')}_{timestamp}.csv")

        # Calculate metrics
        metrics = {}

        # Time-based metrics
        if 'timestamp' in data.columns:
            metrics['start_date'] = data['timestamp'].min().strftime('%Y-%m-%d')
            metrics['end_date'] = data['timestamp'].max().strftime('%Y-%m-%d')
            metrics['days_duration'] = (data['timestamp'].max() - data['timestamp'].min()).days

        # Prediction metrics
        if 'correct' in data.columns:
            metrics['total_predictions'] = len(data)
            metrics['correct_predictions'] = data['correct'].sum()
            metrics['accuracy'] = metrics['correct_predictions'] / metrics['total_predictions']

            # Calculate by month if timestamps available
            if 'timestamp' in data.columns:
                data['month'] = data['timestamp'].dt.to_period('M')
                monthly_accuracy = data.groupby('month')['correct'].mean()
                metrics['best_month'] = monthly_accuracy.idxmax().strftime('%Y-%m')
                metrics['best_month_accuracy'] = monthly_accuracy.max()
                metrics['worst_month'] = monthly_accuracy.idxmin().strftime('%Y-%m')
                metrics['worst_month_accuracy'] = monthly_accuracy.min()

        # Financial metrics
        if 'profit' in data.columns:
            metrics['total_profit'] = data['profit'].sum()
            metrics['avg_profit_per_prediction'] = data['profit'].mean()
            metrics['profit_stddev'] = data['profit'].std()
            metrics['max_profit'] = data['profit'].max()
            metrics['max_loss'] = data['profit'].min()

            # Win rate and profit factor
            profitable_trades = data[data['profit'] > 0]
            losing_trades = data[data['profit'] < 0]

            metrics['win_rate'] = len(profitable_trades) / len(data) if len(data) > 0 else 0

            total_profit = profitable_trades['profit'].sum() if not profitable_trades.empty else 0
            total_loss = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0

            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

            # Drawdown analysis
            if len(data) > 0:
                data = data.sort_values('timestamp')
                data['cumulative_profit'] = data['profit'].cumsum()
                cumulative_max = data['cumulative_profit'].cummax()
                drawdown = (cumulative_max - data['cumulative_profit']) / (cumulative_max + 1e-10)

                metrics['max_drawdown'] = drawdown.max()
                metrics['avg_drawdown'] = drawdown.mean()

        # ROI metrics
        if 'roi' in data.columns:
            metrics['avg_roi'] = data['roi'].mean()
            metrics['median_roi'] = data['roi'].median()
            metrics['roi_stddev'] = data['roi'].std()
            metrics['positive_roi_pct'] = (data['roi'] > 0).mean() * 100

            # Sharpe and Sortino ratios
            # Assuming risk-free rate of 0 for simplicity
            returns = data['roi'].values
            if returns.std() > 0:
                metrics['sharpe_ratio'] = returns.mean() / returns.std()

                # Sortino ratio uses downside deviation
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0 and negative_returns.std() > 0:
                    metrics['sortino_ratio'] = returns.mean() / negative_returns.std()
                else:
                    metrics['sortino_ratio'] = float('inf')
            else:
                metrics['sharpe_ratio'] = float('inf')
                metrics['sortino_ratio'] = float('inf')

        # Create metrics DataFrame and save
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(report_file, index=False)

        logger.info(f"Metrics report saved to {report_file}")
        return report_file

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create visualizer
    visualizer = RevenueVisualizer()

    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 100

    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    profits = np.random.normal(100, 500, n_samples)
    correct = np.random.binomial(1, 0.7, n_samples)
    roi = profits / 1000  # Simulated ROI values

    data = pd.DataFrame({
        'timestamp': dates,
        'profit': profits,
        'correct': correct,
        'roi': roi
    })

    # Create visualizations
    visualizer.plot_revenue_over_time(data)
    visualizer.plot_prediction_accuracy(data)
    visualizer.plot_roi_distribution(data)
    visualizer.create_performance_dashboard(data, "Demo Strategy")
    visualizer.export_metrics_report(data, "Demo Strategy")
