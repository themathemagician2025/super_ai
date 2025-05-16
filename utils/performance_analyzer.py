# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Performance Analyzer for Super AI System

This module analyzes performance data collected by the performance_monitor
to identify bottlenecks and provide optimization recommendations.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from .performance_monitor import get_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Development/logs/performance_analysis.log"),
        logging.StreamHandler()
    ]
)

class PerformanceAnalyzer:
    """Analyzes performance data to identify bottlenecks and recommend optimizations."""

    def __init__(self, metrics_dir="Development/logs/metrics"):
        self.metrics_dir = metrics_dir
        self.logger = logging.getLogger("PerformanceAnalyzer")
        self.bottlenecks = []
        self.recommendations = []
        self.performance_trends = {}

        # Create analysis directory if it doesn't exist
        os.makedirs("Development/logs/analysis", exist_ok=True)

    def load_metrics(self, days=1):
        """Load performance metrics from the last N days."""
        metrics_data = []
        cutoff_time = datetime.now() - timedelta(days=days)

        try:
            # Check if directory exists
            if not os.path.exists(self.metrics_dir):
                self.logger.warning(f"Metrics directory not found: {self.metrics_dir}")
                return pd.DataFrame()

            # Load all metric files from the specified directory
            for filename in os.listdir(self.metrics_dir):
                if not filename.endswith('.json'):
                    continue

                file_path = os.path.join(self.metrics_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))

                # Skip files older than the cutoff time
                if file_time < cutoff_time:
                    continue

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data["file_created"] = file_time.isoformat()
                        metrics_data.append(data)
                except Exception as e:
                    self.logger.error(f"Error loading metrics file {filename}: {str(e)}")

            # Convert to DataFrame if data exists
            if metrics_data:
                # Flatten the nested structure for easier analysis
                flat_data = []
                for entry in metrics_data:
                    flat_entry = {
                        "timestamp": entry.get("timestamp", ""),
                        "file_created": entry.get("file_created", "")
                    }

                    # Flatten metrics
                    for metric_name, metric_data in entry.get("metrics", {}).items():
                        if isinstance(metric_data, dict):
                            for key, value in metric_data.items():
                                flat_entry[f"{metric_name}_{key}"] = value
                        else:
                            flat_entry[metric_name] = metric_data

                    flat_data.append(flat_entry)

                # Convert to DataFrame
                return pd.DataFrame(flat_data)
            else:
                self.logger.warning("No metrics data found for the specified time period")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error loading metrics: {str(e)}")
            return pd.DataFrame()

    def identify_bottlenecks(self, df=None):
        """Identify performance bottlenecks from metrics data."""
        self.bottlenecks = []

        if df is None or df.empty:
            df = self.load_metrics()

        if df.empty:
            self.logger.warning("No data available for bottleneck analysis")
            return self.bottlenecks

        try:
            # Check response time bottlenecks
            if 'response_times_mean' in df.columns and 'response_times_p95' in df.columns:
                avg_response = df['response_times_mean'].mean()
                p95_response = df['response_times_p95'].mean()

                if avg_response > 2.0:  # More than 2 seconds average response time
                    self.bottlenecks.append({
                        "component": "Overall Response Time",
                        "severity": "High" if avg_response > 5.0 else "Medium",
                        "metric": f"Avg: {avg_response:.2f}s, P95: {p95_response:.2f}s",
                        "description": f"Average response time of {avg_response:.2f}s exceeds target of 2.0s"
                    })

            # Check component latencies
            latency_columns = [col for col in df.columns if 'component_latencies' in col and 'mean' in col]
            for col in latency_columns:
                component = col.split('component_latencies_')[1].split('_mean')[0]
                avg_latency = df[col].mean()

                # Different thresholds for different components
                threshold = 1.0  # Default threshold
                if 'memory' in component.lower():
                    threshold = 0.5
                elif 'model' in component.lower():
                    threshold = 1.5

                if avg_latency > threshold:
                    severity = "High" if avg_latency > threshold * 2 else "Medium"
                    self.bottlenecks.append({
                        "component": component,
                        "severity": severity,
                        "metric": f"Avg: {avg_latency:.2f}s",
                        "description": f"Average latency of {avg_latency:.2f}s exceeds target of {threshold:.1f}s"
                    })

            # Check error rates
            if 'total_error_rate_rate' in df.columns:
                avg_error_rate = df['total_error_rate_rate'].mean() * 100  # Convert to percentage
                if avg_error_rate > 1.0:  # More than 1% error rate
                    self.bottlenecks.append({
                        "component": "Error Handling",
                        "severity": "High" if avg_error_rate > 5.0 else "Medium",
                        "metric": f"{avg_error_rate:.2f}%",
                        "description": f"Error rate of {avg_error_rate:.2f}% exceeds target of 1.0%"
                    })

            # Check resource usage
            if 'memory_usage_mean' in df.columns:
                avg_memory = df['memory_usage_mean'].mean()
                if avg_memory > 80:  # More than 80% memory usage
                    self.bottlenecks.append({
                        "component": "Memory Usage",
                        "severity": "High" if avg_memory > 90 else "Medium",
                        "metric": f"{avg_memory:.2f}%",
                        "description": f"Average memory usage of {avg_memory:.2f}% is high"
                    })

            if 'cpu_usage_mean' in df.columns:
                avg_cpu = df['cpu_usage_mean'].mean()
                if avg_cpu > 70:  # More than 70% CPU usage
                    self.bottlenecks.append({
                        "component": "CPU Usage",
                        "severity": "High" if avg_cpu > 85 else "Medium",
                        "metric": f"{avg_cpu:.2f}%",
                        "description": f"Average CPU usage of {avg_cpu:.2f}% is high"
                    })

            # Sort bottlenecks by severity (High first)
            self.bottlenecks.sort(key=lambda x: 0 if x["severity"] == "High" else 1)

            return self.bottlenecks

        except Exception as e:
            self.logger.error(f"Error identifying bottlenecks: {str(e)}")
            return self.bottlenecks

    def generate_recommendations(self):
        """Generate optimization recommendations based on identified bottlenecks."""
        self.recommendations = []

        if not self.bottlenecks:
            self.identify_bottlenecks()

        if not self.bottlenecks:
            return self.recommendations

        try:
            # Define recommendation templates for different bottleneck types
            recommendation_templates = {
                "Overall Response Time": [
                    "Implement response caching for frequent queries",
                    "Optimize the model selection algorithm to reduce overhead",
                    "Consider using a lighter model for initial responses with fallback to more powerful models",
                    "Parallelize independent components of the response generation pipeline"
                ],
                "Memory Usage": [
                    "Implement more aggressive cache expiration policies",
                    "Optimize memory usage in vector embeddings by using dimensionality reduction",
                    "Consider switching to memory-mapped vector storage",
                    "Implement memory sharding across multiple instances"
                ],
                "CPU Usage": [
                    "Profile the application to identify CPU-intensive operations",
                    "Batch processing for multiple similar requests",
                    "Implement request throttling during peak loads",
                    "Consider horizontally scaling the application"
                ],
                "Error Handling": [
                    "Implement more robust retry mechanisms with exponential backoff",
                    "Add circuit breakers for unstable external services",
                    "Enhance fallback mechanisms when primary processing fails",
                    "Implement predictive error detection based on input patterns"
                ]
            }

            # Generate specific recommendations for each bottleneck
            for bottleneck in self.bottlenecks:
                component = bottleneck["component"]

                # Get template recommendations for this component type
                templates = []
                for key, recs in recommendation_templates.items():
                    if key in component or key.lower() in component.lower():
                        templates.extend(recs)

                # If no direct match, use generic recommendations based on component name
                if not templates:
                    if "memory" in component.lower() or "retrieval" in component.lower():
                        templates = recommendation_templates["Memory Usage"]
                    elif "model" in component.lower() or "selection" in component.lower():
                        templates = recommendation_templates["Overall Response Time"]
                    elif "error" in component.lower():
                        templates = recommendation_templates["Error Handling"]
                    else:
                        # Add specific recommendations based on component patterns
                        if "feedback" in component.lower():
                            templates = [
                                "Process feedback asynchronously",
                                "Batch feedback processing during quiet periods",
                                "Implement priority-based feedback processing"
                            ]
                        else:
                            templates = [
                                f"Profile {component} to identify specific slow operations",
                                f"Consider caching {component} results when appropriate",
                                f"Optimize the data structures used in {component}"
                            ]

                # Select recommendations based on severity
                num_recommendations = 3 if bottleneck["severity"] == "High" else 2
                selected_recommendations = templates[:num_recommendations]

                for rec in selected_recommendations:
                    self.recommendations.append({
                        "component": component,
                        "severity": bottleneck["severity"],
                        "recommendation": rec,
                        "expected_impact": "High" if bottleneck["severity"] == "High" else "Medium"
                    })

            return self.recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return self.recommendations

    def analyze_performance_trends(self, days=7):
        """Analyze performance trends over time."""
        self.performance_trends = {}

        try:
            df = self.load_metrics(days=days)
            if df.empty:
                return self.performance_trends

            # Ensure timestamp is in datetime format
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            # Group data by day for trend analysis
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date

                # Define metrics to track
                trend_metrics = {
                    'response_times_mean': 'Response Time (s)',
                    'memory_retrieval_times_mean': 'Memory Retrieval Time (s)',
                    'model_selection_times_mean': 'Model Selection Time (s)',
                    'total_error_rate_rate': 'Error Rate (%)',
                    'memory_usage_mean': 'Memory Usage (%)',
                    'cpu_usage_mean': 'CPU Usage (%)'
                }

                # Calculate daily averages for each metric
                daily_stats = df.groupby('date').mean()

                # Calculate trends
                for metric_col, metric_name in trend_metrics.items():
                    if metric_col in daily_stats.columns:
                        values = daily_stats[metric_col].values

                        if len(values) > 1:
                            # Calculate trend (% change from first to last day)
                            first_value = values[0]
                            last_value = values[-1]

                            if first_value > 0:
                                percent_change = ((last_value - first_value) / first_value) * 100
                            else:
                                percent_change = 0.0 if last_value == 0 else 100.0

                            # Determine trend direction
                            if metric_col in ['response_times_mean', 'memory_retrieval_times_mean',
                                            'model_selection_times_mean', 'total_error_rate_rate']:
                                # For these metrics, decreasing is good
                                trend_direction = "improving" if percent_change < -5 else "worsening" if percent_change > 5 else "stable"
                            else:
                                # For resource usage, it depends on the values
                                if metric_col in ['memory_usage_mean', 'cpu_usage_mean']:
                                    if last_value > 80:
                                        trend_direction = "worsening" if percent_change > 5 else "stable"
                                    else:
                                        trend_direction = "stable"
                                else:
                                    trend_direction = "stable"

                            self.performance_trends[metric_name] = {
                                "first_value": float(first_value),
                                "last_value": float(last_value),
                                "percent_change": float(percent_change),
                                "trend_direction": trend_direction,
                                "values": [float(v) for v in values],
                                "dates": [str(d) for d in daily_stats.index]
                            }

            return self.performance_trends

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {str(e)}")
            return self.performance_trends

    def generate_visualization(self, output_dir="Development/logs/analysis"):
        """Generate visualizations of performance metrics."""
        try:
            df = self.load_metrics(days=7)
            if df.empty:
                return False

            # Ensure timestamp is in datetime format
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            else:
                return False

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Plot response times
            if 'response_times_mean' in df.columns and 'response_times_p95' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df['timestamp'], df['response_times_mean'], label='Mean Response Time')
                plt.plot(df['timestamp'], df['response_times_p95'], label='95th Percentile')
                plt.title('Response Time Trends')
                plt.xlabel('Time')
                plt.ylabel('Response Time (seconds)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'response_times.png'))
                plt.close()

            # Plot component latencies
            latency_columns = [col for col in df.columns if 'component_latencies' in col and 'mean' in col]
            if latency_columns:
                plt.figure(figsize=(12, 7))
                for col in latency_columns:
                    component = col.split('component_latencies_')[1].split('_mean')[0]
                    plt.plot(df['timestamp'], df[col], label=f'{component}')
                plt.title('Component Latency Trends')
                plt.xlabel('Time')
                plt.ylabel('Latency (seconds)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'component_latencies.png'))
                plt.close()

            # Plot resource usage
            if 'memory_usage_mean' in df.columns and 'cpu_usage_mean' in df.columns:
                fig, ax1 = plt.subplots(figsize=(10, 6))

                color = 'tab:blue'
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Memory Usage (%)', color=color)
                ax1.plot(df['timestamp'], df['memory_usage_mean'], color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('CPU Usage (%)', color=color)
                ax2.plot(df['timestamp'], df['cpu_usage_mean'], color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                plt.title('Resource Usage Trends')
                fig.tight_layout()
                plt.savefig(os.path.join(output_dir, 'resource_usage.png'))
                plt.close()

            # Plot error rates if available
            if 'total_error_rate_rate' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df['timestamp'], df['total_error_rate_rate'] * 100)  # Convert to percentage
                plt.title('Error Rate Trends')
                plt.xlabel('Time')
                plt.ylabel('Error Rate (%)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'error_rates.png'))
                plt.close()

            return True

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return False

    def generate_report(self, output_file=None):
        """Generate a comprehensive analysis report."""
        try:
            # Ensure we have fresh bottlenecks and recommendations
            self.identify_bottlenecks()
            self.generate_recommendations()
            self.analyze_performance_trends()

            # Create report content
            report = ["# Super AI System Performance Analysis Report",
                     f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

            # Add bottlenecks section
            report.append("## Performance Bottlenecks")
            if self.bottlenecks:
                for bottleneck in self.bottlenecks:
                    report.append(f"### {bottleneck['component']} (Severity: {bottleneck['severity']})")
                    report.append(f"- **Metric**: {bottleneck['metric']}")
                    report.append(f"- **Description**: {bottleneck['description']}")
                    report.append("")
            else:
                report.append("No significant bottlenecks detected.\n")

            # Add recommendations section
            report.append("## Optimization Recommendations")
            if self.recommendations:
                for i, rec in enumerate(self.recommendations):
                    report.append(f"### Recommendation {i+1}")
                    report.append(f"- **Component**: {rec['component']}")
                    report.append(f"- **Recommendation**: {rec['recommendation']}")
                    report.append(f"- **Expected Impact**: {rec['expected_impact']}")
                    report.append("")
            else:
                report.append("No optimization recommendations at this time.\n")

            # Add performance trends section
            report.append("## Performance Trends")
            if self.performance_trends:
                for metric, trend in self.performance_trends.items():
                    direction = trend['trend_direction'].capitalize()
                    change = f"{trend['percent_change']:.2f}%"
                    change_direction = "increase" if trend['percent_change'] > 0 else "decrease"

                    report.append(f"### {metric}")
                    report.append(f"- **Trend**: {direction} ({change} {change_direction})")
                    report.append(f"- **Current Value**: {trend['last_value']:.2f}")
                    report.append(f"- **7-Day Value**: {trend['first_value']:.2f}")
                    report.append("")
            else:
                report.append("Insufficient data for trend analysis.\n")

            # Add visualization references
            if self.generate_visualization():
                report.append("## Visualizations")
                report.append("Performance visualizations have been generated in the 'Development/logs/analysis' directory:")
                report.append("- response_times.png - Response time trends")
                report.append("- component_latencies.png - Component latency trends")
                report.append("- resource_usage.png - Memory and CPU usage trends")
                report.append("- error_rates.png - Error rate trends")
                report.append("")

            # Add system architecture recommendations
            report.append("## System Architecture Recommendations")

            # Check if we have bottlenecks to make architecture recommendations
            if self.bottlenecks:
                # Check for high CPU usage
                cpu_bottleneck = next((b for b in self.bottlenecks if "CPU Usage" in b["component"]), None)
                memory_bottleneck = next((b for b in self.bottlenecks if "Memory Usage" in b["component"]), None)
                response_bottleneck = next((b for b in self.bottlenecks if "Response Time" in b["component"]), None)

                if cpu_bottleneck and cpu_bottleneck["severity"] == "High":
                    report.append("### 1. Consider Horizontal Scaling")
                    report.append("- Deploy multiple instances of the system behind a load balancer")
                    report.append("- Implement stateless design to facilitate scaling")
                    report.append("- Consider containerization with Kubernetes for automated scaling")
                    report.append("")

                if memory_bottleneck and memory_bottleneck["severity"] == "High":
                    report.append("### 2. Memory Optimization Strategy")
                    report.append("- Implement a distributed caching system (Redis/Memcached)")
                    report.append("- Consider vertical scaling for memory-intensive components")
                    report.append("- Implement memory sharding for vector embeddings")
                    report.append("")

                if response_bottleneck:
                    report.append("### 3. Response Time Optimization")
                    report.append("- Implement a tiered model architecture with cascading complexity")
                    report.append("- Consider using specialized models for different query types")
                    report.append("- Implement precomputation of common query responses")
                    report.append("- Use asynchronous processing for non-critical components")
                    report.append("")

                # General recommendations
                report.append("### 4. General Architecture Improvements")
                report.append("- Implement a microservices architecture to isolate bottlenecks")
                report.append("- Use a message queue for asynchronous processing of non-critical tasks")
                report.append("- Implement circuit breakers for unstable components")
                report.append("- Consider a dedicated monitoring and alerting system")
                report.append("")
            else:
                report.append("No architecture changes recommended at this time - current architecture appears adequate.\n")

            # Combine into final report
            report_text = "\n".join(report)

            # Save report if output file is provided
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(report_text)

            return report_text

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return f"Error generating performance analysis report: {str(e)}"

# Utility function to get a performance analyzer instance
def get_analyzer():
    """Get a PerformanceAnalyzer instance."""
    return PerformanceAnalyzer()

# Main function to generate a performance analysis report
def analyze_performance(output_file="Development/logs/analysis/performance_analysis.md"):
    """Analyze performance and generate a report."""
    analyzer = get_analyzer()
    return analyzer.generate_report(output_file)
