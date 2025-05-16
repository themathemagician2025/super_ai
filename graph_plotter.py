# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Graph Plotter Module

Provides data visualization capabilities using Matplotlib.
"""

import os
import logging
import json
import re
import base64
import io
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Configure output directory
OUTPUT_DIR = "./plot_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GraphPlotter:
    """Tool for plotting data visualizations using Matplotlib."""

    def __init__(self, output_dir: str = OUTPUT_DIR, default_dpi: int = 100):
        """
        Initialize the graph plotter.

        Args:
            output_dir: Directory to save plots
            default_dpi: Default resolution for plots
        """
        self.output_dir = output_dir
        self.default_dpi = default_dpi
        self.supported_plot_types = {
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'scatter': self._create_scatter_plot,
            'pie': self._create_pie_chart,
            'histogram': self._create_histogram,
            'boxplot': self._create_box_plot
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Graph plotter initialized with output directory: {output_dir}")

    def create_plot(self,
                   data: Any,
                   plot_type: str = 'line',
                   title: str = 'Plot',
                   xlabel: str = 'X',
                   ylabel: str = 'Y',
                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a plot from provided data.

        Args:
            data: Data to plot (various formats accepted)
            plot_type: Type of plot to create
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            options: Additional plotting options

        Returns:
            Dict containing plot path, base64 image, and metadata
        """
        if options is None:
            options = {}

        # Setup the figure
        fig_size = options.get('figsize', (8, 6))
        dpi = options.get('dpi', self.default_dpi)

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        try:
            # Parse the data if needed
            parsed_data = self._parse_data(data)

            # Set the plot style
            plt_style = options.get('style', 'default')
            plt.style.use(plt_style)

            # Create the plot based on type
            if plot_type in self.supported_plot_types:
                plot_function = self.supported_plot_types[plot_type]
                plot_function(ax, parsed_data, options)
            else:
                # Default to line plot
                logger.warning(f"Unsupported plot type '{plot_type}', defaulting to line plot")
                self._create_line_plot(ax, parsed_data, options)

            # Configure plot appearance
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Add grid if requested
            if options.get('grid', False):
                ax.grid(True, alpha=0.3)

            # Add legend if data has labels
            if 'labels' in options or hasattr(parsed_data, 'columns'):
                ax.legend()

            # Save the figure
            timestamp = int(os.path.getmtime(__file__)) if os.path.exists(__file__) else 0
            safe_title = re.sub(r'[^\w\-_]', '_', title)
            filename = f"{safe_title}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)

            plt.tight_layout()
            plt.savefig(filepath, dpi=dpi)

            # Get base64 encoded image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            plt.close(fig)

            logger.info(f"Plot created successfully: {filepath}")

            return {
                "status": "success",
                "plot_path": filepath,
                "base64_image": img_str,
                "title": title,
                "plot_type": plot_type,
                "filename": filename
            }

        except Exception as e:
            plt.close(fig)
            logger.error(f"Error creating plot: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _parse_data(self, data: Any) -> Any:
        """
        Parse input data into a format suitable for plotting.

        Args:
            data: Input data (string, list, dict, etc.)

        Returns:
            Parsed data ready for plotting
        """
        if isinstance(data, str):
            # Try to parse as JSON
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Try to parse as CSV/TSV
                if ',' in data or '\t' in data:
                    lines = data.strip().split('\n')
                    if ',' in data:
                        return [line.split(',') for line in lines]
                    else:
                        return [line.split('\t') for line in lines]
                else:
                    # Try to parse as a simple list of numbers
                    try:
                        return [float(x) for x in data.split()]
                    except ValueError:
                        raise ValueError("Could not parse string data for plotting")

        elif isinstance(data, dict):
            # Dictionary data can be used directly for most plots
            return data

        elif isinstance(data, (list, tuple)):
            # Check if it's a list of lists/tuples (2D data)
            if data and isinstance(data[0], (list, tuple)):
                # Convert to columns if needed
                if len(data[0]) == 2:  # x,y pairs
                    x = [item[0] for item in data]
                    y = [item[1] for item in data]
                    return {'x': x, 'y': y}
                else:
                    return data
            else:
                # Simple 1D list, use indices as x-values
                return {'x': list(range(len(data))), 'y': data}

        # For NumPy arrays, Pandas DataFrames, etc.
        return data

    def _create_line_plot(self, ax, data, options):
        """Create a line plot."""
        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                # Simple x,y data
                x = data['x']
                y = data['y']

                if isinstance(y[0], (list, tuple)) or ('y2' in data):
                    # Multiple lines
                    if 'y2' in data:
                        # Named y-series
                        for key, y_vals in data.items():
                            if key != 'x' and isinstance(y_vals, (list, tuple)):
                                label = options.get('labels', {}).get(key, key)
                                ax.plot(x, y_vals, label=label)
                    else:
                        # List of y-values for each line
                        for i, y_vals in enumerate(y):
                            labels = options.get('labels', [])
                            label = labels[i] if i < len(labels) else f"Series {i+1}"
                            ax.plot(x, y_vals, label=label)
                else:
                    # Single line
                    label = options.get('label', "Data")
                    ax.plot(x, y, label=label)
            else:
                # Dictionary of series
                for key, values in data.items():
                    label = options.get('labels', {}).get(key, key)
                    ax.plot(range(len(values)), values, label=label)
        else:
            # Simple list-like data
            ax.plot(data, label=options.get('label', "Data"))

    def _create_bar_plot(self, ax, data, options):
        """Create a bar plot."""
        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                # Simple x,y data
                x = data['x']
                y = data['y']

                if isinstance(y[0], (list, tuple)) or ('y2' in data):
                    # Grouped bars
                    if 'y2' in data:
                        # Named y-series
                        series = []
                        labels = []
                        for key, y_vals in data.items():
                            if key != 'x' and isinstance(y_vals, (list, tuple)):
                                series.append(y_vals)
                                labels.append(options.get('labels', {}).get(key, key))

                        x_pos = np.arange(len(x))
                        width = 0.8 / len(series)

                        for i, (s, label) in enumerate(zip(series, labels)):
                            ax.bar(x_pos + i*width - 0.4 + width/2, s, width, label=label)

                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(x)
                    else:
                        # List of y-values for each group
                        x_pos = np.arange(len(x))
                        width = 0.8 / len(y)

                        for i, y_vals in enumerate(y):
                            labels = options.get('labels', [])
                            label = labels[i] if i < len(labels) else f"Series {i+1}"
                            ax.bar(x_pos + i*width - 0.4 + width/2, y_vals, width, label=label)

                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(x)
                else:
                    # Single series of bars
                    ax.bar(x, y, label=options.get('label', "Data"))
            else:
                # Dictionary of series
                ax.bar(list(data.keys()), list(data.values()))
        else:
            # Simple list-like data
            ax.bar(range(len(data)), data, label=options.get('label', "Data"))

    def _create_scatter_plot(self, ax, data, options):
        """Create a scatter plot."""
        if isinstance(data, dict) and 'x' in data and 'y' in data:
            x = data['x']
            y = data['y']

            # Optional color and size parameters
            c = data.get('color', options.get('color', None))
            s = data.get('size', options.get('size', None))

            ax.scatter(x, y, c=c, s=s, label=options.get('label', "Data"))

            # Add a colorbar if colors are provided
            if c is not None and options.get('colorbar', False):
                plt.colorbar(ax.collections[0], ax=ax)
        else:
            logger.error("Scatter plot requires x and y data")
            raise ValueError("Scatter plot requires 'x' and 'y' data")

    def _create_pie_chart(self, ax, data, options):
        """Create a pie chart."""
        if isinstance(data, dict):
            # Dictionary of labels -> values
            if 'values' in data:
                values = data['values']
                labels = data.get('labels', list(range(len(values))))
            else:
                values = list(data.values())
                labels = list(data.keys())

            # Optional pie chart settings
            explode = options.get('explode', None)
            autopct = options.get('autopct', '%1.1f%%')

            ax.pie(values, labels=labels, explode=explode, autopct=autopct)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        else:
            # Simple list of values
            ax.pie(data, labels=options.get('labels', None), autopct='%1.1f%%')
            ax.axis('equal')

    def _create_histogram(self, ax, data, options):
        """Create a histogram."""
        if isinstance(data, dict) and 'values' in data:
            values = data['values']
        elif isinstance(data, dict) and 'y' in data:
            values = data['y']
        elif isinstance(data, (list, tuple, np.ndarray)):
            values = data
        else:
            values = list(data.values()) if isinstance(data, dict) else data

        bins = options.get('bins', 10)
        ax.hist(values, bins=bins, alpha=0.7, label=options.get('label', "Data"))

    def _create_box_plot(self, ax, data, options):
        """Create a box plot."""
        if isinstance(data, dict):
            if 'groups' in data:
                # Multiple groups
                ax.boxplot(data['groups'], labels=data.get('labels', None))
            elif all(isinstance(v, (list, tuple)) for v in data.values()):
                # Dictionary of group names -> values
                ax.boxplot(list(data.values()), labels=list(data.keys()))
            else:
                # Single group
                ax.boxplot(list(data.values()))
        else:
            # Simple list of values
            ax.boxplot(data)

# Create an instance for use in tool_router
graph_plotter = GraphPlotter()

def graph_tool_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle graph plotting requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Graph plotting result
    """
    raw_input = user_input.get("raw_input", "")

    # Extract data to plot
    data = None
    # Check if we have Python code output or computation result
    if "computation_result" in context:
        data = context["computation_result"].get("result")

    # If no data from context, try to extract from input
    if data is None:
        # Try to find data in input using regex patterns
        data_match = re.search(r'data\s*[:=]\s*(\[.*?\]|\{.*?\})', raw_input, re.DOTALL)
        if data_match:
            try:
                data = json.loads(data_match.group(1))
            except json.JSONDecodeError:
                # Not valid JSON, try as Python literal
                data_str = data_match.group(1)

                # Basic conversion of Python literal to JSON-compatible string
                # Replace Python syntax with JSON syntax
                data_str = data_str.replace("'", '"')  # Replace single quotes with double quotes
                data_str = re.sub(r"(\w+):", r'"\1":', data_str)  # Add quotes to keys

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    pass

    # If still no data, try to parse numbers from the input
    if data is None:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', raw_input)
        if numbers:
            data = [float(num) for num in numbers]

    # Determine plot type
    plot_type = 'line'  # default
    if 'bar' in raw_input.lower():
        plot_type = 'bar'
    elif 'scatter' in raw_input.lower():
        plot_type = 'scatter'
    elif 'pie' in raw_input.lower():
        plot_type = 'pie'
    elif 'histogram' in raw_input.lower() or 'hist' in raw_input.lower():
        plot_type = 'histogram'
    elif 'box' in raw_input.lower():
        plot_type = 'boxplot'

    # Extract plot title and labels
    title_match = re.search(r'title\s*[:=]\s*["\']([^"\']+)["\']', raw_input)
    title = title_match.group(1) if title_match else "Data Visualization"

    xlabel_match = re.search(r'x(?:label|axis)\s*[:=]\s*["\']([^"\']+)["\']', raw_input)
    xlabel = xlabel_match.group(1) if xlabel_match else "X"

    ylabel_match = re.search(r'y(?:label|axis)\s*[:=]\s*["\']([^"\']+)["\']', raw_input)
    ylabel = ylabel_match.group(1) if ylabel_match else "Y"

    # Extract additional options
    options = {}

    # Check for grid
    if 'grid' in raw_input.lower():
        options['grid'] = True

    # Set style
    if 'dark' in raw_input.lower():
        options['style'] = 'dark_background'
    elif 'seaborn' in raw_input.lower():
        options['style'] = 'seaborn'

    # Create the plot if we have data
    if data is not None:
        result = graph_plotter.create_plot(
            data=data,
            plot_type=plot_type,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            options=options
        )

        if result["status"] == "success":
            # Add HTML for embedding the image
            img_html = f'<img src="data:image/png;base64,{result["base64_image"]}" alt="{title}" />'
            result["html"] = img_html

            # Add path info
            result["content"] = f"Graph created successfully: {result['plot_path']}"

            return result
        else:
            return {
                "status": "error",
                "content": f"Failed to create graph: {result.get('error', 'Unknown error')}",
                "error": result.get('error')
            }
    else:
        return {
            "status": "error",
            "content": "Could not extract data to plot from your input. Please provide data in a format like [1, 2, 3, 4] or {x: [1, 2, 3], y: [10, 20, 30]}.",
            "error": "No data available for plotting"
        }

if __name__ == "__main__":
    # Test the graph plotter
    plotter = GraphPlotter()

    # Test data
    test_data = {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 35, 25, 45, 30]
    }

    # Create a test plot
    result = plotter.create_plot(
        data=test_data,
        title="Test Plot",
        xlabel="X-Axis",
        ylabel="Y-Axis",
        options={"grid": True}
    )

    print(f"Plot saved to: {result['plot_path']}")
