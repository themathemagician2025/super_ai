# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Response Generator Module

Formats the final output to be presented to the user.
"""

import json
import markdown
from typing import Dict, Any, List, Optional

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator."""
        self.formatters = {
            "text": self._format_text,
            "json": self._format_json,
            "markdown": self._format_markdown,
            "table": self._format_table,
            "error": self._format_error
        }

    def generate(self, response_data: Dict[str, Any],
                requested_format: str = "text") -> str:
        """
        Generate formatted response from response data.

        Args:
            response_data: Data to be formatted
            requested_format: Format to return (text, json, markdown, etc.)

        Returns:
            Formatted response string
        """
        # Handle errors specially
        if response_data.get("status") == "error":
            return self.formatters["error"](response_data)

        # Use requested format if available, otherwise fall back to text
        formatter = self.formatters.get(requested_format, self.formatters["text"])
        return formatter(response_data)

    def _format_text(self, data: Dict[str, Any]) -> str:
        """Format as plain text."""
        # Pull out content if it exists
        if "content" in data:
            return str(data["content"])

        # For simple responses
        elif "message" in data:
            return str(data["message"])

        # For data responses, create a simple text representation
        else:
            lines = []

            # Add title if exists
            if "title" in data:
                lines.append(data["title"])
                lines.append("=" * len(data["title"]))
                lines.append("")

            # Add description if exists
            if "description" in data:
                lines.append(data["description"])
                lines.append("")

            # Handle data items
            if "items" in data and isinstance(data["items"], list):
                for item in data["items"]:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            lines.append(f"{key}: {value}")
                        lines.append("")
                    else:
                        lines.append(str(item))

            # Return joined lines
            return "\n".join(lines).strip()

    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format as JSON string."""
        # Remove potentially large or sensitive fields
        safe_data = {k: v for k, v in data.items()
                    if k not in ["raw_response", "internal_data"]}
        return json.dumps(safe_data, indent=2)

    def _format_markdown(self, data: Dict[str, Any]) -> str:
        """Format as Markdown."""
        md_lines = []

        # Title
        if "title" in data:
            md_lines.append(f"# {data['title']}")
            md_lines.append("")

        # Description
        if "description" in data:
            md_lines.append(data["description"])
            md_lines.append("")

        # Content
        if "content" in data:
            if isinstance(data["content"], str):
                md_lines.append(data["content"])
            elif isinstance(data["content"], list):
                for item in data["content"]:
                    md_lines.append(f"- {item}")
            md_lines.append("")

        # Data/results
        if "results" in data:
            if isinstance(data["results"], list):
                md_lines.append("## Results")
                md_lines.append("")

                for item in data["results"]:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            md_lines.append(f"**{k}**: {v}")
                        md_lines.append("")
                    else:
                        md_lines.append(f"- {item}")
                md_lines.append("")

        # Return joined
        return "\n".join(md_lines).strip()

    def _format_table(self, data: Dict[str, Any]) -> str:
        """Format as plain text table."""
        if "items" not in data or not isinstance(data["items"], list) or not data["items"]:
            return "No data available for table display"

        items = data["items"]

        # Get all keys
        all_keys = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        if not all_keys:
            return "No data available for table display"

        # Convert set to list for consistent ordering
        headers = sorted(list(all_keys))

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for item in items:
            if isinstance(item, dict):
                for i, key in enumerate(headers):
                    if key in item:
                        col_widths[i] = max(col_widths[i], len(str(item[key])))

        # Build table
        lines = []

        # Header row
        header_row = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_row)

        # Separator row
        sep_row = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
        lines.append(sep_row)

        # Data rows
        for item in items:
            if isinstance(item, dict):
                row = " | ".join(
                    str(item.get(key, "")).ljust(col_widths[i])
                    for i, key in enumerate(headers)
                )
                lines.append(row)

        return "\n".join(lines)

    def _format_error(self, data: Dict[str, Any]) -> str:
        """Format error response."""
        error_msg = data.get("error", "An unknown error occurred")
        return f"ERROR: {error_msg}"
