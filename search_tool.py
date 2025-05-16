# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Search Tool Module

Provides web search functionality using DuckDuckGo.
"""

import logging
import json
import re
import time
import requests
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class SearchTool:
    """Tool for searching the web using DuckDuckGo."""

    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize the search tool.

        Args:
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query.

        Args:
            query: Search query

        Returns:
            List of search results (title, description, url)
        """
        logger.info(f"Searching DuckDuckGo for: {query}")

        try:
            encoded_query = quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            result_elements = soup.select('.result')

            for result in result_elements[:self.max_results]:
                title_element = result.select_one('.result__title')
                snippet_element = result.select_one('.result__snippet')
                url_element = result.select_one('.result__url')

                if title_element and snippet_element:
                    title = title_element.get_text(strip=True)
                    snippet = snippet_element.get_text(strip=True)

                    # Extract URL
                    url = ""
                    link_element = title_element.select_one('a')
                    if link_element and 'href' in link_element.attrs:
                        url_match = re.search(r'uddg=([^&]+)', link_element['href'])
                        if url_match:
                            url = requests.utils.unquote(url_match.group(1))

                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'url': url
                    })

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching DuckDuckGo: {str(e)}")
            return []

    def search_serpapi(self, query: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Search using SerpAPI (requires API key).

        Args:
            query: Search query
            api_key: SerpAPI key

        Returns:
            List of search results
        """
        if not api_key:
            logger.error("SerpAPI requires an API key")
            return []

        logger.info(f"Searching SerpAPI for: {query}")

        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": self.max_results
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            results = []
            if "organic_results" in data:
                for result in data["organic_results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("link", "")
                    })

            logger.info(f"Found {len(results)} results using SerpAPI for query: {query}")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching with SerpAPI: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding SerpAPI response: {str(e)}")
            return []

# Create an instance for use in tool_router
search_tool = SearchTool()

def search_tool_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle web search requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Search results
    """
    # Extract the query from user input
    raw_input = user_input.get("raw_input", "")

    # Try to extract specific search query if available
    search_match = re.search(r"search(?:\s+for)?\s+(?:about\s+)?[\"']?([^\"']+)[\"']?", raw_input, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip()
    else:
        # Use the raw input as the query
        query = raw_input.strip()

    # Get search preferences from context if available
    search_preferences = context.get("preferences", {}).get("search", {})
    use_serpapi = search_preferences.get("use_serpapi", False)
    serpapi_key = search_preferences.get("serpapi_key", None)

    # Perform the search
    if use_serpapi and serpapi_key:
        results = search_tool.search_serpapi(query, serpapi_key)
    else:
        results = search_tool.search_duckduckgo(query)

    # Format the results
    if results:
        return {
            "title": f"Search Results for '{query}'",
            "status": "success",
            "items": results
        }
    else:
        return {
            "title": f"Search Results for '{query}'",
            "status": "error",
            "error": "No results found or error occurred during search",
            "items": []
        }

if __name__ == "__main__":
    # Test the search tool
    tool = SearchTool()
    results = tool.search_duckduckgo("Python programming language")
    print(json.dumps(results, indent=2))
