# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Web Scraping Module for Super AI Prediction System
Implements various scraping strategies using BeautifulSoup4, Selenium, and Scrapy
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

# BeautifulSoup4
from bs4 import BeautifulSoup

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Scrapy (used in separate crawler class)
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Base class for all scrapers"""

    def __init__(self, output_dir: str = "scraped_data"):
        """
        Initialize the base scraper.

        Args:
            output_dir: Directory to save scraped data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    @abstractmethod
    def scrape(self, url: str, **kwargs) -> Dict:
        """
        Scrape data from the given URL

        Args:
            url: URL to scrape
            **kwargs: Additional arguments

        Returns:
            Scraped data
        """
        pass

    def save_results(self, filename: str, format: str = "json") -> str:
        """
        Save scraped results to file

        Args:
            filename: Name of the file (without extension)
            format: File format (json, csv, etc.)

        Returns:
            Path to saved file
        """
        if not self.results:
            logger.warning("No results to save")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"

        if format == "json":
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)

        elif format == "csv":
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            pd.DataFrame(self.results).to_csv(filepath, index=False)

        elif format == "parquet":
            filepath = os.path.join(self.output_dir, f"{filename}.parquet")
            pd.DataFrame(self.results).to_parquet(filepath, index=False)

        else:
            logger.error(f"Unsupported format: {format}")
            return ""

        logger.info(f"Results saved to {filepath}")
        return filepath

    def clear_results(self) -> None:
        """Clear the current results"""
        self.results = []


class SoupScraper(BaseScraper):
    """Web scraper using BeautifulSoup4"""

    def __init__(self, output_dir: str = "scraped_data",
                headers: Dict = None, timeout: int = 30):
        """
        Initialize the BeautifulSoup scraper.

        Args:
            output_dir: Directory to save scraped data
            headers: HTTP headers for requests
            timeout: Request timeout in seconds
        """
        super().__init__(output_dir)
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape(self, url: str, selector: str = None,
              parser: str = "html.parser", extractor: Callable = None,
              **kwargs) -> Dict:
        """
        Scrape data from the given URL using BeautifulSoup

        Args:
            url: URL to scrape
            selector: CSS selector to extract data
            parser: HTML parser to use
            extractor: Custom function to extract data from soup
            **kwargs: Additional arguments

        Returns:
            Scraped data
        """
        try:
            logger.info(f"Scraping {url} with BeautifulSoup")
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, parser)

            if extractor and callable(extractor):
                # Use custom extractor function
                data = extractor(soup)
            elif selector:
                # Use provided CSS selector
                elements = soup.select(selector)
                data = [element.get_text().strip() for element in elements]
            else:
                # Default extraction
                data = {
                    "title": soup.title.string if soup.title else "",
                    "text": soup.get_text().strip()
                }

            result = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {"url": url, "error": str(e)}

    def scrape_multiple(self, urls: List[str], max_workers: int = 5, **kwargs) -> List[Dict]:
        """
        Scrape multiple URLs in parallel

        Args:
            urls: List of URLs to scrape
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments for scrape()

        Returns:
            List of scraped data
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.scrape, url, **kwargs): url for url in urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")

        return results


class SeleniumScraper(BaseScraper):
    """Web scraper using Selenium for JavaScript-rendered pages"""

    def __init__(self, output_dir: str = "scraped_data", headless: bool = True,
                chrome_driver_path: str = None):
        """
        Initialize the Selenium scraper.

        Args:
            output_dir: Directory to save scraped data
            headless: Whether to run browser in headless mode
            chrome_driver_path: Path to Chrome driver (if not using webdriver-manager)
        """
        super().__init__(output_dir)
        self.headless = headless
        self.chrome_driver_path = chrome_driver_path
        self.driver = None

    def _initialize_driver(self):
        """Initialize the Selenium WebDriver"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        # Use either specific driver path or let webdriver-manager handle it
        if self.chrome_driver_path:
            service = Service(executable_path=self.chrome_driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                          options=chrome_options)

        self.driver.implicitly_wait(10)  # seconds

    def scrape(self, url: str, wait_time: int = 5,
              selector: str = None, screenshot: bool = False,
              wait_for_selector: str = None, actions: List[Dict] = None,
              **kwargs) -> Dict:
        """
        Scrape data from the given URL using Selenium

        Args:
            url: URL to scrape
            wait_time: Time to wait after page load (seconds)
            selector: CSS selector to extract data
            screenshot: Whether to capture screenshot
            wait_for_selector: Wait for a specific element to appear
            actions: List of actions to perform before scraping
            **kwargs: Additional arguments

        Returns:
            Scraped data
        """
        if self.driver is None:
            self._initialize_driver()

        try:
            logger.info(f"Scraping {url} with Selenium")
            self.driver.get(url)

            # Wait for selector if provided
            if wait_for_selector:
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )
            else:
                time.sleep(wait_time)  # Default wait

            # Perform actions if any
            if actions:
                for action in actions:
                    action_type = action.get("type")

                    if action_type == "click":
                        element = self.driver.find_element(By.CSS_SELECTOR, action.get("selector"))
                        element.click()
                        time.sleep(action.get("wait", 1))

                    elif action_type == "input":
                        element = self.driver.find_element(By.CSS_SELECTOR, action.get("selector"))
                        element.clear()
                        element.send_keys(action.get("value", ""))
                        time.sleep(action.get("wait", 0.5))

                    elif action_type == "scroll":
                        scroll_script = f"window.scrollBy(0, {action.get('pixels', 500)});"
                        self.driver.execute_script(scroll_script)
                        time.sleep(action.get("wait", 0.5))

            # Take screenshot if requested
            screenshot_path = None
            if screenshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.png")
                self.driver.save_screenshot(screenshot_path)

            # Extract data
            if selector:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                data = [element.text for element in elements]
            else:
                data = {
                    "title": self.driver.title,
                    "content": self.driver.find_element(By.TAG_NAME, "body").text
                }

            # Get HTML source
            html_source = self.driver.page_source

            result = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "html_source": html_source if kwargs.get("include_html", False) else None,
                "screenshot_path": screenshot_path
            }

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error scraping {url} with Selenium: {str(e)}")
            return {"url": url, "error": str(e)}

    def close(self):
        """Close the Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None


class ScrapySpider(scrapy.Spider):
    """Base Scrapy Spider class"""

    name = "super_ai_spider"

    def __init__(self, *args, **kwargs):
        super(ScrapySpider, self).__init__(*args, **kwargs)
        # Get URLs from kwargs or use default
        self.start_urls = kwargs.get('start_urls', [])
        self.selector = kwargs.get('selector', None)
        self.follow_links = kwargs.get('follow_links', False)
        self.max_depth = kwargs.get('max_depth', 1)
        self.callback_method = kwargs.get('callback_method', None)

    def parse(self, response):
        """Parse response from Scrapy"""
        depth = response.meta.get('depth', 0)

        if self.callback_method and hasattr(self, self.callback_method):
            method = getattr(self, self.callback_method)
            yield from method(response)
            return

        # Default parsing logic
        if self.selector:
            for element in response.css(self.selector):
                yield {
                    'url': response.url,
                    'timestamp': datetime.now().isoformat(),
                    'text': element.get(),
                    'depth': depth
                }
        else:
            yield {
                'url': response.url,
                'timestamp': datetime.now().isoformat(),
                'title': response.css('title::text').get(),
                'body': response.css('body::text').getall(),
                'depth': depth
            }

        # Follow links if configured
        if self.follow_links and depth < self.max_depth:
            for href in response.css('a::attr(href)').getall():
                yield response.follow(href, meta={'depth': depth + 1})


class ScrapyScraper(BaseScraper):
    """Web scraper using Scrapy for large-scale crawling"""

    def __init__(self, output_dir: str = "scraped_data", settings=None):
        """
        Initialize the Scrapy scraper.

        Args:
            output_dir: Directory to save scraped data
            settings: Scrapy settings dictionary
        """
        super().__init__(output_dir)
        self.process = None
        self.settings = settings or {
            'USER_AGENT': 'Super AI Prediction System (+https://example.com)',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 16,
            'DOWNLOAD_DELAY': 0.5,
            'COOKIES_ENABLED': False,
            'FEED_FORMAT': 'json',
            'FEED_URI': os.path.join(output_dir, 'scrapy_output.json'),
        }

    def scrape(self, url: str, selector: str = None,
              follow_links: bool = False, max_depth: int = 1,
              spider_class=None, callback_method: str = None,
              **kwargs) -> Dict:
        """
        Scrape data using Scrapy

        Args:
            url: URL to scrape (or list of URLs)
            selector: CSS selector to extract data
            follow_links: Whether to follow links
            max_depth: Maximum depth for link following
            spider_class: Custom spider class
            callback_method: Name of callback method in spider class
            **kwargs: Additional arguments

        Returns:
            Dict with output file path
        """
        try:
            logger.info(f"Starting Scrapy crawler for {url}")

            # Set up crawler process
            self.process = CrawlerProcess(self.settings)

            # Convert single URL to list
            urls = [url] if isinstance(url, str) else url

            # Generate unique output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"scrapy_output_{timestamp}.json")
            self.settings['FEED_URI'] = output_file

            # Use custom spider or default
            spider = spider_class or ScrapySpider

            # Start the crawler
            self.process.crawl(
                spider,
                start_urls=urls,
                selector=selector,
                follow_links=follow_links,
                max_depth=max_depth,
                callback_method=callback_method,
                **kwargs
            )
            self.process.start()  # Blocks until crawling is finished

            # Return result path
            return {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "output_file": output_file
            }

        except Exception as e:
            logger.error(f"Error running Scrapy crawler: {str(e)}")
            return {"url": url, "error": str(e)}

    def load_results(self, filepath: str = None) -> List[Dict]:
        """
        Load results from Scrapy output file

        Args:
            filepath: Path to results file (default: latest output)

        Returns:
            List of scraped items
        """
        if filepath is None:
            # Find latest file
            json_files = list(Path(self.output_dir).glob("scrapy_output_*.json"))
            if not json_files:
                logger.warning("No Scrapy output files found")
                return []

            filepath = str(max(json_files, key=os.path.getctime))

        with open(filepath, 'r') as f:
            self.results = json.load(f)

        return self.results


class ScraperFactory:
    """Factory for creating scrapers"""

    @staticmethod
    def create_scraper(scraper_type: str, **kwargs) -> BaseScraper:
        """
        Create a scraper of the specified type

        Args:
            scraper_type: Type of scraper (soup, selenium, scrapy)
            **kwargs: Additional arguments for the scraper

        Returns:
            Initialized scraper
        """
        if scraper_type.lower() == "soup":
            return SoupScraper(**kwargs)
        elif scraper_type.lower() == "selenium":
            return SeleniumScraper(**kwargs)
        elif scraper_type.lower() == "scrapy":
            return ScrapyScraper(**kwargs)
        else:
            raise ValueError(f"Unknown scraper type: {scraper_type}")
