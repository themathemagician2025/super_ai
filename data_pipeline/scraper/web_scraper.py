# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Web Scraper Module for Sports Data

This module provides automated web scraping capabilities for:
1. Match scores and statistics
2. Betting odds from bookmakers
3. Team information and news

Uses headless browser automation to navigate dynamic websites.
"""

import logging
import time
import json
import csv
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from urllib.parse import urlparse, urljoin

# Optional imports - will be checked at runtime
SELENIUM_AVAILABLE = False
BEAUTIFULSOUP_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    pass

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = False
except ImportError:
    pass

logger = logging.getLogger(__name__)

class WebScraper:
    """Base web scraper with common functionality"""

    def __init__(self, config: Dict = None):
        """
        Initialize the web scraper

        Args:
            config: Configuration dictionary with scraper settings
        """
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'data/scraped'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Browser settings
        self.headless = self.config.get('headless', True)
        self.timeout = self.config.get('timeout', 30)
        self.driver = None
        self.user_agent = self.config.get('user_agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36')

        # Rate limiting to avoid being blocked
        self.request_delay = self.config.get('request_delay', 2)
        self.last_request_time = 0

        # Initialize the browser if configured for immediate start
        if self.config.get('auto_start', False):
            self.start_browser()

    def start_browser(self) -> bool:
        """Initialize and start the headless browser"""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium is not installed. Cannot start browser.")
            return False

        try:
            options = Options()
            if self.headless:
                options.add_argument('--headless')

            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.user_agent}')

            # Add custom browser settings from config
            for arg in self.config.get('browser_args', []):
                options.add_argument(arg)

            # Set up ChromeDriver (can be customized for other browsers)
            driver_path = self.config.get('driver_path')
            if driver_path:
                service = Service(driver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)

            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Browser started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}")
            return False

    def stop_browser(self):
        """Close the browser and clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser stopped")
            except Exception as e:
                logger.error(f"Error stopping browser: {str(e)}")
            finally:
                self.driver = None

    def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL with rate limiting

        Args:
            url: URL to navigate to

        Returns:
            Success status
        """
        if not self.driver:
            logger.error("Browser not started. Call start_browser() first.")
            return False

        # Apply rate limiting
        self._apply_rate_limit()

        try:
            self.driver.get(url)
            logger.info(f"Navigated to {url}")
            return True
        except TimeoutException:
            logger.warning(f"Timeout while loading {url}")
            return False
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            return False

    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def find_element(self, selector: str, by: str = By.CSS_SELECTOR,
                    wait_time: int = None) -> Optional[Any]:
        """
        Find an element in the page

        Args:
            selector: Element selector
            by: Selection method (CSS_SELECTOR, XPATH, etc.)
            wait_time: Time to wait for element to appear

        Returns:
            Element if found, None otherwise
        """
        if not self.driver:
            logger.error("Browser not started. Call start_browser() first.")
            return None

        wait_time = wait_time or self.timeout

        try:
            element = WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            logger.warning(f"Element not found within {wait_time}s: {selector}")
            return None
        except Exception as e:
            logger.error(f"Error finding element {selector}: {str(e)}")
            return None

    def find_elements(self, selector: str, by: str = By.CSS_SELECTOR,
                     wait_time: int = None) -> List[Any]:
        """
        Find multiple elements in the page

        Args:
            selector: Element selector
            by: Selection method (CSS_SELECTOR, XPATH, etc.)
            wait_time: Time to wait for elements to appear

        Returns:
            List of elements found
        """
        if not self.driver:
            logger.error("Browser not started. Call start_browser() first.")
            return []

        wait_time = wait_time or self.timeout

        try:
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((by, selector))
            )
            elements = self.driver.find_elements(by, selector)
            return elements
        except TimeoutException:
            logger.warning(f"Elements not found within {wait_time}s: {selector}")
            return []
        except Exception as e:
            logger.error(f"Error finding elements {selector}: {str(e)}")
            return []

    def save_to_json(self, data: Any, filename: str) -> bool:
        """
        Save data to JSON file

        Args:
            data: Data to save
            filename: Name of the file (without extension)

        Returns:
            Success status
        """
        filepath = self.data_dir / f"{filename}.json"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            return False

    def save_to_csv(self, data: List[Dict], filename: str) -> bool:
        """
        Save data to CSV file

        Args:
            data: List of dictionaries with uniform keys
            filename: Name of the file (without extension)

        Returns:
            Success status
        """
        if not data:
            logger.warning("No data to save to CSV")
            return False

        filepath = self.data_dir / f"{filename}.csv"

        try:
            # Get fieldnames from first item
            fieldnames = list(data[0].keys())

            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            return False

    def extract_page_content(self) -> str:
        """Extract the full page content as HTML"""
        if not self.driver:
            logger.error("Browser not started. Call start_browser() first.")
            return ""

        return self.driver.page_source

    def parse_html(self, html: str) -> Optional[Any]:
        """
        Parse HTML with BeautifulSoup

        Args:
            html: HTML content to parse

        Returns:
            BeautifulSoup object if available, None otherwise
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            logger.error("BeautifulSoup is not installed. Cannot parse HTML.")
            return None

        try:
            return BeautifulSoup(html, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML: {str(e)}")
            return None

class ScoresScraperModule:
    """Module for scraping sports scores and match statistics"""

    def __init__(self, scraper: WebScraper):
        """
        Initialize scores scraper module

        Args:
            scraper: WebScraper instance to use
        """
        self.scraper = scraper
        self.score_sites = {
            'site1': {
                'url': 'https://www.example-scores.com/matches/{date}',
                'match_selector': '.match-item',
                'home_team_selector': '.home-team',
                'away_team_selector': '.away-team',
                'score_selector': '.score',
                'date_format': '%Y-%m-%d'
            },
            'site2': {
                'url': 'https://www.example-livescore.com/{league}/results',
                'match_selector': '.match-row',
                'home_team_selector': '.team1',
                'away_team_selector': '.team2',
                'score_selector': '.result',
                'date_selector': '.match-date'
            }
        }

        # Update with configuration if provided
        config_sites = self.scraper.config.get('score_sites', {})
        self.score_sites.update(config_sites)

    def scrape_scores(self, site_key: str, date: Union[str, datetime] = None) -> List[Dict]:
        """
        Scrape scores from a specific site

        Args:
            site_key: Key of the site to scrape from
            date: Date to scrape scores for (if site supports it)

        Returns:
            List of match data dictionaries
        """
        if site_key not in self.score_sites:
            logger.error(f"Unknown score site: {site_key}")
            return []

        site_config = self.score_sites[site_key]

        # Format URL with date if needed
        url = site_config['url']
        if '{date}' in url and date:
            # Convert to string if it's a datetime
            if isinstance(date, datetime):
                date_format = site_config.get('date_format', '%Y-%m-%d')
                date_str = date.strftime(date_format)
            else:
                date_str = date

            url = url.replace('{date}', date_str)

        # Navigate to the page
        success = self.scraper.navigate_to(url)
        if not success:
            return []

        # Wait for match elements to load
        matches = self.scraper.find_elements(site_config['match_selector'])
        if not matches:
            logger.warning(f"No matches found on {url}")
            return []

        # Extract match data
        match_data = []
        for match in matches:
            try:
                # Extract team names
                home_team_elem = match.find_element(By.CSS_SELECTOR, site_config['home_team_selector'])
                away_team_elem = match.find_element(By.CSS_SELECTOR, site_config['away_team_selector'])
                score_elem = match.find_element(By.CSS_SELECTOR, site_config['score_selector'])

                home_team = home_team_elem.text.strip()
                away_team = away_team_elem.text.strip()
                score_text = score_elem.text.strip()

                # Parse score (assuming format like "2 - 1")
                score_parts = re.split(r'\s*-\s*', score_text)
                if len(score_parts) == 2:
                    try:
                        home_score = int(score_parts[0])
                        away_score = int(score_parts[1])
                    except ValueError:
                        # Handle non-numeric scores (e.g., "PP" for postponed)
                        home_score = away_score = None
                else:
                    home_score = away_score = None

                # Extract match date if available
                match_date = None
                if 'date_selector' in site_config:
                    try:
                        date_elem = match.find_element(By.CSS_SELECTOR, site_config['date_selector'])
                        match_date = date_elem.text.strip()
                    except NoSuchElementException:
                        pass

                match_data.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'score_text': score_text,
                    'match_date': match_date or (date if date else datetime.now().strftime('%Y-%m-%d')),
                    'source': site_key
                })

            except NoSuchElementException as e:
                logger.warning(f"Failed to extract match data: {str(e)}")
                continue

        logger.info(f"Scraped {len(match_data)} matches from {site_key}")
        return match_data

    def scrape_match_details(self, match_url: str) -> Dict:
        """
        Scrape detailed statistics for a specific match

        Args:
            match_url: URL of the match details page

        Returns:
            Dictionary with match statistics
        """
        # Navigate to the match page
        success = self.scraper.navigate_to(match_url)
        if not success:
            return {}

        # Custom extraction logic would be implemented here based on the site structure
        # This is a placeholder implementation
        stats = {
            'possession': {'home': 50, 'away': 50},
            'shots': {'home': 0, 'away': 0},
            'shots_on_target': {'home': 0, 'away': 0},
            'corners': {'home': 0, 'away': 0},
            'fouls': {'home': 0, 'away': 0},
            'yellow_cards': {'home': 0, 'away': 0},
            'red_cards': {'home': 0, 'away': 0}
        }

        # Example extraction logic
        try:
            # Find the stats container
            stats_container = self.scraper.find_element('.match-stats')
            if stats_container:
                # Extract possession
                possession_elems = self.scraper.find_elements('.possession-stat')
                if len(possession_elems) >= 2:
                    try:
                        home_possession = int(possession_elems[0].text.replace('%', ''))
                        away_possession = int(possession_elems[1].text.replace('%', ''))
                        stats['possession'] = {'home': home_possession, 'away': away_possession}
                    except (ValueError, IndexError):
                        pass

                # Extract shots
                shots_elems = self.scraper.find_elements('.shots-stat')
                if len(shots_elems) >= 2:
                    try:
                        home_shots = int(shots_elems[0].text)
                        away_shots = int(shots_elems[1].text)
                        stats['shots'] = {'home': home_shots, 'away': away_shots}
                    except (ValueError, IndexError):
                        pass

                # Additional stats would be extracted similarly
        except Exception as e:
            logger.error(f"Error extracting match statistics: {str(e)}")

        return stats

class OddsScraperModule:
    """Module for scraping betting odds from bookmakers"""

    def __init__(self, scraper: WebScraper):
        """
        Initialize odds scraper module

        Args:
            scraper: WebScraper instance to use
        """
        self.scraper = scraper
        self.odds_sites = {
            'site1': {
                'url': 'https://www.example-odds.com/{sport}/{league}',
                'match_selector': '.event-row',
                'home_team_selector': '.home-team',
                'away_team_selector': '.away-team',
                'home_odds_selector': '.home-odds',
                'draw_odds_selector': '.draw-odds',
                'away_odds_selector': '.away-odds'
            },
            'site2': {
                'url': 'https://www.example-bookie.com/sports/{sport}',
                'match_selector': '.match-container',
                'teams_selector': '.team-names',
                'odds_selector': '.outcome-odds'
            }
        }

        # Update with configuration if provided
        config_sites = self.scraper.config.get('odds_sites', {})
        self.odds_sites.update(config_sites)

    def scrape_odds(self, site_key: str, sport: str = 'football',
                   league: str = None) -> List[Dict]:
        """
        Scrape betting odds for matches

        Args:
            site_key: Key of the odds site to scrape from
            sport: Sport to scrape odds for
            league: Specific league to filter by

        Returns:
            List of match odds dictionaries
        """
        if site_key not in self.odds_sites:
            logger.error(f"Unknown odds site: {site_key}")
            return []

        site_config = self.odds_sites[site_key]

        # Format URL with sport and league if needed
        url = site_config['url']
        url = url.replace('{sport}', sport)
        if '{league}' in url and league:
            url = url.replace('{league}', league)
        elif '{league}' in url:
            # Use default if no league specified
            url = url.replace('{league}', 'all')

        # Navigate to the page
        success = self.scraper.navigate_to(url)
        if not success:
            return []

        # Wait for match elements to load
        matches = self.scraper.find_elements(site_config['match_selector'])
        if not matches:
            logger.warning(f"No matches found on {url}")
            return []

        # Extract odds data
        odds_data = []
        for match in matches:
            try:
                # Site type 1 extraction logic
                if all(selector in site_config for selector in ['home_team_selector', 'away_team_selector']):
                    home_team_elem = match.find_element(By.CSS_SELECTOR, site_config['home_team_selector'])
                    away_team_elem = match.find_element(By.CSS_SELECTOR, site_config['away_team_selector'])

                    home_team = home_team_elem.text.strip()
                    away_team = away_team_elem.text.strip()

                    # Extract odds
                    home_odds = draw_odds = away_odds = None

                    if 'home_odds_selector' in site_config:
                        try:
                            home_odds_elem = match.find_element(By.CSS_SELECTOR, site_config['home_odds_selector'])
                            home_odds = float(home_odds_elem.text.strip())
                        except (NoSuchElementException, ValueError):
                            pass

                    if 'draw_odds_selector' in site_config:
                        try:
                            draw_odds_elem = match.find_element(By.CSS_SELECTOR, site_config['draw_odds_selector'])
                            draw_odds = float(draw_odds_elem.text.strip())
                        except (NoSuchElementException, ValueError):
                            pass

                    if 'away_odds_selector' in site_config:
                        try:
                            away_odds_elem = match.find_element(By.CSS_SELECTOR, site_config['away_odds_selector'])
                            away_odds = float(away_odds_elem.text.strip())
                        except (NoSuchElementException, ValueError):
                            pass

                    odds_data.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': home_odds,
                        'draw_odds': draw_odds,
                        'away_odds': away_odds,
                        'timestamp': datetime.now().isoformat(),
                        'source': site_key,
                        'sport': sport,
                        'league': league
                    })

                # Site type 2 extraction logic (different structure)
                elif 'teams_selector' in site_config and 'odds_selector' in site_config:
                    teams_elem = match.find_element(By.CSS_SELECTOR, site_config['teams_selector'])
                    teams_text = teams_elem.text.strip()

                    # Parse teams (assuming format like "Team A vs Team B")
                    teams_parts = re.split(r'\s+vs\s+', teams_text, flags=re.IGNORECASE)
                    if len(teams_parts) == 2:
                        home_team = teams_parts[0].strip()
                        away_team = teams_parts[1].strip()

                        # Extract all odds elements
                        odds_elems = match.find_elements(By.CSS_SELECTOR, site_config['odds_selector'])
                        odds_values = []

                        for elem in odds_elems[:3]:  # Limit to first 3 (home, draw, away)
                            try:
                                odds_values.append(float(elem.text.strip()))
                            except ValueError:
                                odds_values.append(None)

                        # Pad with None if needed
                        odds_values.extend([None] * (3 - len(odds_values)))

                        odds_data.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_odds': odds_values[0],
                            'draw_odds': odds_values[1],
                            'away_odds': odds_values[2],
                            'timestamp': datetime.now().isoformat(),
                            'source': site_key,
                            'sport': sport,
                            'league': league
                        })

            except Exception as e:
                logger.warning(f"Failed to extract odds data: {str(e)}")
                continue

        logger.info(f"Scraped odds for {len(odds_data)} matches from {site_key}")
        return odds_data

    def scrape_historical_odds(self, match_id: str, site_key: str) -> Dict:
        """
        Scrape historical odds movement for a specific match

        Args:
            match_id: ID of the match
            site_key: Key of the odds site to scrape from

        Returns:
            Dictionary with historical odds data
        """
        # This would be implemented based on the specific site's structure
        # for historical odds data. This is a placeholder implementation.
        return {
            'match_id': match_id,
            'odds_history': [],
            'timestamp': datetime.now().isoformat(),
            'source': site_key
        }

class WebScraperManager:
    """Manager for web scraping operations"""

    def __init__(self, config: Dict = None):
        """
        Initialize web scraper manager

        Args:
            config: Configuration dictionary with scraper settings
        """
        self.config = config or {}
        self.scraper = WebScraper(self.config)
        self.scores_module = ScoresScraperModule(self.scraper)
        self.odds_module = OddsScraperModule(self.scraper)

        # Set up logging to file
        log_dir = Path(self.config.get('log_dir', 'logs/scraper'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"scraper_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to root logger if not already added
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename
                  for h in root_logger.handlers):
            root_logger.addHandler(file_handler)

    def start(self):
        """Start the scraper browser"""
        return self.scraper.start_browser()

    def stop(self):
        """Stop the scraper browser"""
        self.scraper.stop_browser()

    def scrape_scores_batch(self, site_keys: List[str] = None,
                           date: Union[str, datetime] = None,
                           save: bool = True) -> Dict[str, List[Dict]]:
        """
        Scrape scores from multiple sites

        Args:
            site_keys: List of site keys to scrape from (or all if None)
            date: Date to scrape scores for
            save: Whether to save results to files

        Returns:
            Dictionary mapping site keys to match data lists
        """
        all_scores = {}

        # Use all sites if none specified
        site_keys = site_keys or list(self.scores_module.score_sites.keys())

        for site_key in site_keys:
            scores = self.scores_module.scrape_scores(site_key, date)
            all_scores[site_key] = scores

            # Save to file if requested
            if save and scores:
                date_str = datetime.now().strftime('%Y%m%d')
                if isinstance(date, datetime):
                    date_str = date.strftime('%Y%m%d')
                elif isinstance(date, str):
                    date_str = date.replace('-', '')

                filename = f"scores_{site_key}_{date_str}"
                self.scraper.save_to_csv(scores, filename)

        return all_scores

    def scrape_odds_batch(self, site_keys: List[str] = None,
                         sport: str = 'football',
                         leagues: List[str] = None,
                         save: bool = True) -> Dict[str, List[Dict]]:
        """
        Scrape odds from multiple sites

        Args:
            site_keys: List of site keys to scrape from (or all if None)
            sport: Sport to scrape odds for
            leagues: List of leagues to scrape (or all if None)
            save: Whether to save results to files

        Returns:
            Dictionary mapping site keys to odds data lists
        """
        all_odds = {}

        # Use all sites if none specified
        site_keys = site_keys or list(self.odds_module.odds_sites.keys())

        # Use 'all' if no leagues specified
        leagues = leagues or ['all']

        for site_key in site_keys:
            site_odds = []

            for league in leagues:
                odds = self.odds_module.scrape_odds(site_key, sport, league)
                site_odds.extend(odds)

            all_odds[site_key] = site_odds

            # Save to file if requested
            if save and site_odds:
                date_str = datetime.now().strftime('%Y%m%d')
                filename = f"odds_{site_key}_{sport}_{date_str}"
                self.scraper.save_to_csv(site_odds, filename)

        return all_odds

    def run_daily_scrape(self, scores: bool = True, odds: bool = True) -> bool:
        """
        Run a complete daily scraping job

        Args:
            scores: Whether to scrape scores
            odds: Whether to scrape odds

        Returns:
            Success status
        """
        success = self.start()
        if not success:
            logger.error("Failed to start browser for daily scrape")
            return False

        try:
            logger.info("Starting daily scrape job")

            # Scrape today's and yesterday's scores
            if scores:
                today = datetime.now()
                yesterday = today - timedelta(days=1)

                logger.info("Scraping yesterday's scores")
                self.scrape_scores_batch(date=yesterday)

                logger.info("Scraping today's scores")
                self.scrape_scores_batch(date=today)

            # Scrape odds for upcoming matches
            if odds:
                logger.info("Scraping current odds")

                # Could be customized to scrape specific sports and leagues
                self.scrape_odds_batch()

            logger.info("Daily scrape job completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during daily scrape: {str(e)}")
            return False

        finally:
            self.stop()

    def trigger_scrape(self, domain: str = "all", target_date: str = None) -> Dict[str, Any]:
        """
        Trigger a scrape operation from UI or API

        Args:
            domain: Domain to scrape ("sports", "forex", "all")
            target_date: Optional date to scrape in YYYY-MM-DD format

        Returns:
            Dictionary with results and status
        """
        logger.info(f"UI/API triggered scrape for domain: {domain}, date: {target_date}")

        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }

        success = self.start()
        if not success:
            logger.error("Failed to start browser for triggered scrape")
            return {
                "status": "failed",
                "error": "Failed to start browser",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Convert date string to datetime if provided
            date_obj = None
            if target_date:
                try:
                    date_obj = datetime.strptime(target_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid date format: {target_date}, using current date")
                    date_obj = datetime.now()
            else:
                date_obj = datetime.now()

            # Scrape based on domain
            if domain in ["sports", "all"]:
                logger.info(f"Scraping sports data for {date_obj.strftime('%Y-%m-%d')}")
                sports_data = self.scrape_scores_batch(date=date_obj)
                results["details"]["sports"] = {
                    "items_scraped": sum(len(matches) for matches in sports_data.values()),
                    "sources": list(sports_data.keys())
                }

            if domain in ["forex", "all"]:
                logger.info("Scraping forex data")
                forex_data = self.scrape_odds_batch(sport="forex")
                results["details"]["forex"] = {
                    "items_scraped": sum(len(odds) for odds in forex_data.values()),
                    "sources": list(forex_data.keys())
                }

            logger.info("Triggered scrape completed successfully")
            return results

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during triggered scrape: {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

        finally:
            self.stop()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create scraper manager
    manager = WebScraperManager({
        'headless': True,
        'data_dir': 'data/scraped',
        'log_dir': 'logs/scraper',
        'request_delay': 2
    })

    # Run daily scrape
    manager.run_daily_scrape()
