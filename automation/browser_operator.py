#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Browser Operator

Provides automation capabilities for web-based tasks using Selenium.
Enables the system to perform tasks like collecting data from websites,
automated testing, and interacting with web interfaces.
"""

import os
import time
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import random
from datetime import datetime

# Try to import selenium - it's optional but recommended
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        TimeoutException, 
        NoSuchElementException,
        ElementClickInterceptedException,
        StaleElementReferenceException
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

class BrowserOperator:
    """
    Automates browser-based tasks using Selenium.
    Handles tasks like data collection, automated testing, and web interactions.
    """
    
    def __init__(self, 
                config_path: str = None,
                headless: bool = True,
                user_agent: str = None,
                screenshot_dir: str = None,
                timeout: int = 30):
        """
        Initialize the browser operator.
        
        Args:
            config_path: Path to browser automation config file
            headless: Whether to run the browser in headless mode
            user_agent: Custom user agent string
            screenshot_dir: Directory to save screenshots
            timeout: Default timeout for browser operations in seconds
        """
        self.config = self._load_config(config_path)
        self.headless = headless
        self.user_agent = user_agent or self._get_random_user_agent()
        self.screenshot_dir = screenshot_dir or "logs/screenshots"
        self.timeout = timeout
        self.driver = None
        
        # Create screenshot directory if it doesn't exist
        if self.screenshot_dir:
            os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Check if Selenium is available
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium is not available. Browser automation will be simulated.")
            
        logger.info("Browser operator initialized")
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load browser automation configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "tasks": {
                "data_collection": {
                    "websites": [
                        {
                            "name": "Forex Factory",
                            "url": "https://www.forexfactory.com/calendar",
                            "selectors": {
                                "table": ".calendar__table",
                                "rows": ".calendar__row",
                                "date": ".calendar__date"
                            },
                            "frequency": "daily"
                        },
                        {
                            "name": "ESPN",
                            "url": "https://www.espn.com/nba/scoreboard",
                            "selectors": {
                                "scores": ".ScoreboardScoreCell",
                                "teams": ".ScoreCell__TeamName"
                            },
                            "frequency": "hourly"
                        }
                    ]
                },
                "test_solving": {
                    "websites": [
                        {
                            "name": "Trading View",
                            "url": "https://www.tradingview.com/chart/",
                            "actions": [
                                {"type": "click", "selector": ".loginButtonsDesktop"},
                                {"type": "wait", "seconds": 2},
                                {"type": "input", "selector": "#email", "value": "{{EMAIL}}"},
                                {"type": "input", "selector": "#password", "value": "{{PASSWORD}}"},
                                {"type": "click", "selector": ".submitButton"}
                            ],
                            "env_vars": ["TRADINGVIEW_EMAIL", "TRADINGVIEW_PASSWORD"]
                        }
                    ]
                }
            },
            "proxy_settings": {
                "use_proxy": False,
                "proxy_list": []
            },
            "browser_settings": {
                "window_size": "1920,1080",
                "disable_images": True,
                "disable_javascript": False,
                "disable_cookies": False
            }
        }
        
        if not config_path:
            logger.info("No config path provided, using default browser automation config")
            return default_config
            
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using default config")
            return default_config
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Browser automation config loaded from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading browser config: {str(e)}")
            return default_config
            
    def _get_random_user_agent(self) -> str:
        """
        Get a random user agent to avoid detection.
        
        Returns:
            Random user agent string
        """
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        ]
        return random.choice(user_agents)
        
    def _initialize_driver(self) -> None:
        """Initialize the Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, returning simulated driver")
            return
            
        try:
            chrome_options = Options()
            
            # Apply browser settings
            settings = self.config.get("browser_settings", {})
            
            if self.headless:
                chrome_options.add_argument("--headless")
                
            # Set window size
            window_size = settings.get("window_size", "1920,1080")
            chrome_options.add_argument(f"--window-size={window_size}")
            
            # Set user agent
            chrome_options.add_argument(f"user-agent={self.user_agent}")
            
            # Disable images if configured
            if settings.get("disable_images", False):
                chrome_options.add_argument("--blink-settings=imagesEnabled=false")
                
            # Disable JavaScript if configured
            if settings.get("disable_javascript", False):
                chrome_options.add_argument("--disable-javascript")
                
            # Disable cookies if configured
            if settings.get("disable_cookies", False):
                chrome_options.add_argument("--disable-cookies")
                
            # Add proxy if configured
            proxy_settings = self.config.get("proxy_settings", {})
            if proxy_settings.get("use_proxy", False) and proxy_settings.get("proxy_list"):
                proxy = random.choice(proxy_settings["proxy_list"])
                chrome_options.add_argument(f"--proxy-server={proxy}")
                
            # Other useful options
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Initialize the WebDriver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {str(e)}")
            self.driver = None
            
    def _close_driver(self) -> None:
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
                
    def take_screenshot(self, name: str = None) -> Optional[str]:
        """
        Take a screenshot of the current browser state.
        
        Args:
            name: Name for the screenshot file
            
        Returns:
            Path to the saved screenshot or None if failed
        """
        if not self.driver:
            logger.warning("Cannot take screenshot: WebDriver not initialized")
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = name or f"screenshot_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            self.driver.save_screenshot(filepath)
            logger.info(f"Screenshot saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return None
            
    def run_task(self, task_name: str) -> Dict[str, Any]:
        """
        Run a browser automation task.
        
        Args:
            task_name: Name of the task to run
            
        Returns:
            Dictionary with task results
        """
        logger.info(f"Running browser task: {task_name}")
        
        if task_name not in self.config.get("tasks", {}):
            return {"success": False, "error": f"Task not found: {task_name}"}
            
        if not SELENIUM_AVAILABLE:
            return self._simulate_task(task_name)
            
        try:
            # Initialize WebDriver
            self._initialize_driver()
            
            if not self.driver:
                return {"success": False, "error": "Failed to initialize WebDriver"}
                
            # Get task configuration
            task_config = self.config["tasks"][task_name]
            websites = task_config.get("websites", [])
            
            results = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "task": task_name,
                "website_results": []
            }
            
            # Process each website in the task
            for website in websites:
                website_name = website.get("name", "Unknown")
                url = website.get("url")
                
                if not url:
                    logger.warning(f"No URL provided for website: {website_name}")
                    continue
                    
                logger.info(f"Processing website: {website_name} ({url})")
                
                # Navigate to the website
                try:
                    self.driver.get(url)
                    logger.info(f"Navigated to {url}")
                    
                    # Wait for the page to load
                    time.sleep(2)
                    
                    # Take a screenshot
                    screenshot = self.take_screenshot(f"{task_name}_{website_name.lower().replace(' ', '_')}.png")
                    
                    # Process according to task type
                    if task_name == "data_collection":
                        website_result = self._collect_data(website)
                    elif task_name == "test_solving":
                        website_result = self._perform_actions(website)
                    else:
                        website_result = {"success": False, "error": f"Unknown task type: {task_name}"}
                        
                    website_result["website"] = website_name
                    website_result["url"] = url
                    website_result["screenshot"] = screenshot
                    
                    results["website_results"].append(website_result)
                    
                except Exception as e:
                    logger.error(f"Error processing website {website_name}: {str(e)}")
                    results["website_results"].append({
                        "website": website_name,
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error running browser task {task_name}: {str(e)}")
            return {"success": False, "error": str(e)}
            
        finally:
            # Always close the driver when done
            self._close_driver()
            
    def _collect_data(self, website: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect data from a website.
        
        Args:
            website: Website configuration
            
        Returns:
            Dictionary with collected data
        """
        selectors = website.get("selectors", {})
        if not selectors:
            return {"success": False, "error": "No selectors provided for data collection"}
            
        result = {
            "success": True,
            "data": {}
        }
        
        try:
            # Process each selector
            for name, selector in selectors.items():
                try:
                    # Wait for the element to be present
                    elements = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    
                    # Extract text from elements
                    data = [elem.text for elem in elements if elem.text]
                    result["data"][name] = data
                    
                    logger.info(f"Collected {len(data)} items for selector '{name}'")
                    
                except (TimeoutException, NoSuchElementException) as e:
                    logger.warning(f"Error finding elements with selector '{selector}': {str(e)}")
                    result["data"][name] = []
                    
            return result
            
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _perform_actions(self, website: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a sequence of actions on a website.
        
        Args:
            website: Website configuration with actions
            
        Returns:
            Dictionary with action results
        """
        actions = website.get("actions", [])
        if not actions:
            return {"success": False, "error": "No actions provided"}
            
        result = {
            "success": True,
            "actions_performed": []
        }
        
        try:
            # Replace env var placeholders in action values
            env_vars = website.get("env_vars", [])
            action_env = {}
            
            for var in env_vars:
                action_env[var] = os.environ.get(var, "")
                
            # Process each action
            for action in actions:
                action_type = action.get("type")
                if not action_type:
                    continue
                    
                logger.info(f"Performing action: {action_type}")
                
                try:
                    if action_type == "wait":
                        # Wait for specified seconds
                        seconds = action.get("seconds", 1)
                        time.sleep(seconds)
                        action_result = {"type": "wait", "seconds": seconds, "success": True}
                        
                    elif action_type == "click":
                        # Click on an element
                        selector = action.get("selector")
                        element = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        element.click()
                        action_result = {"type": "click", "selector": selector, "success": True}
                        
                    elif action_type == "input":
                        # Input text into a field
                        selector = action.get("selector")
                        value = action.get("value", "")
                        
                        # Replace placeholders with environment variables
                        for var_name, var_value in action_env.items():
                            value = value.replace(f"{{{{{var_name}}}}}", var_value)
                            
                        element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        element.clear()
                        element.send_keys(value)
                        
                        # Mask sensitive values in logs
                        masked_value = "******" if "password" in selector.lower() else value
                        action_result = {"type": "input", "selector": selector, "value": masked_value, "success": True}
                        
                    else:
                        logger.warning(f"Unknown action type: {action_type}")
                        action_result = {"type": action_type, "success": False, "error": "Unknown action type"}
                        
                    result["actions_performed"].append(action_result)
                    
                except (TimeoutException, NoSuchElementException, ElementClickInterceptedException) as e:
                    logger.error(f"Error performing action {action_type}: {str(e)}")
                    result["actions_performed"].append({
                        "type": action_type,
                        "success": False,
                        "error": str(e)
                    })
                    
            return result
            
        except Exception as e:
            logger.error(f"Error performing actions: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _simulate_task(self, task_name: str) -> Dict[str, Any]:
        """
        Simulate a browser task when Selenium is not available.
        
        Args:
            task_name: Name of the task to simulate
            
        Returns:
            Simulated task results
        """
        logger.info(f"Simulating browser task: {task_name} (Selenium not available)")
        
        task_config = self.config["tasks"][task_name]
        websites = task_config.get("websites", [])
        
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "task": task_name,
            "website_results": [],
            "simulated": True
        }
        
        for website in websites:
            website_name = website.get("name", "Unknown")
            url = website.get("url", "")
            
            logger.info(f"Simulating task for website: {website_name} ({url})")
            
            website_result = {
                "website": website_name,
                "url": url,
                "success": True,
                "simulated": True
            }
            
            if task_name == "data_collection":
                selectors = website.get("selectors", {})
                simulated_data = {}
                
                for name in selectors.keys():
                    simulated_data[name] = [f"Simulated data for {name} {i}" for i in range(3)]
                    
                website_result["data"] = simulated_data
                
            elif task_name == "test_solving":
                actions = website.get("actions", [])
                website_result["actions_performed"] = [
                    {"type": action.get("type"), "success": True, "simulated": True}
                    for action in actions
                ]
                
            results["website_results"].append(website_result)
            
        return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and test the browser operator
    operator = BrowserOperator()
    result = operator.run_task("data_collection")
    print(json.dumps(result, indent=2))
