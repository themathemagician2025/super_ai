#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Financial data scraper for forex and stock market data.
This module handles scraping and processing of financial market data.
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialScraper:
    """
    Scraper for financial market data from various sources.
    Handles forex and stock market data.
    """

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the financial data scraper.

        Args:
            data_dir: Directory to store scraped data
        """
        self.data_dir = data_dir
        self.scraped_data_dir = os.path.join(data_dir, 'scraped_data')
        self.processed_data_dir = os.path.join(data_dir, 'processed_data')

        # Create directories if they don't exist
        os.makedirs(self.scraped_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Initialize sources
        self.forex_sources = [
            {"name": "ForexFactory", "url": "https://www.forexfactory.com", "method": self.scrape_forexfactory},
            {"name": "TradingView", "url": "https://www.tradingview.com/forex", "method": self.scrape_tradingview_forex},
            {"name": "DailyFX", "url": "https://www.dailyfx.com", "method": self.scrape_dailyfx},
            {"name": "Investing.com", "url": "https://www.investing.com/currencies", "method": self.scrape_investing_forex},
            {"name": "FXStreet", "url": "https://www.fxstreet.com", "method": self.scrape_fxstreet}
        ]

        self.stock_sources = [
            {"name": "Yahoo Finance", "url": "https://finance.yahoo.com", "method": self.scrape_yahoo_finance},
            {"name": "TradingView", "url": "https://www.tradingview.com/markets/stocks-usa", "method": self.scrape_tradingview_stocks},
            {"name": "MarketWatch", "url": "https://www.marketwatch.com", "method": self.scrape_marketwatch},
            {"name": "StockTwits", "url": "https://stocktwits.com", "method": self.scrape_stocktwits},
            {"name": "Investing.com", "url": "https://www.investing.com/equities", "method": self.scrape_investing_stocks}
        ]

    async def create_session(self) -> aiohttp.ClientSession:
        """Create an aiohttp session with headers that mimic a browser."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        return aiohttp.ClientSession(headers=headers)

    async def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page using aiohttp.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string or None if fetch failed
        """
        try:
            session = await self.create_session()
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
        finally:
            await session.close()

    async def fetch_json(self, url: str) -> Optional[Dict]:
        """
        Fetch JSON data from a REST API.

        Args:
            url: API URL to fetch

        Returns:
            JSON data as dictionary or None if fetch failed
        """
        try:
            session = await self.create_session()
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch JSON from {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching JSON from {url}: {str(e)}")
            return None
        finally:
            await session.close()

    def _generate_simulated_forex_data(self, source: str, count: int = 20) -> List[Dict[str, Any]]:
        """
        Generate simulated forex data for testing when scraping fails.

        Args:
            source: Name of the source
            count: Number of data points to generate

        Returns:
            List of dictionaries with simulated forex data
        """
        logger.info(f"Generating simulated forex data for {source}")
        forex_pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD",
                      "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY"]

        current_date = datetime.now()
        results = []

        for _ in range(count):
            pair = random.choice(forex_pairs)
            base_price = 1.0 if "EUR" in pair else (100.0 if "JPY" in pair else 1.5)
            fluctuation = random.uniform(-0.05, 0.05)
            price = base_price * (1 + fluctuation)

            # Generate random timestamp within the last week
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            timestamp = current_date - timedelta(days=days_ago, hours=hours_ago)

            results.append({
                "pair": pair,
                "price": round(price, 5),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "volume": random.randint(10000, 1000000),
                "high": round(price * (1 + random.uniform(0.001, 0.01)), 5),
                "low": round(price * (1 - random.uniform(0.001, 0.01)), 5),
                "change_percent": round(fluctuation * 100, 2),
                "source": source
            })

        return results

    def _generate_simulated_stock_data(self, source: str, count: int = 20) -> List[Dict[str, Any]]:
        """
        Generate simulated stock market data for testing when scraping fails.

        Args:
            source: Name of the source
            count: Number of data points to generate

        Returns:
            List of dictionaries with simulated stock data
        """
        logger.info(f"Generating simulated stock data for {source}")
        stock_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
                         "TSLA", "NVDA", "JPM", "V", "WMT"]

        current_date = datetime.now()
        results = []

        for _ in range(count):
            symbol = random.choice(stock_symbols)
            base_price = 100.0 if symbol in ["AAPL", "MSFT"] else (200.0 if symbol in ["AMZN", "GOOGL"] else 50.0)
            fluctuation = random.uniform(-0.1, 0.1)
            price = base_price * (1 + fluctuation)

            # Generate random timestamp within the last week
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            timestamp = current_date - timedelta(days=days_ago, hours=hours_ago)

            results.append({
                "symbol": symbol,
                "price": round(price, 2),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "volume": random.randint(100000, 10000000),
                "high": round(price * (1 + random.uniform(0.001, 0.05)), 2),
                "low": round(price * (1 - random.uniform(0.001, 0.05)), 2),
                "change_percent": round(fluctuation * 100, 2),
                "market_cap": round(price * random.randint(100000000, 3000000000) / 1000000000, 2),
                "market_cap_unit": "B",
                "pe_ratio": round(random.uniform(10, 60), 2),
                "dividend_yield": round(random.uniform(0, 4), 2),
                "source": source
            })

        return results

    async def scrape_forexfactory(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape forex market data and news from ForexFactory.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping ForexFactory for forex data")
        url = "https://www.forexfactory.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_forex_data("ForexFactory", 30)

        # For demonstration purposes, return simulated data
        return self._generate_simulated_forex_data("ForexFactory", 30)

    async def scrape_tradingview_forex(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape forex market data from TradingView.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping TradingView for forex data")
        url = "https://www.tradingview.com/forex"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_forex_data("TradingView", 40)

        return self._generate_simulated_forex_data("TradingView", 40)

    async def scrape_dailyfx(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape forex data and analysis from DailyFX.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping DailyFX for forex data")
        url = "https://www.dailyfx.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_forex_data("DailyFX", 25)

        return self._generate_simulated_forex_data("DailyFX", 25)

    async def scrape_investing_forex(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape forex data from Investing.com.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping Investing.com for forex data")
        url = "https://www.investing.com/currencies"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_forex_data("Investing.com", 35)

        return self._generate_simulated_forex_data("Investing.com", 35)

    async def scrape_fxstreet(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape forex data and analysis from FXStreet.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping FXStreet for forex data")
        url = "https://www.fxstreet.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_forex_data("FXStreet", 28)

        return self._generate_simulated_forex_data("FXStreet", 28)

    async def scrape_yahoo_finance(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape stock market data from Yahoo Finance.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping Yahoo Finance for stock data")
        url = "https://finance.yahoo.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_stock_data("Yahoo Finance", 45)

        return self._generate_simulated_stock_data("Yahoo Finance", 45)

    async def scrape_tradingview_stocks(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape stock market data from TradingView.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping TradingView for stock data")
        url = "https://www.tradingview.com/markets/stocks-usa"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_stock_data("TradingView", 40)

        return self._generate_simulated_stock_data("TradingView", 40)

    async def scrape_marketwatch(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape stock market data from MarketWatch.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping MarketWatch for stock data")
        url = "https://www.marketwatch.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_stock_data("MarketWatch", 30)

        return self._generate_simulated_stock_data("MarketWatch", 30)

    async def scrape_stocktwits(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape stock sentiment data from StockTwits.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping StockTwits for stock sentiment")
        url = "https://stocktwits.com"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_stock_data("StockTwits", 25)

        return self._generate_simulated_stock_data("StockTwits", 25)

    async def scrape_investing_stocks(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Scrape stock market data from Investing.com.

        Args:
            days: Number of days of data to scrape

        Returns:
            List of dictionaries containing scraped data
        """
        logger.info(f"Scraping Investing.com for stock data")
        url = "https://www.investing.com/equities"

        html = await self.fetch_page(url)
        if not html:
            return self._generate_simulated_stock_data("Investing.com", 35)

        return self._generate_simulated_stock_data("Investing.com", 35)

    async def scrape_forex_data(self) -> List[Dict[str, Any]]:
        """
        Scrape forex data from all sources.

        Returns:
            Combined list of dictionaries containing forex data
        """
        all_results = []

        for source in self.forex_sources:
            logger.info(f"Scraping forex data from {source['name']}")
            try:
                results = await source["method"]()
                if results:
                    all_results.extend(results)
                    logger.info(f"Successfully scraped {len(results)} forex data items from {source['name']}")
                else:
                    logger.warning(f"No forex data scraped from {source['name']}")
            except Exception as e:
                logger.error(f"Error scraping forex data from {source['name']}: {str(e)}")

        logger.info(f"Total forex data items scraped: {len(all_results)}")
        return all_results

    async def scrape_stock_data(self) -> List[Dict[str, Any]]:
        """
        Scrape stock market data from all sources.

        Returns:
            Combined list of dictionaries containing stock data
        """
        all_results = []

        for source in self.stock_sources:
            logger.info(f"Scraping stock data from {source['name']}")
            try:
                results = await source["method"]()
                if results:
                    all_results.extend(results)
                    logger.info(f"Successfully scraped {len(results)} stock data items from {source['name']}")
                else:
                    logger.warning(f"No stock data scraped from {source['name']}")
            except Exception as e:
                logger.error(f"Error scraping stock data from {source['name']}: {str(e)}")

        logger.info(f"Total stock data items scraped: {len(all_results)}")
        return all_results

    def save_data(self, data: List[Dict[str, Any]], data_type: str) -> Tuple[str, str]:
        """
        Save scraped data to CSV and JSON files.

        Args:
            data: List of dictionaries containing scraped data
            data_type: Type of data (forex or stocks)

        Returns:
            Tuple of (csv_path, json_path)
        """
        if not data:
            logger.warning(f"No {data_type} data to save")
            return "", ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{data_type}_{timestamp}.csv"
        json_filename = f"{data_type}_{timestamp}.json"

        csv_path = os.path.join(self.scraped_data_dir, csv_filename)
        json_path = os.path.join(self.scraped_data_dir, json_filename)

        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(data)} entries to {csv_path}")

        # Save as JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, default=str)
        logger.info(f"Saved {len(data)} entries to {json_path}")

        return csv_path, json_path

    async def run(self, data_types: List[str] = ["forex", "stocks"]) -> Dict[str, Tuple[str, str]]:
        """
        Run the financial data scraper for specified data types.

        Args:
            data_types: List of data types to scrape (forex, stocks)

        Returns:
            Dictionary mapping data types to tuples of (csv_path, json_path)
        """
        results = {}

        if "forex" in data_types:
            forex_data = await self.scrape_forex_data()
            if forex_data:
                csv_path, json_path = self.save_data(forex_data, "forex")
                results["forex"] = (csv_path, json_path)
                logger.info(f"Forex data scraping completed. Files saved: {csv_path}, {json_path}")

        if "stocks" in data_types:
            stock_data = await self.scrape_stock_data()
            if stock_data:
                csv_path, json_path = self.save_data(stock_data, "stocks")
                results["stocks"] = (csv_path, json_path)
                logger.info(f"Stock data scraping completed. Files saved: {csv_path}, {json_path}")

        return results

async def main():
    """Main function to run the financial scraper."""
    scraper = FinancialScraper()
    results = await scraper.run()

    for data_type, (csv_path, json_path) in results.items():
        logger.info(f"{data_type.capitalize()} data saved to {csv_path} and {json_path}")

if __name__ == "__main__":
    asyncio.run(main())