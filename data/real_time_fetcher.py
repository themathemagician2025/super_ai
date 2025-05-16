"""
Real-Time Data Fetcher Implementation

Handles fetching of:
- Forex rates
- Sports odds
- Stock prices
- Market events
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import json
from pathlib import Path
import pandas as pd
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class DataSourceConfig:
    """Configuration for data sources"""
    
    FOREX_API_URL = "https://api.exchangerate-api.com/v4/latest/{base_currency}"
    SPORTS_ODDS_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
    STOCK_API_URL = "https://api.marketdata.app/v1/stocks/{symbol}/quotes"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            'rate_limits': {
                'forex': 100,  # requests per minute
                'sports': 500,
                'stocks': 300
            },
            'retry_config': {
                'max_attempts': 3,
                'min_wait': 1,
                'max_wait': 10
            }
        }

class RealTimeDataFetcher:
    """Fetches real-time data from various sources"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = DataSourceConfig(config_path)
        self.session = None
        self.request_counts = {
            'forex': 0,
            'sports': 0,
            'stocks': 0
        }
        self.last_reset = {
            'forex': datetime.now(),
            'sports': datetime.now(),
            'stocks': datetime.now()
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _check_rate_limit(self, source: str) -> None:
        """Check and handle rate limiting"""
        now = datetime.now()
        if (now - self.last_reset[source]).seconds >= 60:
            self.request_counts[source] = 0
            self.last_reset[source] = now
            
        if self.request_counts[source] >= self.config.config['rate_limits'][source]:
            wait_time = 60 - (now - self.last_reset[source]).seconds
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_counts[source] = 0
            self.last_reset[source] = datetime.now()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_forex_rates(self, base_currency: str = 'USD') -> Dict:
        """Fetch current forex rates"""
        await self._check_rate_limit('forex')
        url = self.config.FOREX_API_URL.format(base_currency=base_currency)
        
        async with self.session.get(url) as response:
            self.request_counts['forex'] += 1
            if response.status != 200:
                logger.error(f"Failed to fetch forex rates: {response.status}")
                raise Exception(f"API request failed with status {response.status}")
            return await response.json()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_sports_odds(self, 
                              sport: str,
                              regions: List[str] = ['us'],
                              markets: List[str] = ['h2h', 'spreads']) -> Dict:
        """Fetch current sports odds"""
        await self._check_rate_limit('sports')
        url = self.config.SPORTS_ODDS_URL.format(sport=sport)
        params = {
            'regions': ','.join(regions),
            'markets': ','.join(markets)
        }
        
        async with self.session.get(url, params=params) as response:
            self.request_counts['sports'] += 1
            if response.status != 200:
                logger.error(f"Failed to fetch sports odds: {response.status}")
                raise Exception(f"API request failed with status {response.status}")
            return await response.json()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_stock_data(self, symbol: str) -> Dict:
        """Fetch current stock data"""
        await self._check_rate_limit('stocks')
        url = self.config.STOCK_API_URL.format(symbol=symbol)
        
        async with self.session.get(url) as response:
            self.request_counts['stocks'] += 1
            if response.status != 200:
                logger.error(f"Failed to fetch stock data: {response.status}")
                raise Exception(f"API request failed with status {response.status}")
            return await response.json()
            
    async def fetch_all_data(self,
                           forex_currencies: List[str],
                           sports_list: List[str],
                           stock_symbols: List[str]) -> Dict[str, Dict]:
        """Fetch all types of data concurrently"""
        tasks = []
        
        # Add forex tasks
        for currency in forex_currencies:
            tasks.append(self.fetch_forex_rates(currency))
            
        # Add sports tasks
        for sport in sports_list:
            tasks.append(self.fetch_sports_odds(sport))
            
        # Add stock tasks
        for symbol in stock_symbols:
            tasks.append(self.fetch_stock_data(symbol))
            
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data = {
            'forex': {},
            'sports': {},
            'stocks': {}
        }
        
        current_idx = 0
        
        # Process forex results
        for currency in forex_currencies:
            if isinstance(results[current_idx], Exception):
                logger.error(f"Failed to fetch forex data for {currency}: {results[current_idx]}")
            else:
                data['forex'][currency] = results[current_idx]
            current_idx += 1
            
        # Process sports results
        for sport in sports_list:
            if isinstance(results[current_idx], Exception):
                logger.error(f"Failed to fetch sports data for {sport}: {results[current_idx]}")
            else:
                data['sports'][sport] = results[current_idx]
            current_idx += 1
            
        # Process stock results
        for symbol in stock_symbols:
            if isinstance(results[current_idx], Exception):
                logger.error(f"Failed to fetch stock data for {symbol}: {results[current_idx]}")
            else:
                data['stocks'][symbol] = results[current_idx]
            current_idx += 1
            
        return data

class DataCache:
    """Caches fetched data with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
            del self.cache[key]
        return None
        
    def set(self, key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[key] = (data, datetime.now())
        
    def clear_expired(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl)
        ]
        for key in expired_keys:
            del self.cache[key]

# Example usage
async def main():
    async with RealTimeDataFetcher() as fetcher:
        # Fetch forex data
        forex_data = await fetcher.fetch_forex_rates('USD')
        print("Forex rates:", forex_data)
        
        # Fetch sports odds
        sports_data = await fetcher.fetch_sports_odds('soccer_epl')
        print("Sports odds:", sports_data)
        
        # Fetch stock data
        stock_data = await fetcher.fetch_stock_data('AAPL')
        print("Stock data:", stock_data)
        
        # Fetch all data concurrently
        all_data = await fetcher.fetch_all_data(
            forex_currencies=['USD', 'EUR', 'GBP'],
            sports_list=['soccer_epl', 'nba'],
            stock_symbols=['AAPL', 'GOOGL', 'MSFT']
        )
        print("All data:", all_data)

if __name__ == "__main__":
    asyncio.run(main()) 