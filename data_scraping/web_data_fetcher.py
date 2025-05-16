# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Web Data Fetcher

This module provides functionality to fetch financial and market data from various APIs
for use in the Super AI prediction system.
"""

import os
import json
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union

import aiohttp
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)

class WebDataFetcher:
    """Class for fetching data from various web APIs."""

    def __init__(self, output_dir: str = None, api_keys: Dict[str, str] = None):
        """
        Initialize the web data fetcher.

        Args:
            output_dir: Directory to save fetched data (defaults to data/fetched_data)
            api_keys: Dictionary of API keys for different services
        """
        self.output_dir = output_dir or os.path.join('data', 'fetched_data')
        os.makedirs(self.output_dir, exist_ok=True)

        self.api_keys = api_keys or {}
        self.session = None

        # Base URLs for different APIs
        self.api_urls = {
            'forex': 'https://api.apilayer.com/exchangerates_data',
            'stocks': 'https://www.alphavantage.co/query',
            'crypto': 'https://api.coingecko.com/api/v3'
        }

    async def create_session(self):
        """Create an aiohttp session for making requests."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_data(self, url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch data from an API.

        Args:
            url: The URL to fetch
            params: Query parameters
            headers: HTTP headers

        Returns:
            JSON response as a dictionary or None if fetch failed
        """
        try:
            session = await self.create_session()
            async with session.get(url, params=params, headers=headers, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def fetch_forex_data(self, base_currency: str = 'USD', symbols: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch forex exchange rate data.

        Args:
            base_currency: Base currency for exchange rates
            symbols: List of currency symbols to fetch

        Returns:
            Dictionary containing forex data
        """
        logger.info(f"Fetching forex data for {base_currency}")

        if 'exchange_rates' not in self.api_keys:
            logger.warning("No API key found for forex data. Using simulation mode.")
            return self._simulate_forex_data(base_currency, symbols)

        url = f"{self.api_urls['forex']}/latest"

        params = {
            'base': base_currency
        }

        if symbols:
            params['symbols'] = ','.join(symbols)

        headers = {
            'apikey': self.api_keys['exchange_rates']
        }

        data = await self.fetch_data(url, params, headers)

        if data:
            logger.info(f"Successfully fetched forex data for {base_currency}")

            # Save the data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"forex_{base_currency}_{timestamp}.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved forex data to {output_path}")

        return data

    def _simulate_forex_data(self, base_currency: str = 'USD', symbols: List[str] = None) -> Dict[str, Any]:
        """
        Simulate forex data for demo purposes.

        Args:
            base_currency: Base currency
            symbols: List of currency symbols

        Returns:
            Simulated forex data
        """
        import random

        # Default symbols if none provided
        if not symbols:
            symbols = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'CNY']

        # Remove base currency from symbols if present
        if base_currency in symbols:
            symbols.remove(base_currency)

        # Generate random rates
        rates = {}
        for symbol in symbols:
            # Use realistic ranges for major currencies
            if symbol == 'EUR':
                rates[symbol] = round(random.uniform(0.8, 1.2), 4)
            elif symbol == 'GBP':
                rates[symbol] = round(random.uniform(0.7, 0.9), 4)
            elif symbol == 'JPY':
                rates[symbol] = round(random.uniform(100, 150), 2)
            else:
                rates[symbol] = round(random.uniform(0.5, 2.0), 4)

        # Create response in the format similar to exchange rates API
        data = {
            'success': True,
            'timestamp': int(datetime.datetime.now().timestamp()),
            'base': base_currency,
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'rates': rates
        }

        return data

    async def fetch_stock_data(self, symbol: str, interval: str = 'daily') -> Optional[Dict[str, Any]]:
        """
        Fetch stock market data.

        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT)
            interval: Time interval (daily, weekly, monthly)

        Returns:
            Dictionary containing stock data
        """
        logger.info(f"Fetching stock data for {symbol}")

        if 'alpha_vantage' not in self.api_keys:
            logger.warning("No API key found for stock data. Using simulation mode.")
            return self._simulate_stock_data(symbol, interval)

        url = self.api_urls['stocks']

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_keys['alpha_vantage'],
            'outputsize': 'compact'
        }

        if interval == 'weekly':
            params['function'] = 'TIME_SERIES_WEEKLY'
        elif interval == 'monthly':
            params['function'] = 'TIME_SERIES_MONTHLY'

        data = await self.fetch_data(url, params)

        if data:
            logger.info(f"Successfully fetched stock data for {symbol}")

            # Save the data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"stock_{symbol}_{interval}_{timestamp}.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved stock data to {output_path}")

        return data

    def _simulate_stock_data(self, symbol: str, interval: str = 'daily') -> Dict[str, Any]:
        """
        Simulate stock data for demo purposes.

        Args:
            symbol: Stock symbol
            interval: Time interval

        Returns:
            Simulated stock data
        """
        import random

        # Generate time series based on interval
        time_series = {}

        today = datetime.datetime.now()

        # Set initial price based on popular stocks
        if symbol.upper() in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            base_price = random.uniform(100, 300)
        else:
            base_price = random.uniform(20, 200)

        # Generate data points
        num_points = 30 if interval == 'daily' else 12

        for i in range(num_points):
            if interval == 'daily':
                date = today - datetime.timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
            elif interval == 'weekly':
                date = today - datetime.timedelta(weeks=i)
                date_str = date.strftime('%Y-%m-%d')
            else:  # monthly
                date = today.replace(month=((today.month - i - 1) % 12) + 1)
                date_str = date.strftime('%Y-%m-%d')

            # Simulate price movement
            price_change = random.uniform(-0.05, 0.05)  # -5% to +5%
            current_price = base_price * (1 + price_change)

            # Add some volatility
            open_price = current_price * random.uniform(0.98, 1.02)
            high_price = current_price * random.uniform(1.01, 1.05)
            low_price = current_price * random.uniform(0.95, 0.99)

            time_series[date_str] = {
                '1. open': str(round(open_price, 2)),
                '2. high': str(round(high_price, 2)),
                '3. low': str(round(low_price, 2)),
                '4. close': str(round(current_price, 2)),
                '5. volume': str(random.randint(1000000, 10000000))
            }

            # Update base price for next iteration
            base_price = current_price

        # Create response in the format similar to Alpha Vantage API
        time_series_key = f"Time Series ({interval.capitalize()})"

        data = {
            'Meta Data': {
                '1. Information': f"{interval.capitalize()} Prices for {symbol}",
                '2. Symbol': symbol,
                '3. Last Refreshed': today.strftime('%Y-%m-%d'),
                '4. Output Size': 'Compact',
                '5. Time Zone': 'US/Eastern'
            },
            time_series_key: time_series
        }

        return data

    async def fetch_crypto_data(self, coin_ids: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch cryptocurrency data.

        Args:
            coin_ids: List of coin IDs (e.g., bitcoin, ethereum)

        Returns:
            Dictionary containing crypto data
        """
        logger.info(f"Fetching crypto data")

        # Default to top cryptocurrencies if none specified
        if not coin_ids:
            coin_ids = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana']

        # CoinGecko API doesn't require an API key for basic usage
        url = f"{self.api_urls['crypto']}/coins/markets"

        params = {
            'vs_currency': 'usd',
            'ids': ','.join(coin_ids),
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': 'false'
        }

        data = await self.fetch_data(url, params)

        if data:
            logger.info(f"Successfully fetched crypto data for {len(data)} coins")

            # Save the data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"crypto_data_{timestamp}.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved crypto data to {output_path}")
        else:
            # If API fetch failed, use simulation
            logger.warning("Failed to fetch real crypto data. Using simulation.")
            data = self._simulate_crypto_data(coin_ids)

            # Save the simulated data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"crypto_data_simulated_{timestamp}.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved simulated crypto data to {output_path}")

        return data

    def _simulate_crypto_data(self, coin_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Simulate cryptocurrency data for demo purposes.

        Args:
            coin_ids: List of coin IDs

        Returns:
            Simulated crypto data
        """
        import random

        # Define price ranges for known cryptocurrencies
        price_ranges = {
            'bitcoin': (15000, 30000),
            'ethereum': (1000, 3000),
            'ripple': (0.2, 1.0),
            'cardano': (0.25, 0.75),
            'solana': (20, 100),
            'dogecoin': (0.05, 0.15),
            'polkadot': (5, 15)
        }

        result = []

        for coin_id in coin_ids:
            # Get price range or use default
            price_range = price_ranges.get(coin_id, (1, 100))
            price = round(random.uniform(*price_range), 2)

            # Generate 24h change percentage
            price_change_24h = round(random.uniform(-10, 10), 2)

            # Calculate market cap (price * circulating supply)
            circulating_supply = random.randint(1000000, 1000000000)
            market_cap = price * circulating_supply

            coin_data = {
                'id': coin_id,
                'symbol': coin_id[:3].upper(),
                'name': coin_id.title(),
                'current_price': price,
                'market_cap': market_cap,
                'market_cap_rank': random.randint(1, 100),
                'fully_diluted_valuation': market_cap * 1.2,
                'total_volume': market_cap * random.uniform(0.05, 0.2),
                'high_24h': price * (1 + random.uniform(0, 0.1)),
                'low_24h': price * (1 - random.uniform(0, 0.1)),
                'price_change_24h': price * (price_change_24h / 100),
                'price_change_percentage_24h': price_change_24h,
                'circulating_supply': circulating_supply,
                'total_supply': circulating_supply * random.uniform(1, 2),
                'max_supply': circulating_supply * random.uniform(2, 4) if random.random() > 0.3 else None,
                'last_updated': datetime.datetime.now().isoformat()
            }

            result.append(coin_data)

        return result

    async def run_all_fetchers(self) -> Dict[str, Any]:
        """
        Run all data fetchers and collect the results.

        Returns:
            Dictionary with results from each fetcher
        """
        # Define tasks for different data types
        tasks = [
            self.fetch_forex_data('USD', ['EUR', 'GBP', 'JPY', 'CAD', 'AUD']),
            self.fetch_stock_data('AAPL', 'daily'),
            self.fetch_stock_data('MSFT', 'daily'),
            self.fetch_stock_data('AMZN', 'daily'),
            self.fetch_crypto_data(['bitcoin', 'ethereum', 'cardano', 'solana', 'dogecoin'])
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        data = {
            'forex': results[0],
            'stocks': {
                'AAPL': results[1],
                'MSFT': results[2],
                'AMZN': results[3]
            },
            'crypto': results[4]
        }

        # Save combined results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"market_data_{timestamp}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved combined market data to {output_path}")

        return data

    def process_for_model(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process fetched market data into a format suitable for model training.

        Args:
            data: Dictionary containing fetched data

        Returns:
            Processed data as a DataFrame
        """
        processed_data = []
        timestamp = datetime.datetime.now()

        # Process forex data
        if 'forex' in data and data['forex']:
            forex_data = data['forex'].get('rates', {})
            for currency, rate in forex_data.items():
                processed_data.append({
                    'timestamp': timestamp.isoformat(),
                    'data_type': 'forex',
                    'symbol': currency,
                    'value': rate,
                    'base_currency': data['forex'].get('base', 'USD')
                })

        # Process stock data
        if 'stocks' in data:
            for symbol, stock_data in data['stocks'].items():
                time_series_key = next((k for k in stock_data if k.startswith('Time Series')), None)
                if time_series_key and time_series_key in stock_data:
                    time_series = stock_data[time_series_key]
                    # Get the most recent data point
                    if time_series:
                        latest_date = max(time_series.keys())
                        latest_data = time_series[latest_date]

                        processed_data.append({
                            'timestamp': timestamp.isoformat(),
                            'data_type': 'stock',
                            'symbol': symbol,
                            'date': latest_date,
                            'open': float(latest_data.get('1. open', 0)),
                            'high': float(latest_data.get('2. high', 0)),
                            'low': float(latest_data.get('3. low', 0)),
                            'close': float(latest_data.get('4. close', 0)),
                            'volume': int(latest_data.get('5. volume', 0))
                        })

        # Process crypto data
        if 'crypto' in data and isinstance(data['crypto'], list):
            for coin in data['crypto']:
                processed_data.append({
                    'timestamp': timestamp.isoformat(),
                    'data_type': 'crypto',
                    'symbol': coin.get('symbol', '').upper(),
                    'name': coin.get('name', ''),
                    'price': coin.get('current_price', 0),
                    'market_cap': coin.get('market_cap', 0),
                    'volume': coin.get('total_volume', 0),
                    'change_24h': coin.get('price_change_percentage_24h', 0)
                })

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)

        # Save processed data
        processed_dir = os.path.join('data', 'processed_data')
        os.makedirs(processed_dir, exist_ok=True)

        output_path = os.path.join(processed_dir, f"processed_market_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(processed_data)} processed market data entries to {output_path}")

        return df

async def fetch_market_data(api_keys: Dict[str, str] = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Utility function to fetch market data.

    Args:
        api_keys: Dictionary of API keys
        output_dir: Directory to save data

    Returns:
        Dictionary with results from fetchers
    """
    fetcher = WebDataFetcher(output_dir, api_keys)
    try:
        data = await fetcher.run_all_fetchers()

        # Process data for model input
        df = fetcher.process_for_model(data)

        return data
    finally:
        await fetcher.close_session()

async def main():
    """Main function to demonstrate data fetcher usage."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Try to load API keys from environment variables
    api_keys = {
        'exchange_rates': os.environ.get('EXCHANGE_RATES_API_KEY'),
        'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY')
    }

    # Run data fetcher
    try:
        logger.info("Starting market data fetching")
        data = await fetch_market_data(api_keys)
        logger.info("Market data fetching completed successfully")
    except Exception as e:
        logger.error(f"Error in data fetching process: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
