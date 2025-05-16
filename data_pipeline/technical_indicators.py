# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import talib
from scipy import stats


class TechnicalIndicators:
    """Class for calculating technical indicators for financial time series data."""

    def __init__(self):
        """Initialize the technical indicators calculator."""
        # Define indicator categories
        self.trend_indicators = ['sma', 'ema', 'macd', 'adx', 'ichimoku']
        self.momentum_indicators = ['rsi', 'stoch', 'cci', 'williams_r', 'mfi']
        self.volatility_indicators = ['atr', 'bollinger', 'keltner', 'donchian']
        self.volume_indicators = ['obv', 'ad', 'cmf', 'vwap']

        # All available indicators
        self.all_indicators = (self.trend_indicators + self.momentum_indicators +
                              self.volatility_indicators + self.volume_indicators)

    def add_all_indicators(self, df, include=None, exclude=None):
        """
        Add all or selected technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data
            include: List of indicators to include (if None, include all)
            exclude: List of indicators to exclude (if None, exclude none)

        Returns:
            DataFrame with added indicators
        """
        # Make a copy of the dataframe to avoid modifying the original
        result = df.copy()

        # Determine which indicators to add
        if include is not None:
            indicators_to_add = [ind for ind in include if ind in self.all_indicators]
        else:
            indicators_to_add = self.all_indicators.copy()

        if exclude is not None:
            indicators_to_add = [ind for ind in indicators_to_add if ind not in exclude]

        # Add selected indicators
        for indicator in indicators_to_add:
            method_name = f"add_{indicator}"
            if hasattr(self, method_name):
                result = getattr(self, method_name)(result)

        return result

    def add_sma(self, df, periods=[5, 10, 20, 50, 200]):
        """Add Simple Moving Averages."""
        result = df.copy()
        for period in periods:
            result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
        return result

    def add_ema(self, df, periods=[5, 10, 20, 50, 200]):
        """Add Exponential Moving Averages."""
        result = df.copy()
        for period in periods:
            result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        return result

    def add_macd(self, df, fast=12, slow=26, signal=9):
        """Add MACD (Moving Average Convergence Divergence)."""
        result = df.copy()

        # Calculate MACD components
        fast_ema = result['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = result['close'].ewm(span=slow, adjust=False).mean()

        result['macd'] = fast_ema - slow_ema
        result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']

        return result

    def add_adx(self, df, period=14):
        """Add ADX (Average Directional Index)."""
        result = df.copy()

        # Calculate +DI and -DI
        high = result['high'].values
        low = result['low'].values
        close = result['close'].values

        try:
            # Use talib for ADX calculation if available
            result['adx'] = talib.ADX(high, low, close, timeperiod=period)
            result['+di'] = talib.PLUS_DI(high, low, close, timeperiod=period)
            result['-di'] = talib.MINUS_DI(high, low, close, timeperiod=period)
        except:
            # Fallback calculation if talib is not available
            # This is a simplified version - not as accurate as talib
            result['tr'] = np.maximum(
                result['high'] - result['low'],
                np.maximum(
                    abs(result['high'] - result['close'].shift(1)),
                    abs(result['low'] - result['close'].shift(1))
                )
            )

            result['+dm'] = np.where(
                (result['high'] - result['high'].shift(1)) > (result['low'].shift(1) - result['low']),
                np.maximum(result['high'] - result['high'].shift(1), 0),
                0
            )

            result['-dm'] = np.where(
                (result['low'].shift(1) - result['low']) > (result['high'] - result['high'].shift(1)),
                np.maximum(result['low'].shift(1) - result['low'], 0),
                0
            )

            result['+di'] = 100 * (result['+dm'].rolling(period).sum() / result['tr'].rolling(period).sum())
            result['-di'] = 100 * (result['-dm'].rolling(period).sum() / result['tr'].rolling(period).sum())
            result['dx'] = 100 * abs(result['+di'] - result['-di']) / (result['+di'] + result['-di'])
            result['adx'] = result['dx'].rolling(period).mean()

            # Clean up intermediate columns
            result.drop(['tr', '+dm', '-dm', 'dx'], axis=1, inplace=True)

        return result

    def add_ichimoku(self, df, tenkan=9, kijun=26, senkou_b=52):
        """Add Ichimoku Cloud indicators."""
        result = df.copy()

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period_high = result['high'].rolling(window=tenkan).max()
        period_low = result['low'].rolling(window=tenkan).min()
        result['tenkan_sen'] = (period_high + period_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period_high = result['high'].rolling(window=kijun).max()
        period_low = result['low'].rolling(window=kijun).min()
        result['kijun_sen'] = (period_high + period_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 (26 periods ahead)
        result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(kijun)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 (26 periods ahead)
        period_high = result['high'].rolling(window=senkou_b).max()
        period_low = result['low'].rolling(window=senkou_b).min()
        result['senkou_span_b'] = ((period_high + period_low) / 2).shift(kijun)

        # Chikou Span (Lagging Span): Close price shifted backwards 26 periods
        result['chikou_span'] = result['close'].shift(-kijun)

        return result

    def add_rsi(self, df, period=14):
        """Add RSI (Relative Strength Index)."""
        result = df.copy()

        # Calculate RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # Add RSI bands
        result['rsi_oversold'] = 30
        result['rsi_overbought'] = 70

        return result

    def add_stoch(self, df, k_period=14, d_period=3, slowing=3):
        """Add Stochastic Oscillator."""
        result = df.copy()

        try:
            # Use talib for Stochastic calculation if available
            result['stoch_k'], result['stoch_d'] = talib.STOCH(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                fastk_period=k_period,
                slowk_period=slowing,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
        except:
            # Fallback calculation if talib is not available
            # Calculate %K
            low_min = result['low'].rolling(window=k_period).min()
            high_max = result['high'].rolling(window=k_period).max()

            result['stoch_k'] = 100 * ((result['close'] - low_min) / (high_max - low_min))
            result['stoch_k'] = result['stoch_k'].rolling(window=slowing).mean()

            # Calculate %D
            result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()

        return result

    def add_cci(self, df, period=20):
        """Add CCI (Commodity Channel Index)."""
        result = df.copy()

        try:
            # Use talib for CCI calculation if available
            result['cci'] = talib.CCI(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
        except:
            # Fallback calculation if talib is not available
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            mean_deviation = np.abs(typical_price - typical_price.rolling(window=period).mean()).rolling(window=period).mean()
            result['cci'] = (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * mean_deviation)

        return result

    def add_williams_r(self, df, period=14):
        """Add Williams %R."""
        result = df.copy()

        try:
            # Use talib for Williams %R calculation if available
            result['williams_r'] = talib.WILLR(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
        except:
            # Fallback calculation if talib is not available
            highest_high = result['high'].rolling(window=period).max()
            lowest_low = result['low'].rolling(window=period).min()

            result['williams_r'] = -100 * (highest_high - result['close']) / (highest_high - lowest_low)

        return result

    def add_mfi(self, df, period=14):
        """Add MFI (Money Flow Index)."""
        result = df.copy()

        try:
            # Use talib for MFI calculation if available
            result['mfi'] = talib.MFI(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                result['volume'].values,
                timeperiod=period
            )
        except:
            # Fallback calculation if talib is not available
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            money_flow = typical_price * result['volume']

            # Get positive and negative money flow
            delta = typical_price.diff()
            positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
            negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

            # Calculate money flow ratio and MFI
            money_ratio = positive_flow / negative_flow
            result['mfi'] = 100 - (100 / (1 + money_ratio))

        return result

    def add_atr(self, df, period=14):
        """Add ATR (Average True Range)."""
        result = df.copy()

        try:
            # Use talib for ATR calculation if available
            result['atr'] = talib.ATR(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
        except:
            # Fallback calculation if talib is not available
            # Calculate true range
            result['tr'] = np.maximum(
                result['high'] - result['low'],
                np.maximum(
                    abs(result['high'] - result['close'].shift(1)),
                    abs(result['low'] - result['close'].shift(1))
                )
            )

            # Calculate ATR
            result['atr'] = result['tr'].rolling(window=period).mean()
            result.drop(['tr'], axis=1, inplace=True)

        return result

    def add_bollinger(self, df, period=20, std_dev=2):
        """Add Bollinger Bands."""
        result = df.copy()

        # Calculate Bollinger Bands
        result['bollinger_middle'] = result['close'].rolling(window=period).mean()
        result['bollinger_std'] = result['close'].rolling(window=period).std()

        result['bollinger_upper'] = result['bollinger_middle'] + (result['bollinger_std'] * std_dev)
        result['bollinger_lower'] = result['bollinger_middle'] - (result['bollinger_std'] * std_dev)

        # Calculate bandwidth and %B
        result['bollinger_bandwidth'] = (result['bollinger_upper'] - result['bollinger_lower']) / result['bollinger_middle']
        result['bollinger_b'] = (result['close'] - result['bollinger_lower']) / (result['bollinger_upper'] - result['bollinger_lower'])

        result.drop(['bollinger_std'], axis=1, inplace=True)

        return result

    def add_keltner(self, df, period=20, atr_period=10, multiplier=2):
        """Add Keltner Channels."""
        result = df.copy()

        # Calculate ATR if not already present
        if 'atr' not in result.columns:
            result = self.add_atr(result, period=atr_period)

        # Calculate Keltner Channels
        result['keltner_middle'] = result['close'].rolling(window=period).mean()
        result['keltner_upper'] = result['keltner_middle'] + (result['atr'] * multiplier)
        result['keltner_lower'] = result['keltner_middle'] - (result['atr'] * multiplier)

        return result

    def add_donchian(self, df, period=20):
        """Add Donchian Channels."""
        result = df.copy()

        # Calculate Donchian Channels
        result['donchian_upper'] = result['high'].rolling(window=period).max()
        result['donchian_lower'] = result['low'].rolling(window=period).min()
        result['donchian_middle'] = (result['donchian_upper'] + result['donchian_lower']) / 2

        return result

    def add_obv(self, df):
        """Add OBV (On-Balance Volume)."""
        result = df.copy()

        try:
            # Use talib for OBV calculation if available
            result['obv'] = talib.OBV(result['close'].values, result['volume'].values)
        except:
            # Fallback calculation if talib is not available
            obv = [0]
            for i in range(1, len(result)):
                if result['close'].iloc[i] > result['close'].iloc[i-1]:
                    obv.append(obv[-1] + result['volume'].iloc[i])
                elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                    obv.append(obv[-1] - result['volume'].iloc[i])
                else:
                    obv.append(obv[-1])

            result['obv'] = obv

        return result

    def add_ad(self, df):
        """Add A/D (Accumulation/Distribution Line)."""
        result = df.copy()

        try:
            # Use talib for A/D calculation if available
            result['ad'] = talib.AD(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                result['volume'].values
            )
        except:
            # Fallback calculation if talib is not available
            # Calculate money flow multiplier
            mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)
            mfm = mfm.fillna(0)

            # Calculate money flow volume
            mfv = mfm * result['volume']

            # Calculate A/D line
            result['ad'] = mfv.cumsum()

        return result

    def add_cmf(self, df, period=20):
        """Add CMF (Chaikin Money Flow)."""
        result = df.copy()

        # Calculate money flow multiplier
        mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfm = mfm.fillna(0)

        # Calculate money flow volume
        mfv = mfm * result['volume']

        # Calculate CMF
        result['cmf'] = mfv.rolling(window=period).sum() / result['volume'].rolling(window=period).sum()

        return result

    def add_vwap(self, df):
        """Add VWAP (Volume-Weighted Average Price)."""
        result = df.copy()

        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3

        # Calculate TP * volume
        result['tp_vol'] = result['typical_price'] * result['volume']

        # Calculate cumulative values
        result['cum_tp_vol'] = result['tp_vol'].cumsum()
        result['cum_vol'] = result['volume'].cumsum()

        # Calculate VWAP
        result['vwap'] = result['cum_tp_vol'] / result['cum_vol']

        # Clean up intermediate columns
        result.drop(['typical_price', 'tp_vol', 'cum_tp_vol', 'cum_vol'], axis=1, inplace=True)

        return result

    def add_custom_indicators(self, df):
        """Add custom indicators and trend strength metrics."""
        result = df.copy()

        # Ensure required indicators are present
        if 'sma_20' not in result.columns:
            result = self.add_sma(result)
        if 'ema_20' not in result.columns:
            result = self.add_ema(result)
        if 'bollinger_upper' not in result.columns:
            result = self.add_bollinger(result)
        if 'rsi' not in result.columns:
            result = self.add_rsi(result)

        # Price position relative to moving averages
        result['price_to_sma_20'] = result['close'] / result['sma_20'] - 1
        result['price_to_sma_50'] = result['close'] / result['sma_50'] - 1

        # Moving average crossovers
        result['sma_5_crossover_20'] = np.where(
            result['sma_5'] > result['sma_20'], 1,
            np.where(result['sma_5'] < result['sma_20'], -1, 0)
        )

        # Trend strength based on consecutive closes
        for period in [3, 5, 10]:
            # Count consecutive up/down days
            direction = np.sign(result['close'].diff())
            result[f'consec_moves_{period}'] = direction.rolling(period).sum()

        # Volatility ratio
        result['volatility_ratio'] = result['bollinger_bandwidth'] / result['bollinger_bandwidth'].rolling(window=20).mean()

        # Combined momentum indicator
        if all(x in result.columns for x in ['rsi', 'williams_r', 'stoch_k']):
            result['momentum_combined'] = (
                (result['rsi'] / 100) +
                ((result['williams_r'] + 100) / 100) +
                (result['stoch_k'] / 100)
            ) / 3

        return result


if __name__ == "__main__":
    # Example usage
    from forex_data_loader import ForexDataLoader

    # Load sample data
    loader = ForexDataLoader()
    data = loader.load_data("EUR/USD", "1h", start_date="2022-01-01", end_date="2022-01-31")

    # Calculate indicators
    indicators = TechnicalIndicators()

    # Add all indicators
    result = indicators.add_all_indicators(data)

    # Print info about the resulting dataset
    print(f"Original columns: {data.columns.tolist()}")
    print(f"Columns after adding indicators: {result.columns.tolist()}")
    print(f"Added {len(result.columns) - len(data.columns)} indicators")
