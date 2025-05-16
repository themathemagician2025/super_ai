"""
Advanced Data Preprocessing Module

Features:
- Real-time data fetching and integration
- Automatic normalization detection
- Intelligent missing data imputation
- Dynamic outlier removal
- Contextual time feature generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from datetime import datetime, timedelta
import aiohttp
import asyncio
import logging
from dataclasses import dataclass
import pytz
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class DataStats:
    """Statistics about the dataset"""
    skewness: float
    kurtosis: float
    missing_ratio: float
    outlier_ratio: float
    value_range: Tuple[float, float]
    distribution_type: str

class DataNormalizer:
    """Automatically selects and applies the best normalization method"""
    
    def __init__(self):
        self.scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.selected_scaler = None
        self.stats = None
        
    def analyze_distribution(self, data: np.ndarray) -> DataStats:
        """Analyze data distribution to select best normalization"""
        stats = DataStats(
            skewness=float(stats.skew(data, nan_policy='omit')),
            kurtosis=float(stats.kurtosis(data, nan_policy='omit')),
            missing_ratio=np.isnan(data).mean(),
            outlier_ratio=self._calculate_outlier_ratio(data),
            value_range=(float(np.nanmin(data)), float(np.nanmax(data))),
            distribution_type=self._determine_distribution(data)
        )
        self.stats = stats
        return stats
        
    def _determine_distribution(self, data: np.ndarray) -> str:
        """Determine the type of distribution"""
        _, p_value = stats.normaltest(data[~np.isnan(data)])
        if p_value > 0.05:
            return 'normal'
        elif np.abs(stats.skew(data, nan_policy='omit')) > 1:
            return 'skewed'
        else:
            return 'unknown'
            
    def _calculate_outlier_ratio(self, data: np.ndarray) -> float:
        """Calculate ratio of outliers using IQR method"""
        q1, q3 = np.nanpercentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
        return float(np.sum(outliers)) / len(data)
        
    def select_best_scaler(self, data: np.ndarray) -> str:
        """Select best scaler based on data characteristics"""
        stats = self.analyze_distribution(data)
        
        if stats.distribution_type == 'normal':
            selected = 'standard'
        elif stats.outlier_ratio > 0.1:
            selected = 'robust'
        else:
            selected = 'minmax'
            
        self.selected_scaler = self.scalers[selected]
        return selected
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data using best scaler"""
        if self.selected_scaler is None:
            self.select_best_scaler(data)
        return self.selected_scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if self.selected_scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        return self.selected_scaler.transform(data.reshape(-1, 1)).ravel()

class MissingDataImputer:
    """Intelligent missing data imputation"""
    
    def __init__(self):
        self.imputers = {
            'knn': KNNImputer(n_neighbors=5),
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent')
        }
        self.selected_imputer = None
        
    def select_best_imputer(self, data: np.ndarray) -> str:
        """Select best imputation method based on data characteristics"""
        missing_ratio = np.isnan(data).mean()
        
        if missing_ratio < 0.1:
            selected = 'knn'  # Use KNN for low missing ratios
        elif np.abs(stats.skew(data[~np.isnan(data)])) > 1:
            selected = 'median'  # Use median for skewed data
        else:
            selected = 'mean'  # Use mean for roughly normal data
            
        self.selected_imputer = self.imputers[selected]
        return selected
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data using best imputer"""
        if self.selected_imputer is None:
            self.select_best_imputer(data)
        return self.selected_imputer.fit_transform(data.reshape(-1, 1)).ravel()

class OutlierRemover:
    """Dynamic outlier removal with adaptive thresholds"""
    
    def __init__(self):
        self.threshold_multiplier = 1.5
        self.min_threshold = 0.1
        self.max_threshold = 0.9
        
    def calculate_bounds(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate outlier bounds using IQR method"""
        q1, q3 = np.nanpercentile(data, [25, 75])
        iqr = q3 - q1
        
        # Adjust threshold based on data spread
        spread_ratio = iqr / (np.nanmax(data) - np.nanmin(data))
        adjusted_multiplier = self.threshold_multiplier * (1 + spread_ratio)
        
        lower_bound = q1 - adjusted_multiplier * iqr
        upper_bound = q3 + adjusted_multiplier * iqr
        
        return float(lower_bound), float(upper_bound)
        
    def remove_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers and return cleaned data and outlier mask"""
        lower_bound, upper_bound = self.calculate_bounds(data)
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        cleaned_data = np.where(outlier_mask, np.nan, data)
        return cleaned_data, outlier_mask

class TimeFeatureGenerator:
    """Generate contextual time-based features"""
    
    def __init__(self):
        self.timezone = pytz.UTC
        self.event_markers = {}  # Dict to store special events
        
    def set_timezone(self, timezone_str: str):
        """Set timezone for feature generation"""
        self.timezone = pytz.timezone(timezone_str)
        
    def add_event_marker(self, event_name: str, start_time: datetime, end_time: datetime):
        """Add special event timeframe"""
        self.event_markers[event_name] = (start_time, end_time)
        
    def generate_features(self, timestamps: List[datetime]) -> pd.DataFrame:
        """Generate time-based features from timestamps"""
        features = []
        
        for ts in timestamps:
            if not ts.tzinfo:
                ts = pytz.UTC.localize(ts)
            ts = ts.astimezone(self.timezone)
            
            feature_dict = {
                'hour': ts.hour / 24.0,  # Normalized hour
                'day_of_week': ts.weekday() / 6.0,  # Normalized day of week
                'day_of_month': (ts.day - 1) / 30.0,  # Normalized day of month
                'month': (ts.month - 1) / 11.0,  # Normalized month
                'is_weekend': float(ts.weekday() >= 5),
                'is_business_hour': float(9 <= ts.hour <= 17),
                'quarter_of_day': ts.hour // 6,
            }
            
            # Add event markers
            for event_name, (start, end) in self.event_markers.items():
                feature_dict[f'is_{event_name}'] = float(start <= ts <= end)
                
            features.append(feature_dict)
            
        return pd.DataFrame(features)

class RealTimeDataFetcher:
    """Asynchronous real-time data fetching"""
    
    def __init__(self):
        self.session = None
        self.rate_limit = 100  # requests per minute
        self.request_count = 0
        self.last_reset = datetime.now()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_data(self, url: str) -> Dict:
        """Fetch data with rate limiting"""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.request_count = 0
            self.last_reset = now
            
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (now - self.last_reset).seconds
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_reset = datetime.now()
            
        async with self.session.get(url) as response:
            self.request_count += 1
            return await response.json()

class DataPreprocessor:
    """Main preprocessing pipeline"""
    
    def __init__(self):
        self.normalizer = DataNormalizer()
        self.imputer = MissingDataImputer()
        self.outlier_remover = OutlierRemover()
        self.time_feature_generator = TimeFeatureGenerator()
        
    def preprocess(self, 
                  data: np.ndarray,
                  timestamps: Optional[List[datetime]] = None) -> Tuple[np.ndarray, Dict]:
        """Run full preprocessing pipeline"""
        # Remove outliers
        cleaned_data, outlier_mask = self.outlier_remover.remove_outliers(data)
        
        # Impute missing values
        imputed_data = self.imputer.fit_transform(cleaned_data)
        
        # Normalize data
        normalized_data = self.normalizer.fit_transform(imputed_data)
        
        # Generate time features if timestamps provided
        time_features = None
        if timestamps:
            time_features = self.time_feature_generator.generate_features(timestamps)
            
        # Collect statistics
        stats = {
            'original_shape': data.shape,
            'outliers_removed': int(outlier_mask.sum()),
            'missing_values': int(np.isnan(data).sum()),
            'normalization_method': self.normalizer.selected_scaler.__class__.__name__,
            'imputation_method': self.imputer.selected_imputer.__class__.__name__,
            'data_stats': self.normalizer.stats.__dict__ if self.normalizer.stats else None
        }
        
        return normalized_data, stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = np.random.randn(1000)
    data[::10] = np.nan  # Add some missing values
    data[::20] = data[::20] * 10  # Add some outliers
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(data))]
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Add some event markers
    event_start = datetime.now()
    event_end = event_start + timedelta(days=1)
    preprocessor.time_feature_generator.add_event_marker('special_event', event_start, event_end)
    
    # Process data
    processed_data, stats = preprocessor.preprocess(data, timestamps)
    print("Preprocessing stats:", stats) 