"""
Create Sample Feature Data for Phase 2 Testing
=============================================

This script creates a sample feature file that can be used for testing Phase 2
when Hopsworks feature groups are not yet ready.

Author: Data Science Team
Date: August 12, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_features():
    """Create sample feature data for testing"""
    print("🔧 Creating sample feature data for Phase 2 testing...")
    
    # Create sample data
    n_samples = 100  # Enough for testing
    
    # Generate timestamps (last 100 hours)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=100)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)
    
    # Create sample features
    np.random.seed(42)  # For reproducible results
    
    data = {
        'timestamp': timestamps,
        'aqi_numeric': np.random.randint(50, 200, n_samples),  # Target variable
        'pm2_5': np.random.uniform(10, 80, n_samples),
        'pm10': np.random.uniform(20, 150, n_samples),
        'no2': np.random.uniform(5, 60, n_samples),
        'o3': np.random.uniform(20, 100, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'pressure': np.random.uniform(1000, 1020, n_samples),
        'wind_speed': np.random.uniform(0, 20, n_samples),
        'hour_sin': np.sin(2 * np.pi * timestamps.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamps.hour / 24),
        'day_sin': np.sin(2 * np.pi * timestamps.dayofyear / 365),
        'day_cos': np.cos(2 * np.pi * timestamps.dayofyear / 365),
        'is_weekend': timestamps.dayofweek.isin([5, 6]).astype(int),
        'is_morning_rush': ((timestamps.hour >= 7) & (timestamps.hour <= 9)).astype(int),
        'is_evening_rush': ((timestamps.hour >= 17) & (timestamps.hour <= 19)).astype(int),
    }
    
    # Create DataFrame first
    df = pd.DataFrame(data)
    
    # Add lag features
    for lag in [1, 3, 6, 12, 24]:
        df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
        df[f'aqi_numeric_lag_{lag}h'] = df['aqi_numeric'].shift(lag)
    
    # Add rolling features
    for window in [3, 6, 12, 24]:
        df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window=window, min_periods=1).mean()
        df[f'pm2_5_rolling_std_{window}h'] = df['pm2_5'].rolling(window=window, min_periods=1).std()
        df[f'aqi_rolling_mean_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).mean()
    
    # Add advanced features
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    df['temperature_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['aqi_change_1h'] = df['aqi_numeric'].diff()
    df['pm2_5_change_3h'] = df['pm2_5'].diff(3)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Ensure directory exists
    os.makedirs('data_repositories/features/engineered', exist_ok=True)
    
    # Save sample features
    output_file = 'data_repositories/features/engineered/realtime_features.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample features created: {output_file}")
    print(f"📊 Records: {len(df)}")
    print(f"🔢 Features: {len(df.columns)}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

if __name__ == "__main__":
    create_sample_features()
