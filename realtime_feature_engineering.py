"""
Real-time Feature Engineering Pipeline
=====================================

This script processes hourly collected data and applies the same feature engineering
pipeline that was used to create the 215-feature dataset, ensuring consistency
between training and real-time prediction features.

Features:
- Applies identical feature engineering to new data
- Maintains feature consistency with training dataset
- Handles incremental feature computation
- Validates feature quality in real-time
- Prepares features for Hopsworks integration

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class RealTimeFeatureEngineer:
    """Real-time feature engineering pipeline for AQI prediction"""
    
    def __init__(self):
        """Initialize real-time feature engineer"""
        logger.info("🔧 REAL-TIME FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 50)
        
        # Setup directories
        self.data_dir = os.path.join("data_repositories", "merged_data", "processed")
        self.feature_dir = os.path.join("data_repositories", "features", "engineered")
        self.metadata_dir = os.path.join("data_repositories", "features", "metadata")
        
        # Create directories
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Feature engineering configuration (matches training pipeline)
        self.lag_hours = [1, 3, 6, 12, 24, 48, 72]
        self.rolling_windows = [3, 6, 12, 24]
        self.weather_vars = ['temperature', 'relative_humidity', 'wind_speed', 'pressure']
        self.pollution_vars = ['pm2_5', 'pm10', 'no2', 'o3', 'aqi_numeric']
        
        self.feature_metadata = {
            "pipeline_version": "2.0",
            "feature_count_target": 215,
            "categories": {
                "base_features": [],
                "lag_features": [],
                "rolling_features": [],
                "temporal_features": [],
                "advanced_features": []
            }
        }
        
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Feature output: {self.feature_dir}")

    def load_latest_data(self) -> pd.DataFrame:
        """Load the latest merged dataset"""
        logger.info("\n📊 STEP 1: Loading Latest Data")
        logger.info("-" * 35)
        
        try:
            merged_file = os.path.join(self.data_dir, "merged_data.csv")
            
            if not os.path.exists(merged_file):
                raise FileNotFoundError(f"Merged data file not found: {merged_file}")
            
            df = pd.read_csv(merged_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Data validation
            required_columns = ['timestamp', 'aqi_numeric'] + self.weather_vars + self.pollution_vars
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"✅ Data loaded: {len(df)} records")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"🔢 Columns: {len(df.columns)}")
            
            # Check data freshness
            latest_timestamp = df['timestamp'].max()
            hours_since_latest = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            if hours_since_latest > 3:
                logger.warning(f"⚠️ Data is {hours_since_latest:.1f} hours old")
            else:
                logger.info(f"✅ Data is fresh: {hours_since_latest:.1f} hours old")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features (matches training pipeline)"""
        logger.info("\n🕐 STEP 2: Creating Temporal Features")
        logger.info("-" * 40)
        
        try:
            df = df.copy()
            
            # Basic temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            
            # Advanced temporal features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Rush hour indicators
            df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
            df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            
            temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                               'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                               'is_morning_rush', 'is_evening_rush', 'is_night']
            
            self.feature_metadata["categories"]["temporal_features"] = temporal_features
            
            logger.info(f"✅ Created {len(temporal_features)} temporal features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating temporal features: {str(e)}")
            raise

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for key variables (matches training pipeline)"""
        logger.info("\n⏰ STEP 3: Creating Lag Features")
        logger.info("-" * 32)
        
        try:
            df = df.copy()
            lag_features = []
            
            # Lag features for pollution variables
            for var in self.pollution_vars:
                if var in df.columns:
                    for lag_hours in self.lag_hours:
                        lag_col = f"{var}_lag{lag_hours}h"
                        df[lag_col] = df[var].shift(lag_hours)
                        lag_features.append(lag_col)
            
            # Lag features for weather variables
            for var in self.weather_vars:
                if var in df.columns:
                    for lag_hours in [1, 6, 24]:  # Less frequent for weather
                        lag_col = f"{var}_lag{lag_hours}h"
                        df[lag_col] = df[var].shift(lag_hours)
                        lag_features.append(lag_col)
            
            self.feature_metadata["categories"]["lag_features"] = lag_features
            
            logger.info(f"✅ Created {len(lag_features)} lag features")
            logger.info(f"📊 Variables: {len(self.pollution_vars)} pollution + {len(self.weather_vars)} weather")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating lag features: {str(e)}")
            raise

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features (matches training pipeline)"""
        logger.info("\n📊 STEP 4: Creating Rolling Features")
        logger.info("-" * 35)
        
        try:
            df = df.copy()
            rolling_features = []
            
            # Rolling statistics for pollution variables
            for var in self.pollution_vars:
                if var in df.columns:
                    for window in self.rolling_windows:
                        # Rolling mean
                        mean_col = f"{var}_rolling{window}h_mean"
                        df[mean_col] = df[var].rolling(window=window, min_periods=1).mean()
                        rolling_features.append(mean_col)
                        
                        # Rolling std (volatility)
                        std_col = f"{var}_rolling{window}h_std"
                        df[std_col] = df[var].rolling(window=window, min_periods=1).std().fillna(0)
                        rolling_features.append(std_col)
                        
                        # Rolling min/max
                        if window >= 6:  # Only for larger windows
                            min_col = f"{var}_rolling{window}h_min"
                            max_col = f"{var}_rolling{window}h_max"
                            df[min_col] = df[var].rolling(window=window, min_periods=1).min()
                            df[max_col] = df[var].rolling(window=window, min_periods=1).max()
                            rolling_features.extend([min_col, max_col])
            
            # Rolling statistics for weather variables
            for var in self.weather_vars:
                if var in df.columns:
                    for window in [6, 24]:  # Less frequent for weather
                        mean_col = f"{var}_rolling{window}h_mean"
                        df[mean_col] = df[var].rolling(window=window, min_periods=1).mean()
                        rolling_features.append(mean_col)
            
            self.feature_metadata["categories"]["rolling_features"] = rolling_features
            
            logger.info(f"✅ Created {len(rolling_features)} rolling features")
            logger.info(f"📈 Windows: {self.rolling_windows} hours")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating rolling features: {str(e)}")
            raise

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features (matches training pipeline)"""
        logger.info("\n🔬 STEP 5: Creating Advanced Features")
        logger.info("-" * 38)
        
        try:
            df = df.copy()
            advanced_features = []
            
            # Pollution ratios and interactions
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df['pm2_5_to_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
                advanced_features.append('pm2_5_to_pm10_ratio')
            
            # Weather-pollution interactions
            if 'wind_speed' in df.columns and 'pm2_5' in df.columns:
                df['wind_pm_interaction'] = df['wind_speed'] * df['pm2_5']
                advanced_features.append('wind_pm_interaction')
            
            if 'relative_humidity' in df.columns and 'pm2_5' in df.columns:
                df['humidity_pm_interaction'] = df['relative_humidity'] * df['pm2_5']
                advanced_features.append('humidity_pm_interaction')
            
            # Temperature effects
            if 'temperature' in df.columns:
                df['temperature_squared'] = df['temperature'] ** 2
                df['temperature_normalized'] = (df['temperature'] - df['temperature'].mean()) / (df['temperature'].std() + 1e-6)
                advanced_features.extend(['temperature_squared', 'temperature_normalized'])
            
            # Pressure variations
            if 'pressure' in df.columns:
                df['pressure_change_1h'] = df['pressure'].diff(1).fillna(0)
                df['pressure_change_3h'] = df['pressure'].diff(3).fillna(0)
                advanced_features.extend(['pressure_change_1h', 'pressure_change_3h'])
            
            # AQI persistence features
            if 'aqi_numeric' in df.columns:
                df['aqi_change_1h'] = df['aqi_numeric'].diff(1).fillna(0)
                df['aqi_change_3h'] = df['aqi_numeric'].diff(3).fillna(0)
                df['aqi_volatility_6h'] = df['aqi_numeric'].rolling(6, min_periods=1).std().fillna(0)
                advanced_features.extend(['aqi_change_1h', 'aqi_change_3h', 'aqi_volatility_6h'])
            
            # Season and time-of-day interactions
            df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
            df['temp_humidity_interaction'] = df.get('temperature', 0) * df.get('relative_humidity', 0)
            advanced_features.extend(['hour_weekend_interaction', 'temp_humidity_interaction'])
            
            self.feature_metadata["categories"]["advanced_features"] = advanced_features
            
            logger.info(f"✅ Created {len(advanced_features)} advanced features")
            logger.info(f"🔬 Types: ratios, interactions, changes, volatility")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating advanced features: {str(e)}")
            raise

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate engineered features match training expectations"""
        logger.info("\n✅ STEP 6: Validating Features")
        logger.info("-" * 30)
        
        try:
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "total_features": len(df.columns),
                "target_features": 215,
                "validation_status": "unknown",
                "issues": [],
                "feature_summary": {}
            }
            
            # Count features by category
            total_engineered = 0
            for category, features in self.feature_metadata["categories"].items():
                count = len(features)
                total_engineered += count
                validation_results["feature_summary"][category] = count
                logger.info(f"   {category}: {count} features")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            high_missing = missing_counts[missing_counts > len(df) * 0.5]
            
            if len(high_missing) > 0:
                validation_results["issues"].append(f"High missing values in: {list(high_missing.index)}")
                logger.warning(f"⚠️ High missing values in {len(high_missing)} features")
            
            # Check for infinite values
            inf_columns = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if np.isinf(df[col]).any():
                    inf_columns.append(col)
            
            if inf_columns:
                validation_results["issues"].append(f"Infinite values in: {inf_columns}")
                logger.warning(f"⚠️ Infinite values in {len(inf_columns)} features")
            
            # Overall validation status
            if len(validation_results["issues"]) == 0:
                validation_results["validation_status"] = "passed"
                logger.info("✅ Feature validation passed")
            else:
                validation_results["validation_status"] = "warnings"
                logger.warning(f"⚠️ Feature validation completed with {len(validation_results['issues'])} warnings")
            
            logger.info(f"📊 Total features created: {total_engineered}")
            logger.info(f"🎯 Target features: {validation_results['target_features']}")
            
            return validation_results["validation_status"] == "passed", validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating features: {str(e)}")
            return False, {"error": str(e)}

    def save_features(self, df: pd.DataFrame, validation_results: Dict) -> bool:
        """Save engineered features and metadata"""
        logger.info("\n💾 STEP 7: Saving Features")
        logger.info("-" * 25)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save feature dataset
            feature_file = os.path.join(self.feature_dir, "realtime_features.csv")
            df.to_csv(feature_file, index=False)
            
            # Save timestamped version
            timestamped_file = os.path.join(self.feature_dir, f"features_{timestamp}.csv")
            df.to_csv(timestamped_file, index=False)
            
            # Update feature metadata
            self.feature_metadata.update({
                "last_update": datetime.now().isoformat(),
                "total_features": len(df.columns),
                "total_records": len(df),
                "date_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "validation_results": validation_results
            })
            
            # Save metadata
            metadata_file = os.path.join(self.metadata_dir, "realtime_feature_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.feature_metadata, f, indent=4)
            
            logger.info(f"✅ Features saved: {feature_file}")
            logger.info(f"📊 Dataset: {len(df)} records, {len(df.columns)} features")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving features: {str(e)}")
            return False

    def run_realtime_pipeline(self) -> bool:
        """Run the complete real-time feature engineering pipeline"""
        logger.info("\n🚀 STARTING REAL-TIME FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        try:
            # Step 1: Load latest data
            df = self.load_latest_data()
            
            # Step 2: Create temporal features
            df = self.create_temporal_features(df)
            
            # Step 3: Create lag features
            df = self.create_lag_features(df)
            
            # Step 4: Create rolling features
            df = self.create_rolling_features(df)
            
            # Step 5: Create advanced features
            df = self.create_advanced_features(df)
            
            # Step 6: Validate features
            validation_passed, validation_results = self.validate_features(df)
            
            # Step 7: Save features
            save_success = self.save_features(df, validation_results)
            
            if save_success and validation_passed:
                logger.info("\n🎉 REAL-TIME FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
                logger.info("✅ Features ready for Hopsworks integration")
                return True
            else:
                logger.warning("\n⚠️ Real-time feature engineering completed with issues")
                return False
                
        except Exception as e:
            logger.error(f"\n❌ Real-time feature engineering failed: {str(e)}")
            return False

def main():
    """Main function for real-time feature engineering"""
    engineer = RealTimeFeatureEngineer()
    success = engineer.run_realtime_pipeline()
    
    if success:
        print("\n🎯 REAL-TIME FEATURE ENGINEERING SUCCESS!")
        print("🚀 Ready for Hopsworks integration")
    else:
        print("\n❌ Real-time feature engineering failed")
        print("📋 Check logs for details")

if __name__ == "__main__":
    main()
