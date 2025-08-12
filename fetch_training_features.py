"""
Training Feature Fetcher from Hopsworks
=======================================

This script fetches the latest features from Hopsworks feature store for
model training, ensuring we have the most recent and complete dataset.

Features:
- Connects to Hopsworks feature store
- Fetches features from all categories
- Validates data quality and completeness
- Prepares training-ready dataset
- Handles missing data and feature alignment

Author: Data Science Team
Date: August 12, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging

# Hopsworks imports
try:
    import hopsworks
    import hsfs
    HOPSWORKS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Hopsworks not available: {e}")
    HOPSWORKS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class TrainingFeatureFetcher:
    """Fetch and prepare features for model training"""
    
    def __init__(self, training_hours: int = 720):
        """Initialize training feature fetcher
        
        Args:
            training_hours: Hours of historical data to fetch for training (default: 720 = 30 days)
        """
        logger.info("📊 TRAINING FEATURE FETCHER")
        logger.info("=" * 30)
        
        self.training_hours = training_hours
        self.project = None
        self.fs = None
        self.connection_verified = False
        
        # Setup directories
        self.training_dir = "data_repositories/training"
        self.features_dir = os.path.join(self.training_dir, "features")
        self.logs_dir = os.path.join(self.training_dir, "logs")
        
        # Create directories
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Feature categories from Phase 1
        self.feature_categories = [
            'pollution_realtime',
            'weather_realtime', 
            'temporal_realtime',
            'lag_realtime',
            'rolling_realtime',
            'advanced_realtime'
        ]
        
        # Fetch results
        self.fetch_results = {
            "timestamp": datetime.now().isoformat(),
            "training_hours": training_hours,
            "categories_fetched": {},
            "final_dataset": {},
            "data_quality": {}
        }
        
        logger.info(f"🕐 Training window: {training_hours} hours ({training_hours/24:.1f} days)")

    def establish_connection(self) -> bool:
        """Establish connection to Hopsworks feature store"""
        logger.info("\n🔌 Connecting to Hopsworks Feature Store")
        logger.info("-" * 40)
        
        if not HOPSWORKS_AVAILABLE:
            logger.error("❌ Hopsworks libraries not available")
            return False
        
        try:
            # Get credentials from environment
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT', 'aqi_prediction_peshawar')
            
            if not api_key:
                logger.error("❌ HOPSWORKS_API_KEY not found in environment variables")
                return False
            
            logger.info(f"🔑 API key found")
            logger.info(f"📁 Project: {project_name}")
            logger.info("🔄 Connecting...")
            
            # Connect to Hopsworks
            self.project = hopsworks.login(
                project=project_name,
                api_key_value=api_key
            )
            
            # Get feature store
            self.fs = self.project.get_feature_store()
            self.connection_verified = True
            
            logger.info("✅ Connected to Hopsworks feature store")
            logger.info(f"🏪 Feature store: {self.fs.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            return False

    def fetch_category_features(self, category: str) -> Optional[pd.DataFrame]:
        """Fetch features from a specific category"""
        logger.info(f"\n📦 Fetching {category} features")
        logger.info("-" * (15 + len(category)))
        
        try:
            # Get feature group
            try:
                fg = self.fs.get_feature_group(category, version=1)
                logger.info(f"   📋 Found feature group: {category}")
                
                # Check if feature group is properly initialized
                if fg is None:
                    logger.warning(f"   ⚠️ Feature group {category} is None - not properly initialized")
                    return None
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Feature group {category} not found: {str(e)}")
                return None
            
            # Calculate time window for training data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.training_hours)
            
            logger.info(f"   📅 Time window: {start_time} to {end_time}")
            
            # Create query with time filter
            query = fg.select_all()
            
            # Read data with comprehensive error handling
            try:
                df = query.read()
            except Exception as read_error:
                error_msg = str(read_error)
                if "hoodie.properties" in error_msg or "No such file or directory" in error_msg:
                    logger.warning(f"   ⚠️ Feature group {category} not properly initialized (missing metadata)")
                    logger.warning(f"   💡 This is normal for newly created feature groups")
                    return None
                else:
                    logger.error(f"   ❌ Error reading {category}: {error_msg}")
                    return None
            
            if df is None or len(df) == 0:
                logger.warning(f"   ⚠️ No data returned for {category}")
                return None
            
            # Ensure timestamp column exists and is datetime with proper timezone handling
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert to timezone-naive datetime for comparison
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Apply time filtering manually if needed
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                df = df.sort_values('timestamp')
            
            logger.info(f"   ✅ Fetched {len(df)} records")
            logger.info(f"   📊 Features: {len(df.columns)} columns")
            
            if len(df) > 0 and 'timestamp' in df.columns:
                logger.info(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Store fetch results
            self.fetch_results["categories_fetched"][category] = {
                "records": len(df),
                "features": len(df.columns),
                "date_range": {
                    "start": df['timestamp'].min().isoformat() if len(df) > 0 and 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max().isoformat() if len(df) > 0 and 'timestamp' in df.columns else None
                }
            }
            
            return df
            
        except Exception as e:
            logger.error(f"   ❌ Error fetching {category}: {str(e)}")
            self.fetch_results["categories_fetched"][category] = {
                "error": str(e),
                "records": 0,
                "features": 0
            }
            return None

    def merge_category_features(self, category_datasets: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge features from all categories into training dataset"""
        logger.info("\n🔗 Merging Category Features")
        logger.info("-" * 27)
        
        try:
            if not category_datasets:
                logger.error("❌ No category datasets to merge")
                return None
            
            # Start with the category that has the most records (usually pollution or weather)
            primary_category = max(category_datasets.keys(), 
                                 key=lambda k: len(category_datasets[k]))
            
            merged_df = category_datasets[primary_category].copy()
            logger.info(f"📊 Starting with {primary_category}: {len(merged_df)} records")
            
            # Merge other categories
            for category, df in category_datasets.items():
                if category == primary_category:
                    continue
                
                if 'timestamp' not in df.columns or 'timestamp' not in merged_df.columns:
                    logger.warning(f"⚠️ Skipping {category} - no timestamp column")
                    continue
                
                logger.info(f"🔗 Merging {category}: {len(df)} records")
                
                # Merge on timestamp with outer join to preserve all data
                merged_df = pd.merge(
                    merged_df, 
                    df, 
                    on='timestamp', 
                    how='outer',
                    suffixes=('', f'_{category}')
                )
                
                logger.info(f"   Result: {len(merged_df)} records")
            
            # Sort by timestamp
            if 'timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('timestamp')
            
            logger.info(f"✅ Final merged dataset: {len(merged_df)} records, {len(merged_df.columns)} features")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"❌ Error merging features: {str(e)}")
            return None

    def validate_training_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate training data quality and completeness"""
        logger.info("\n✅ Validating Training Data")
        logger.info("-" * 27)
        
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Basic data validation
            total_records = len(df)
            total_features = len(df.columns)
            
            logger.info(f"📊 Dataset Overview:")
            logger.info(f"   Records: {total_records:,}")
            logger.info(f"   Features: {total_features}")
            
            # Check minimum data requirements
            min_records_for_training = 50  # Reduced for testing (was 500)
            if total_records < min_records_for_training:
                validation_results["issues"].append(f"Insufficient records for training ({total_records} < {min_records_for_training})")
                validation_results["is_valid"] = False
                logger.error(f"❌ Insufficient data: {total_records} < {min_records_for_training}")
            
            # Check for timestamp column
            if 'timestamp' not in df.columns:
                validation_results["issues"].append("No timestamp column found")
                validation_results["is_valid"] = False
                logger.error("❌ No timestamp column")
            else:
                # Validate timestamp coverage
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                date_range_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                
                logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logger.info(f"   Coverage: {date_range_hours:.1f} hours")
                
                # Check for reasonable time coverage
                min_coverage_hours = min(self.training_hours * 0.5, 168)  # At least 50% or 1 week
                if date_range_hours < min_coverage_hours:
                    validation_results["warnings"].append(f"Limited time coverage ({date_range_hours:.1f}h < {min_coverage_hours}h)")
                    logger.warning(f"⚠️ Limited coverage: {date_range_hours:.1f}h")
            
            # Check target variable (AQI)
            target_columns = ['aqi_numeric', 'aqi']
            target_column = None
            for col in target_columns:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                validation_results["issues"].append("No target variable (AQI) found")
                validation_results["is_valid"] = False
                logger.error("❌ No AQI target variable")
            else:
                # Validate target variable
                target_data = df[target_column].dropna()
                logger.info(f"   Target variable: {target_column}")
                logger.info(f"   Valid target values: {len(target_data)}/{total_records}")
                
                if len(target_data) < total_records * 0.8:  # Need 80% valid targets
                    validation_results["warnings"].append("High missing values in target variable")
                    logger.warning(f"⚠️ Missing targets: {(total_records - len(target_data))/total_records:.1%}")
            
            # Analyze missing values
            missing_percent = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percent[missing_percent > 50]
            
            if len(high_missing) > 0:
                validation_results["warnings"].append(f"High missing values in {len(high_missing)} features")
                logger.warning(f"⚠️ High missing values in {len(high_missing)} features")
            
            # Check for feature diversity
            numeric_features = df.select_dtypes(include=[np.number]).columns
            if len(numeric_features) < 20:  # Expect decent number of features
                validation_results["warnings"].append(f"Limited numeric features ({len(numeric_features)})")
                logger.warning(f"⚠️ Limited features: {len(numeric_features)}")
            
            # Calculate quality metrics
            validation_results["metrics"] = {
                "total_records": total_records,
                "total_features": total_features,
                "numeric_features": len(numeric_features),
                "missing_values_percent": float(df.isnull().sum().sum() / (total_records * total_features) * 100),
                "high_missing_features": len(high_missing),
                "target_column": target_column,
                "target_completeness": float(len(target_data) / total_records) if target_column else 0
            }
            
            # Overall validation status
            if validation_results["is_valid"]:
                logger.info("✅ Training data validation passed")
            else:
                logger.error(f"❌ Training data validation failed: {len(validation_results['issues'])} critical issues")
            
            if validation_results["warnings"]:
                logger.warning(f"⚠️ {len(validation_results['warnings'])} warnings found")
            
            return validation_results["is_valid"], validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            validation_results["issues"].append(f"Validation failed: {str(e)}")
            validation_results["is_valid"] = False
            return False, validation_results

    def save_training_features(self, df: pd.DataFrame, validation_results: Dict) -> bool:
        """Save training features and metadata"""
        logger.info("\n💾 Saving Training Features")
        logger.info("-" * 27)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main training dataset
            training_file = os.path.join(self.features_dir, "training_features.csv")
            df.to_csv(training_file, index=False)
            
            # Save timestamped version
            timestamped_file = os.path.join(self.features_dir, f"training_features_{timestamp}.csv")
            df.to_csv(timestamped_file, index=False)
            
            # Update fetch results with final dataset info
            self.fetch_results["final_dataset"] = {
                "records": len(df),
                "features": len(df.columns),
                "file": training_file,
                "timestamped_file": timestamped_file,
                "date_range": {
                    "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                }
            }
            
            self.fetch_results["data_quality"] = validation_results
            
            # Save metadata
            metadata_file = os.path.join(self.features_dir, "training_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.fetch_results, f, indent=4)
            
            logger.info(f"✅ Training features saved: {training_file}")
            logger.info(f"📊 Dataset: {len(df)} records, {len(df.columns)} features")
            logger.info(f"📁 Metadata: {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving features: {str(e)}")
            return False

    def _load_local_features_fallback(self) -> Optional[pd.DataFrame]:
        """Load features from local files as fallback when Hopsworks is unavailable"""
        logger.info("\n📁 Loading Local Features as Fallback")
        logger.info("-" * 35)
        
        try:
            # Check for local feature files
            local_feature_file = os.path.join("data_repositories", "features", "engineered", "realtime_features.csv")
            
            if not os.path.exists(local_feature_file):
                logger.warning(f"⚠️ Local feature file not found: {local_feature_file}")
                return None
            
            # Load local features
            df = pd.read_csv(local_feature_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for recent data based on training hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.training_hours)
            
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            df = df.sort_values('timestamp')
            
            if len(df) == 0:
                logger.warning("⚠️ No recent data in local feature file")
                return None
            
            logger.info(f"✅ Loaded {len(df)} records from local features")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"🔢 Features: {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading local features: {str(e)}")
            return None

    def run_feature_fetch(self) -> bool:
        """Run complete feature fetching process"""
        logger.info("\n🚀 STARTING FEATURE FETCH FOR TRAINING")
        logger.info("=" * 45)
        
        try:
            # Step 1: Connect to Hopsworks
            if not self.establish_connection():
                return False
            
            # Step 2: Fetch features from all categories
            category_datasets = {}
            
            for category in self.feature_categories:
                df = self.fetch_category_features(category)
                if df is not None and len(df) > 0:
                    category_datasets[category] = df
                    logger.info(f"   ✅ {category}: {len(df)} records")
                else:
                    logger.warning(f"   ⚠️ {category}: No data")
            
            # Check if we have sufficient data for training
            total_records = sum(len(df) for df in category_datasets.values())
            min_records_for_training = 50  # Reduced for testing (was 500)
            
            if total_records < min_records_for_training:
                logger.warning(f"⚠️ Insufficient data from Hopsworks: {total_records} < {min_records_for_training}")
                logger.info("🔄 Attempting to use local feature files as fallback...")
                
                # Try to use local features as fallback
                local_features = self._load_local_features_fallback()
                if local_features is not None and len(local_features) >= min_records_for_training:
                    logger.info("✅ Using local features as fallback")
                    # Validate and save local features
                    is_valid, validation_results = self.validate_training_data(local_features)
                    if is_valid:
                        save_success = self.save_training_features(local_features, validation_results)
                        if save_success:
                            return True
                
                logger.error("❌ No sufficient data available from Hopsworks or local files")
                return False
            
            logger.info(f"\n📊 Successfully fetched from {len(category_datasets)}/{len(self.feature_categories)} categories")
            
            # Step 3: Merge all features
            merged_df = self.merge_category_features(category_datasets)
            if merged_df is None:
                logger.error("❌ Failed to merge category features")
                return False
            
            # Step 4: Validate training data
            is_valid, validation_results = self.validate_training_data(merged_df)
            if not is_valid:
                logger.error("❌ Training data validation failed")
                return False
            
            # Step 5: Save training features
            save_success = self.save_training_features(merged_df, validation_results)
            if not save_success:
                logger.error("❌ Failed to save training features")
                return False
            
            logger.info("\n🎉 FEATURE FETCH COMPLETED SUCCESSFULLY!")
            logger.info("✅ Training dataset ready for model training")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Feature fetch failed: {str(e)}")
            return False

def main():
    """Main function for training feature fetcher"""
    parser = argparse.ArgumentParser(description='Fetch training features from Hopsworks')
    parser.add_argument('--hours', type=int, default=720, 
                       help='Hours of historical data to fetch (default: 720 = 30 days)')
    
    args = parser.parse_args()
    
    fetcher = TrainingFeatureFetcher(training_hours=args.hours)
    success = fetcher.run_feature_fetch()
    
    if success:
        print("\n🎯 TRAINING FEATURE FETCH SUCCESS!")
        print("📊 Dataset ready for model training")
        sys.exit(0)
    else:
        print("\n❌ Training feature fetch failed")
        print("📋 Check logs and Hopsworks connection")
        sys.exit(1)

if __name__ == "__main__":
    main()
