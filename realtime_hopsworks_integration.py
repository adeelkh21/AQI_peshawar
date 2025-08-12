"""
Real-time Hopsworks Integration
==============================

This script updates Hopsworks feature store with newly engineered features
in real-time, maintaining the feature store current with hourly data updates.

Features:
- Connects to Hopsworks using environment credentials
- Updates feature groups incrementally
- Maintains feature versioning and lineage
- Validates feature store updates
- Provides detailed logging and monitoring

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
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

class RealTimeHopsworksManager:
    """Real-time Hopsworks feature store integration"""
    
    def __init__(self):
        """Initialize real-time Hopsworks manager"""
        logger.info("🏪 REAL-TIME HOPSWORKS INTEGRATION")
        logger.info("=" * 45)
        
        self.project = None
        self.fs = None
        self.feature_groups = {}
        self.connection_verified = False
        
        # Setup directories
        self.feature_dir = os.path.join("data_repositories", "features", "engineered")
        self.hopsworks_dir = os.path.join("data_repositories", "hopsworks")
        self.logs_dir = os.path.join(self.hopsworks_dir, "logs")
        
        # Create directories
        os.makedirs(self.hopsworks_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Feature group configuration
        self.feature_categories = {
            'pollution': {
                'description': 'Real-time air pollution measurements and derived features',
                'prefix_patterns': ['pm2_5', 'pm10', 'no2', 'o3', 'aqi'],
                'update_frequency': 'hourly'
            },
            'weather': {
                'description': 'Real-time weather data and meteorological features',
                'prefix_patterns': ['temperature', 'humidity', 'wind', 'pressure'],
                'update_frequency': 'hourly'
            },
            'temporal': {
                'description': 'Time-based features and cyclical encodings',
                'prefix_patterns': ['hour', 'day', 'month', 'weekend', 'rush', 'night', 'sin', 'cos'],
                'update_frequency': 'hourly'
            },
            'lag_features': {
                'description': 'Multi-horizon lag features for temporal patterns',
                'prefix_patterns': ['lag'],
                'update_frequency': 'hourly'
            },
            'rolling_stats': {
                'description': 'Rolling statistics and volatility measures',
                'prefix_patterns': ['rolling'],
                'update_frequency': 'hourly'
            },
            'advanced': {
                'description': 'Advanced engineered features and interactions',
                'prefix_patterns': ['ratio', 'interaction', 'change', 'volatility', 'normalized'],
                'update_frequency': 'hourly'
            }
        }

    def step1_verify_hopsworks_connection(self) -> bool:
        """Step 1: Verify Hopsworks connection with environment credentials"""
        logger.info("\n🔌 STEP 1: Verifying Hopsworks Connection")
        logger.info("-" * 42)
        
        if not HOPSWORKS_AVAILABLE:
            logger.error("❌ Hopsworks libraries not available")
            return False
        
        try:
            # Get credentials from environment
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT', 'aqi_prediction_peshawar')
            
            if not api_key:
                logger.error("❌ HOPSWORKS_API_KEY not found in environment variables")
                logger.info("💡 Set HOPSWORKS_API_KEY in GitHub secrets")
                return False
            
            logger.info(f"🔑 API key found in environment")
            logger.info(f"📁 Project: {project_name}")
            logger.info("🔄 Connecting to Hopsworks...")
            
            # Establish connection
            self.project = hopsworks.login(
                project=project_name,
                api_key_value=api_key
            )
            
            # Get feature store
            self.fs = self.project.get_feature_store()
            self.connection_verified = True
            
            logger.info("✅ Hopsworks connection established")
            logger.info(f"🏪 Feature store: {self.fs.name}")
            logger.info(f"📊 Project ID: {self.project.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Hopsworks: {str(e)}")
            logger.info("💡 Check API key and project name")
            return False

    def step2_load_latest_features(self) -> Optional[pd.DataFrame]:
        """Step 2: Load latest engineered features"""
        logger.info("\n📊 STEP 2: Loading Latest Features")
        logger.info("-" * 35)
        
        try:
            feature_file = os.path.join(self.feature_dir, "realtime_features.csv")
            
            if not os.path.exists(feature_file):
                logger.error(f"❌ Feature file not found: {feature_file}")
                return None
            
            df = pd.read_csv(feature_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"✅ Features loaded: {len(df)} records")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"🔢 Feature columns: {len(df.columns)}")
            
            # Check data freshness
            latest_timestamp = df['timestamp'].max()
            hours_since_latest = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            if hours_since_latest > 2:
                logger.warning(f"⚠️ Features are {hours_since_latest:.1f} hours old")
            else:
                logger.info(f"✅ Features are fresh: {hours_since_latest:.1f} hours old")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading features: {str(e)}")
            return None

    def step3_categorize_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Step 3: Categorize features for feature groups"""
        logger.info("\n📂 STEP 3: Categorizing Features")
        logger.info("-" * 32)
        
        try:
            categorized = {category: [] for category in self.feature_categories.keys()}
            
            # Always include timestamp
            base_columns = ['timestamp']
            
            # Categorize each feature column
            for column in df.columns:
                if column in base_columns:
                    continue
                
                categorized_flag = False
                
                # Match features to categories by patterns
                for category, config in self.feature_categories.items():
                    patterns = config['prefix_patterns']
                    
                    if any(pattern in column.lower() for pattern in patterns):
                        categorized[category].append(column)
                        categorized_flag = True
                        break
                
                # If no category match, add to advanced
                if not categorized_flag:
                    categorized['advanced'].append(column)
            
            # Report categorization
            logger.info("📋 Feature categorization:")
            total_categorized = 0
            for category, features in categorized.items():
                if features:
                    logger.info(f"   {category}: {len(features)} features")
                    total_categorized += len(features)
            
            logger.info(f"✅ Categorized {total_categorized}/{len(df.columns)-1} features")
            
            return categorized
            
        except Exception as e:
            logger.error(f"❌ Error categorizing features: {str(e)}")
            return {}

    def step4_update_feature_groups(self, df: pd.DataFrame, categorized_features: Dict[str, List[str]]) -> bool:
        """Step 4: Update Hopsworks feature groups with new data"""
        logger.info("\n🔄 STEP 4: Updating Feature Groups")
        logger.info("-" * 35)
        
        if not self.connection_verified:
            logger.error("❌ Hopsworks connection not verified")
            return False
        
        try:
            update_results = {}
            
            for category, features in categorized_features.items():
                if not features:
                    logger.info(f"⚠️ Skipping {category} - no features")
                    continue
                
                logger.info(f"\n📦 Updating feature group: {category}")
                
                # Prepare feature group data
                fg_columns = ['timestamp'] + features
                fg_data = df[fg_columns].copy()
                
                # Handle missing values
                fg_data = fg_data.fillna(0)
                
                # Get only the latest records for incremental update
                # For real-time, we typically want the last few hours of data
                latest_hours = 6  # Update with last 6 hours of data
                cutoff_time = datetime.now() - timedelta(hours=latest_hours)
                recent_data = fg_data[fg_data['timestamp'] >= cutoff_time]
                
                if len(recent_data) == 0:
                    logger.warning(f"⚠️ No recent data for {category}")
                    continue
                
                logger.info(f"   📊 Features: {len(features)}")
                logger.info(f"   📈 Records: {len(recent_data)} (last {latest_hours}h)")
                logger.info(f"   📅 Range: {recent_data['timestamp'].min()} to {recent_data['timestamp'].max()}")
                
                try:
                    # Feature group name
                    fg_name = f"aqi_{category}_realtime"
                    
                    # Get or create feature group
                    try:
                        fg = self.fs.get_feature_group(fg_name, version=1)
                        logger.info(f"   📋 Using existing feature group: {fg_name}")
                    except:
                        # Create new feature group
                        logger.info(f"   🆕 Creating new feature group: {fg_name}")
                        fg = self.fs.create_feature_group(
                            name=fg_name,
                            version=1,
                            description=f"Real-time {self.feature_categories[category]['description']}",
                            primary_key=['timestamp'],
                            event_time='timestamp',
                            online_enabled=False  # Disable online for timestamp compatibility
                        )
                    
                    # Insert data
                    logger.info(f"   📤 Inserting {len(recent_data)} records...")
                    fg.insert(recent_data, write_options={"start_offline_materialization": False})
                    
                    update_results[category] = {
                        "status": "success",
                        "records_inserted": len(recent_data),
                        "features_count": len(features),
                        "feature_group_name": fg_name
                    }
                    
                    logger.info(f"   ✅ Successfully updated {fg_name}")
                    
                except Exception as fg_error:
                    logger.error(f"   ❌ Error updating {category}: {str(fg_error)}")
                    update_results[category] = {
                        "status": "failed",
                        "error": str(fg_error)
                    }
            
            # Save update results
            self._save_update_results(update_results)
            
            # Check overall success
            successful_updates = sum(1 for result in update_results.values() if result.get("status") == "success")
            total_attempts = len(update_results)
            
            if successful_updates > 0:
                logger.info(f"\n✅ Feature store update completed")
                logger.info(f"📊 Success rate: {successful_updates}/{total_attempts}")
                return True
            else:
                logger.error(f"\n❌ All feature store updates failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating feature groups: {str(e)}")
            return False

    def step5_validate_updates(self) -> bool:
        """Step 5: Validate feature store updates"""
        logger.info("\n✅ STEP 5: Validating Updates")
        logger.info("-" * 30)
        
        if not self.connection_verified:
            logger.error("❌ Hopsworks connection not verified")
            return False
        
        try:
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "feature_groups_checked": 0,
                "accessible_groups": 0,
                "validation_status": "unknown",
                "issues": []
            }
            
            # Check each feature group
            for category in self.feature_categories.keys():
                fg_name = f"aqi_{category}_realtime"
                
                try:
                    fg = self.fs.get_feature_group(fg_name, version=1)
                    validation_results["feature_groups_checked"] += 1
                    
                    # Try to read schema
                    schema = fg.schema
                    logger.info(f"   {category}: ✅ Accessible ({len(schema)} columns)")
                    validation_results["accessible_groups"] += 1
                    
                except Exception as e:
                    logger.warning(f"   {category}: ⚠️ Issue - {str(e)[:50]}...")
                    validation_results["issues"].append(f"{category}: {str(e)}")
            
            # Overall validation status
            if validation_results["accessible_groups"] == validation_results["feature_groups_checked"]:
                validation_results["validation_status"] = "passed"
                logger.info("✅ All feature groups validated successfully")
            elif validation_results["accessible_groups"] > 0:
                validation_results["validation_status"] = "partial"
                logger.warning(f"⚠️ Partial validation: {validation_results['accessible_groups']}/{validation_results['feature_groups_checked']} groups accessible")
            else:
                validation_results["validation_status"] = "failed"
                logger.error("❌ Feature group validation failed")
            
            # Save validation results
            self._save_validation_results(validation_results)
            
            return validation_results["validation_status"] in ["passed", "partial"]
            
        except Exception as e:
            logger.error(f"❌ Error validating updates: {str(e)}")
            return False

    def _save_update_results(self, results: Dict) -> None:
        """Save update results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.logs_dir, f"update_results_{timestamp}.json")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"📁 Update results saved: {results_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save update results: {str(e)}")

    def _save_validation_results(self, results: Dict) -> None:
        """Save validation results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            validation_file = os.path.join(self.logs_dir, f"validation_results_{timestamp}.json")
            
            with open(validation_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"📁 Validation results saved: {validation_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save validation results: {str(e)}")

    def run_realtime_integration(self) -> bool:
        """Run complete real-time Hopsworks integration"""
        logger.info("\n🚀 STARTING REAL-TIME HOPSWORKS INTEGRATION")
        logger.info("=" * 50)
        
        try:
            # Step 1: Verify connection
            if not self.step1_verify_hopsworks_connection():
                return False
            
            # Step 2: Load features
            df = self.step2_load_latest_features()
            if df is None:
                return False
            
            # Step 3: Categorize features
            categorized_features = self.step3_categorize_features(df)
            if not categorized_features:
                return False
            
            # Step 4: Update feature groups
            update_success = self.step4_update_feature_groups(df, categorized_features)
            if not update_success:
                logger.warning("⚠️ Feature group updates had issues")
            
            # Step 5: Validate updates
            validation_success = self.step5_validate_updates()
            
            # Overall success assessment
            if update_success and validation_success:
                logger.info("\n🎉 REAL-TIME HOPSWORKS INTEGRATION COMPLETED SUCCESSFULLY!")
                logger.info("✅ Feature store updated with latest features")
                return True
            elif update_success:
                logger.warning("\n⚠️ Real-time integration completed with validation issues")
                return True
            else:
                logger.error("\n❌ Real-time Hopsworks integration failed")
                return False
                
        except Exception as e:
            logger.error(f"\n❌ Real-time integration failed: {str(e)}")
            return False

def main():
    """Main function for real-time Hopsworks integration"""
    manager = RealTimeHopsworksManager()
    success = manager.run_realtime_integration()
    
    if success:
        print("\n🎯 REAL-TIME HOPSWORKS INTEGRATION SUCCESS!")
        print("🏪 Feature store updated with latest data")
        print("🚀 Ready for model retraining")
    else:
        print("\n❌ Real-time Hopsworks integration failed")
        print("📋 Check logs and credentials")

if __name__ == "__main__":
    main()
