"""
Training Dataset Preparation
===========================

This script prepares the training dataset for model training by:
- Loading features from Hopsworks fetch
- Creating proper train/validation/test splits
- Handling missing values and outliers
- Feature scaling and preprocessing
- Temporal validation setup for time series

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class TrainingDatasetPreparator:
    """Prepare training dataset with proper validation splits"""
    
    def __init__(self):
        """Initialize training dataset preparator"""
        logger.info("🔧 TRAINING DATASET PREPARATION")
        logger.info("=" * 35)
        
        # Directories
        self.training_dir = "data_repositories/training"
        self.features_dir = os.path.join(self.training_dir, "features")
        self.datasets_dir = os.path.join(self.training_dir, "datasets")
        self.metadata_dir = os.path.join(self.training_dir, "metadata")
        
        # Create directories
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Preparation configuration
        self.config = {
            "validation_strategy": "temporal",  # temporal split for time series
            "test_size": 0.15,                 # 15% for final test
            "validation_size": 0.20,           # 20% for validation
            "min_train_samples": 1000,         # Minimum training samples
            "feature_selection": {
                "remove_low_variance": True,
                "correlation_threshold": 0.95,
                "missing_threshold": 0.8
            },
            "preprocessing": {
                "scaling_method": "robust",     # robust to outliers
                "handle_outliers": True,
                "fill_strategy": "forward_fill"
            }
        }
        
        self.preparation_results = {
            "timestamp": datetime.now().isoformat(),
            "input_data": {},
            "preprocessing": {},
            "splits": {},
            "feature_selection": {},
            "validation_setup": {}
        }

    def load_training_features(self) -> Optional[pd.DataFrame]:
        """Load training features from fetch process"""
        logger.info("\n📊 Loading Training Features")
        logger.info("-" * 28)
        
        try:
            # Load main training dataset
            training_file = os.path.join(self.features_dir, "training_features.csv")
            
            if not os.path.exists(training_file):
                logger.error(f"❌ Training features not found: {training_file}")
                return None
            
            df = pd.read_csv(training_file)
            
            # Load metadata
            metadata_file = os.path.join(self.features_dir, "training_metadata.json")
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"✅ Loaded training features: {len(df)} records")
            logger.info(f"📊 Features: {len(df.columns)} columns")
            
            # Basic data info
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Store input data info
            self.preparation_results["input_data"] = {
                "records": len(df),
                "features": len(df.columns),
                "date_range": {
                    "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                },
                "metadata": metadata.get("data_quality", {})
            }
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading training features: {str(e)}")
            return None

    def identify_target_and_features(self, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
        """Identify target variable and feature columns"""
        logger.info("\n🎯 Identifying Target and Features")
        logger.info("-" * 34)
        
        # Identify target variable (AQI)
        target_candidates = ['aqi_numeric', 'aqi', 'target']
        target_column = None
        
        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                logger.info(f"✅ Target variable found: {target_column}")
                break
        
        if target_column is None:
            logger.error("❌ No target variable found in dataset")
            return None, []
        
        # Identify feature columns (exclude metadata and target)
        exclude_columns = ['timestamp', target_column, 'index']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        logger.info(f"📊 Feature columns: {len(feature_columns)}")
        logger.info(f"🎯 Target column: {target_column}")
        
        # Check target variable quality
        target_values = df[target_column].dropna()
        logger.info(f"📈 Target statistics:")
        logger.info(f"   Valid values: {len(target_values)}/{len(df)} ({len(target_values)/len(df):.1%})")
        logger.info(f"   Range: {target_values.min():.1f} to {target_values.max():.1f}")
        logger.info(f"   Mean: {target_values.mean():.1f} ± {target_values.std():.1f}")
        
        return target_column, feature_columns

    def preprocess_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Preprocess features with cleaning and transformations"""
        logger.info("\n🧹 Preprocessing Features")
        logger.info("-" * 23)
        
        try:
            df_processed = df.copy()
            preprocessing_stats = {
                "original_features": len(feature_columns),
                "missing_values_handled": 0,
                "outliers_handled": 0,
                "features_removed": [],
                "final_features": 0
            }
            
            # Handle missing values
            logger.info("🔧 Handling missing values...")
            missing_before = df_processed[feature_columns].isnull().sum().sum()
            
            if self.config["preprocessing"]["fill_strategy"] == "forward_fill":
                # Forward fill for time series data
                df_processed[feature_columns] = df_processed[feature_columns].fillna(method='ffill')
                # Backward fill for remaining NaNs at the beginning
                df_processed[feature_columns] = df_processed[feature_columns].fillna(method='bfill')
                # Fill any remaining with median
                for col in feature_columns:
                    if df_processed[col].isnull().any():
                        median_val = df_processed[col].median()
                        df_processed[col] = df_processed[col].fillna(median_val)
            
            missing_after = df_processed[feature_columns].isnull().sum().sum()
            preprocessing_stats["missing_values_handled"] = missing_before - missing_after
            logger.info(f"   Missing values: {missing_before} → {missing_after}")
            
            # Remove features with too many missing values
            if self.config["feature_selection"]["remove_low_variance"]:
                logger.info("🔧 Removing low-quality features...")
                
                missing_threshold = self.config["feature_selection"]["missing_threshold"]
                features_to_remove = []
                
                for col in feature_columns:
                    missing_ratio = df_processed[col].isnull().sum() / len(df_processed)
                    if missing_ratio > (1 - missing_threshold):
                        features_to_remove.append(col)
                
                if features_to_remove:
                    logger.info(f"   Removing {len(features_to_remove)} features with >80% missing values")
                    feature_columns = [col for col in feature_columns if col not in features_to_remove]
                    preprocessing_stats["features_removed"].extend(features_to_remove)
            
            # Handle outliers (optional)
            if self.config["preprocessing"]["handle_outliers"]:
                logger.info("🔧 Handling outliers...")
                outliers_handled = 0
                
                for col in feature_columns:
                    if df_processed[col].dtype in ['float64', 'int64']:
                        # Use IQR method to cap outliers
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers_before = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
                        
                        # Cap outliers
                        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                        outliers_handled += outliers_before
                
                preprocessing_stats["outliers_handled"] = outliers_handled
                logger.info(f"   Outliers capped: {outliers_handled}")
            
            # Remove highly correlated features
            correlation_threshold = self.config["feature_selection"]["correlation_threshold"]
            if correlation_threshold < 1.0:
                logger.info(f"🔧 Removing highly correlated features (>{correlation_threshold})...")
                
                # Calculate correlation matrix for numeric features only
                numeric_features = df_processed[feature_columns].select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 1:
                    corr_matrix = df_processed[numeric_features].corr().abs()
                    
                    # Find highly correlated feature pairs
                    high_corr_features = set()
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > correlation_threshold:
                                # Remove the feature with lower variance
                                feature1 = corr_matrix.columns[i]
                                feature2 = corr_matrix.columns[j]
                                
                                var1 = df_processed[feature1].var()
                                var2 = df_processed[feature2].var()
                                
                                if var1 < var2:
                                    high_corr_features.add(feature1)
                                else:
                                    high_corr_features.add(feature2)
                    
                    if high_corr_features:
                        logger.info(f"   Removing {len(high_corr_features)} highly correlated features")
                        feature_columns = [col for col in feature_columns if col not in high_corr_features]
                        preprocessing_stats["features_removed"].extend(list(high_corr_features))
            
            preprocessing_stats["final_features"] = len(feature_columns)
            
            logger.info(f"✅ Preprocessing completed:")
            logger.info(f"   Features: {preprocessing_stats['original_features']} → {preprocessing_stats['final_features']}")
            logger.info(f"   Removed: {len(preprocessing_stats['features_removed'])} features")
            
            self.preparation_results["preprocessing"] = preprocessing_stats
            
            return df_processed
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing features: {str(e)}")
            return df

    def create_temporal_splits(self, df: pd.DataFrame, target_column: str, feature_columns: List[str]) -> Dict:
        """Create temporal train/validation/test splits for time series"""
        logger.info("\n📊 Creating Temporal Data Splits")
        logger.info("-" * 30)
        
        try:
            # Ensure data is sorted by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Prepare features and target
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Remove rows where target is missing
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
            timestamps = df['timestamp'][valid_indices] if 'timestamp' in df.columns else None
            
            logger.info(f"📊 Dataset for splitting: {len(X)} samples")
            
            # Check minimum samples requirement
            min_samples = self.config["min_train_samples"]
            if len(X) < min_samples:
                logger.error(f"❌ Insufficient samples: {len(X)} < {min_samples}")
                return {}
            
            # Calculate split indices for temporal split
            test_size = self.config["test_size"]
            val_size = self.config["validation_size"]
            
            # Temporal split: train on early data, validate on middle, test on recent
            n_total = len(X)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)
            n_train = n_total - n_test - n_val
            
            logger.info(f"📊 Split sizes:")
            logger.info(f"   Train: {n_train} samples ({n_train/n_total:.1%})")
            logger.info(f"   Validation: {n_val} samples ({n_val/n_total:.1%})")
            logger.info(f"   Test: {n_test} samples ({n_test/n_total:.1%})")
            
            # Create temporal splits
            train_end = n_train
            val_end = n_train + n_val
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
            
            X_test = X.iloc[val_end:]
            y_test = y.iloc[val_end:]
            
            # Log temporal ranges if timestamps available
            if timestamps is not None:
                train_timestamps = timestamps.iloc[:train_end]
                val_timestamps = timestamps.iloc[train_end:val_end]
                test_timestamps = timestamps.iloc[val_end:]
                
                logger.info(f"📅 Temporal ranges:")
                logger.info(f"   Train: {train_timestamps.min()} to {train_timestamps.max()}")
                logger.info(f"   Validation: {val_timestamps.min()} to {val_timestamps.max()}")
                logger.info(f"   Test: {test_timestamps.min()} to {test_timestamps.max()}")
            
            # Feature scaling
            scaler_method = self.config["preprocessing"]["scaling_method"]
            logger.info(f"⚖️ Applying {scaler_method} scaling...")
            
            if scaler_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            # Fit scaler on training data only
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrames
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
            
            splits = {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled, 
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'target_column': target_column
            }
            
            # Store split information
            self.preparation_results["splits"] = {
                "strategy": "temporal",
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "train_ratio": n_train / n_total,
                "val_ratio": n_val / n_total,
                "test_ratio": n_test / n_total,
                "scaling_method": scaler_method,
                "feature_count": len(feature_columns)
            }
            
            logger.info("✅ Temporal splits created successfully")
            
            return splits
            
        except Exception as e:
            logger.error(f"❌ Error creating splits: {str(e)}")
            return {}

    def setup_time_series_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Setup time series cross-validation for hyperparameter tuning"""
        logger.info("\n⏰ Setting Up Time Series Validation")
        logger.info("-" * 35)
        
        try:
            # Time series cross-validation setup
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Test the splits
            split_info = []
            for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                split_info.append({
                    "fold": i + 1,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "train_end_idx": train_idx[-1],
                    "val_start_idx": val_idx[0]
                })
            
            logger.info(f"📊 Time Series CV Setup:")
            logger.info(f"   Splits: {n_splits}")
            logger.info(f"   Average train size: {np.mean([s['train_size'] for s in split_info]):.0f}")
            logger.info(f"   Average val size: {np.mean([s['val_size'] for s in split_info]):.0f}")
            
            validation_setup = {
                "cv_strategy": "TimeSeriesSplit",
                "n_splits": n_splits,
                "split_info": split_info,
                "cv_object": tscv
            }
            
            self.preparation_results["validation_setup"] = {
                "cv_strategy": "TimeSeriesSplit",
                "n_splits": n_splits,
                "split_details": split_info
            }
            
            logger.info("✅ Time series validation setup completed")
            
            return validation_setup
            
        except Exception as e:
            logger.error(f"❌ Error setting up validation: {str(e)}")
            return {}

    def save_prepared_datasets(self, splits: Dict, cv_setup: Dict) -> bool:
        """Save prepared datasets and metadata"""
        logger.info("\n💾 Saving Prepared Datasets")
        logger.info("-" * 27)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save training data splits
            datasets = {
                'train': {
                    'X': splits['X_train'],
                    'y': splits['y_train']
                },
                'validation': {
                    'X': splits['X_val'],
                    'y': splits['y_val']
                },
                'test': {
                    'X': splits['X_test'],
                    'y': splits['y_test']
                }
            }
            
            # Save each dataset
            for split_name, data in datasets.items():
                X_file = os.path.join(self.datasets_dir, f"X_{split_name}.csv")
                y_file = os.path.join(self.datasets_dir, f"y_{split_name}.csv")
                
                data['X'].to_csv(X_file, index=False)
                data['y'].to_csv(y_file, index=False)
                
                logger.info(f"   {split_name}: {len(data['X'])} samples")
            
            # Save scaler
            import joblib
            scaler_file = os.path.join(self.datasets_dir, "feature_scaler.pkl")
            joblib.dump(splits['scaler'], scaler_file)
            
            # Save feature information
            feature_info = {
                "feature_columns": splits['feature_columns'],
                "target_column": splits['target_column'],
                "scaler_file": scaler_file,
                "n_features": len(splits['feature_columns'])
            }
            
            feature_info_file = os.path.join(self.datasets_dir, "feature_info.json")
            with open(feature_info_file, 'w') as f:
                json.dump(feature_info, f, indent=4)
            
            # Save complete preparation metadata
            self.preparation_results["output_files"] = {
                "datasets_directory": self.datasets_dir,
                "feature_scaler": scaler_file,
                "feature_info": feature_info_file,
                "cv_setup": cv_setup.get("cv_strategy", "TimeSeriesSplit")
            }
            
            metadata_file = os.path.join(self.metadata_dir, f"preparation_metadata_{timestamp}.json")
            with open(metadata_file, 'w') as f:
                # Remove non-serializable objects before saving
                metadata_copy = self.preparation_results.copy()
                json.dump(metadata_copy, f, indent=4)
            
            # Save latest metadata
            latest_metadata_file = os.path.join(self.metadata_dir, "latest_preparation.json")
            with open(latest_metadata_file, 'w') as f:
                json.dump(metadata_copy, f, indent=4)
            
            logger.info(f"✅ Datasets saved to: {self.datasets_dir}")
            logger.info(f"📁 Metadata saved: {metadata_file}")
            logger.info(f"⚖️ Scaler saved: {scaler_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving datasets: {str(e)}")
            return False

    def run_dataset_preparation(self) -> bool:
        """Run complete dataset preparation process"""
        logger.info("\n🚀 STARTING DATASET PREPARATION")
        logger.info("=" * 40)
        
        try:
            # Step 1: Load training features
            df = self.load_training_features()
            if df is None:
                return False
            
            # Step 2: Identify target and features
            target_column, feature_columns = self.identify_target_and_features(df)
            if target_column is None:
                return False
            
            # Step 3: Preprocess features
            df_processed = self.preprocess_features(df, feature_columns)
            
            # Step 4: Create temporal splits
            splits = self.create_temporal_splits(df_processed, target_column, feature_columns)
            if not splits:
                return False
            
            # Step 5: Setup time series validation
            cv_setup = self.setup_time_series_validation(splits['X_train'], splits['y_train'])
            
            # Step 6: Save prepared datasets
            save_success = self.save_prepared_datasets(splits, cv_setup)
            if not save_success:
                return False
            
            logger.info("\n🎉 DATASET PREPARATION COMPLETED SUCCESSFULLY!")
            logger.info("✅ Training datasets ready for model training")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Dataset preparation failed: {str(e)}")
            return False

def main():
    """Main function for dataset preparation"""
    preparator = TrainingDatasetPreparator()
    success = preparator.run_dataset_preparation()
    
    if success:
        print("\n🎯 DATASET PREPARATION SUCCESS!")
        print("📊 Training datasets ready")
    else:
        print("\n❌ Dataset preparation failed")
        print("📋 Check logs and input data")

if __name__ == "__main__":
    main()
