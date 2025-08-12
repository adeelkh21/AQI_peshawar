"""
Champion/Challenger Model Training System
========================================

This script implements the champion/challenger framework for continuous model improvement:
- Trains multiple candidate models (challengers)
- Compares performance against current champion
- Selects best performing model as new champion
- Implements hyperparameter optimization
- Provides detailed performance comparison

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-optimize not available. Using grid search.")
    BAYESIAN_OPT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ChampionChallengerTrainer:
    """Champion/Challenger model training and selection system"""
    
    def __init__(self):
        """Initialize champion/challenger trainer"""
        logger.info("🏆 CHAMPION/CHALLENGER MODEL TRAINING")
        logger.info("=" * 42)
        
        # Directories
        self.training_dir = "data_repositories/training"
        self.datasets_dir = os.path.join(self.training_dir, "datasets")
        self.models_dir = "data_repositories/models"
        self.trained_dir = os.path.join(self.models_dir, "trained")
        self.metadata_dir = os.path.join(self.models_dir, "metadata")
        self.performance_dir = os.path.join(self.models_dir, "performance")
        
        # Create directories
        os.makedirs(self.trained_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'lightgbm': {
                'enabled': True,
                'priority': 1,
                'param_space': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'num_leaves': Integer(10, 100),
                    'min_child_samples': Integer(10, 100),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0)
                },
                'fixed_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'random_state': 42
                }
            },
            'xgboost': {
                'enabled': True,
                'priority': 2,
                'param_space': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'min_child_weight': Integer(1, 10)
                },
                'fixed_params': {
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'random_forest': {
                'enabled': True,
                'priority': 3,
                'param_space': {
                    'n_estimators': Integer(100, 300),
                    'max_depth': Integer(5, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Real(0.3, 1.0)
                },
                'fixed_params': {
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
        }
        
        # Training results
        self.training_results = {
            "timestamp": datetime.now().isoformat(),
            "champion_model": None,
            "challenger_models": {},
            "performance_comparison": {},
            "optimization_details": {}
        }

    def load_training_datasets(self) -> Optional[Dict]:
        """Load prepared training datasets"""
        logger.info("\n📊 Loading Training Datasets")
        logger.info("-" * 28)
        
        try:
            # Load training data
            datasets = {}
            for split in ['train', 'validation', 'test']:
                X_file = os.path.join(self.datasets_dir, f"X_{split}.csv")
                y_file = os.path.join(self.datasets_dir, f"y_{split}.csv")
                
                if not os.path.exists(X_file) or not os.path.exists(y_file):
                    logger.error(f"❌ Missing {split} dataset files")
                    return None
                
                X = pd.read_csv(X_file)
                y = pd.read_csv(y_file).iloc[:, 0]  # First column is target
                
                datasets[f'X_{split}'] = X
                datasets[f'y_{split}'] = y
                
                logger.info(f"   {split}: {len(X)} samples, {len(X.columns)} features")
            
            # Load feature info and scaler
            feature_info_file = os.path.join(self.datasets_dir, "feature_info.json")
            scaler_file = os.path.join(self.datasets_dir, "feature_scaler.pkl")
            
            if os.path.exists(feature_info_file):
                with open(feature_info_file, 'r') as f:
                    datasets['feature_info'] = json.load(f)
            
            if os.path.exists(scaler_file):
                datasets['scaler'] = joblib.load(scaler_file)
            
            logger.info("✅ Training datasets loaded successfully")
            return datasets
            
        except Exception as e:
            logger.error(f"❌ Error loading datasets: {str(e)}")
            return None

    def load_current_champion(self) -> Optional[Dict]:
        """Load current champion model if exists"""
        logger.info("\n🏆 Loading Current Champion Model")
        logger.info("-" * 32)
        
        try:
            # Look for existing champion model
            champion_file = os.path.join(self.trained_dir, "champion_model.pkl")
            champion_metadata_file = os.path.join(self.metadata_dir, "champion_metadata.json")
            
            if not os.path.exists(champion_file):
                logger.info("ℹ️ No existing champion model found - first training")
                return None
            
            # Load champion model and metadata
            champion_model = joblib.load(champion_file)
            
            champion_metadata = {}
            if os.path.exists(champion_metadata_file):
                with open(champion_metadata_file, 'r') as f:
                    champion_metadata = json.load(f)
            
            logger.info(f"✅ Current champion loaded: {champion_metadata.get('model_type', 'Unknown')}")
            logger.info(f"   Performance: R² = {champion_metadata.get('performance', {}).get('r2_score', 'Unknown')}")
            logger.info(f"   Trained: {champion_metadata.get('training_date', 'Unknown')}")
            
            return {
                'model': champion_model,
                'metadata': champion_metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading champion: {str(e)}")
            return None

    def train_challenger_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                             X_val: pd.DataFrame, y_val: pd.Series) -> Optional[Dict]:
        """Train a challenger model with hyperparameter optimization"""
        logger.info(f"\n🎯 Training {model_name.upper()} Challenger")
        logger.info(f"-" * (15 + len(model_name)))
        
        try:
            config = self.model_configs[model_name]
            
            if not config['enabled']:
                logger.info(f"⚠️ {model_name} is disabled - skipping")
                return None
            
            # Create base model
            if model_name == 'lightgbm':
                base_model = lgb.LGBMRegressor(**config['fixed_params'])
            elif model_name == 'xgboost':
                base_model = xgb.XGBRegressor(**config['fixed_params'])
            elif model_name == 'random_forest':
                base_model = RandomForestRegressor(**config['fixed_params'])
            else:
                logger.error(f"❌ Unknown model type: {model_name}")
                return None
            
            # Hyperparameter optimization
            if BAYESIAN_OPT_AVAILABLE and len(X_train) > 1000:  # Use Bayesian optimization for larger datasets
                logger.info("🔧 Running Bayesian hyperparameter optimization...")
                
                # Set up time series cross-validation
                cv = TimeSeriesSplit(n_splits=3)
                
                # Bayesian search
                opt = BayesSearchCV(
                    base_model,
                    config['param_space'],
                    n_iter=20,  # Reduced for speed
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,  # Single job to avoid memory issues
                    random_state=42
                )
                
                opt.fit(X_train, y_train)
                best_model = opt.best_estimator_
                best_params = opt.best_params_
                cv_score = -opt.best_score_
                
                logger.info(f"   Best CV RMSE: {np.sqrt(cv_score):.3f}")
                
            else:
                # Simplified optimization with default params
                logger.info("🔧 Training with optimized default parameters...")
                
                # Use reasonable default parameters
                if model_name == 'lightgbm':
                    params = {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'num_leaves': 31,
                        'min_child_samples': 20,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }
                elif model_name == 'xgboost':
                    params = {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'min_child_weight': 3
                    }
                else:  # random_forest
                    params = {
                        'n_estimators': 150,
                        'max_depth': 10,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'max_features': 0.8
                    }
                
                # Combine with fixed params
                all_params = {**config['fixed_params'], **params}
                
                if model_name == 'lightgbm':
                    best_model = lgb.LGBMRegressor(**all_params)
                elif model_name == 'xgboost':
                    best_model = xgb.XGBRegressor(**all_params)
                else:
                    best_model = RandomForestRegressor(**all_params)
                
                best_params = params
                cv_score = None
            
            # Train final model
            logger.info("🔧 Training final model...")
            best_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics
            val_r2 = r2_score(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_.tolist()
            
            performance = {
                'r2_score': val_r2,
                'rmse': val_rmse,
                'mae': val_mae,
                'cv_rmse': np.sqrt(cv_score) if cv_score else None
            }
            
            logger.info(f"✅ {model_name} performance:")
            logger.info(f"   Validation R²: {val_r2:.4f}")
            logger.info(f"   Validation RMSE: {val_rmse:.3f}")
            logger.info(f"   Validation MAE: {val_mae:.3f}")
            
            challenger_result = {
                'model': best_model,
                'model_type': model_name,
                'parameters': best_params,
                'performance': performance,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            return challenger_result
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {str(e)}")
            return None

    def train_all_challengers(self, datasets: Dict) -> Dict:
        """Train all challenger models"""
        logger.info("\n🎯 TRAINING ALL CHALLENGER MODELS")
        logger.info("=" * 40)
        
        challengers = {}
        
        X_train = datasets['X_train']
        y_train = datasets['y_train']
        X_val = datasets['X_validation']
        y_val = datasets['y_validation']
        
        # Sort models by priority
        sorted_models = sorted(
            self.model_configs.items(),
            key=lambda x: x[1]['priority']
        )
        
        for model_name, config in sorted_models:
            if config['enabled']:
                logger.info(f"\n{'='*20} {model_name.upper()} {'='*20}")
                
                challenger = self.train_challenger_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                if challenger:
                    challengers[model_name] = challenger
                    logger.info(f"✅ {model_name} challenger trained successfully")
                else:
                    logger.error(f"❌ {model_name} challenger training failed")
        
        logger.info(f"\n📊 Challenger Training Summary:")
        logger.info(f"   Total challengers: {len(challengers)}")
        for name, challenger in challengers.items():
            perf = challenger['performance']
            logger.info(f"   {name}: R² = {perf['r2_score']:.4f}, RMSE = {perf['rmse']:.3f}")
        
        self.training_results["challenger_models"] = {
            name: {
                'model_type': challenger['model_type'],
                'performance': challenger['performance'],
                'parameters': challenger['parameters'],
                'training_samples': challenger['training_samples']
            }
            for name, challenger in challengers.items()
        }
        
        return challengers

    def select_champion(self, challengers: Dict, current_champion: Optional[Dict], 
                       datasets: Dict) -> Tuple[str, Dict]:
        """Select the best performing model as new champion"""
        logger.info("\n🏆 SELECTING NEW CHAMPION")
        logger.info("=" * 30)
        
        try:
            if not challengers:
                logger.error("❌ No challengers available for selection")
                return None, {}
            
            # Evaluate all models on test set
            X_test = datasets['X_test']
            y_test = datasets['y_test']
            
            model_performances = {}
            
            # Evaluate current champion if exists
            if current_champion:
                try:
                    champion_pred = current_champion['model'].predict(X_test)
                    champion_r2 = r2_score(y_test, champion_pred)
                    champion_rmse = np.sqrt(mean_squared_error(y_test, champion_pred))
                    champion_mae = mean_absolute_error(y_test, champion_pred)
                    
                    model_performances['current_champion'] = {
                        'r2_score': champion_r2,
                        'rmse': champion_rmse,
                        'mae': champion_mae,
                        'model_type': current_champion['metadata'].get('model_type', 'Unknown'),
                        'is_champion': True
                    }
                    
                    logger.info(f"🏆 Current Champion Performance:")
                    logger.info(f"   R²: {champion_r2:.4f}, RMSE: {champion_rmse:.3f}, MAE: {champion_mae:.3f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Could not evaluate current champion: {str(e)}")
            
            # Evaluate challengers
            logger.info(f"\n🎯 Challenger Performance on Test Set:")
            for name, challenger in challengers.items():
                try:
                    challenger_pred = challenger['model'].predict(X_test)
                    challenger_r2 = r2_score(y_test, challenger_pred)
                    challenger_rmse = np.sqrt(mean_squared_error(y_test, challenger_pred))
                    challenger_mae = mean_absolute_error(y_test, challenger_pred)
                    
                    model_performances[name] = {
                        'r2_score': challenger_r2,
                        'rmse': challenger_rmse,
                        'mae': challenger_mae,
                        'model_type': challenger['model_type'],
                        'is_champion': False,
                        'model_object': challenger['model'],
                        'parameters': challenger['parameters'],
                        'feature_importance': challenger.get('feature_importance')
                    }
                    
                    logger.info(f"   {name}: R² = {challenger_r2:.4f}, RMSE = {challenger_rmse:.3f}, MAE = {challenger_mae:.3f}")
                
                except Exception as e:
                    logger.error(f"❌ Error evaluating {name}: {str(e)}")
            
            # Select best model based on R² score
            best_model_name = max(model_performances.keys(), 
                                key=lambda k: model_performances[k]['r2_score'])
            
            best_performance = model_performances[best_model_name]
            
            logger.info(f"\n🏅 CHAMPION SELECTION RESULT:")
            logger.info(f"   Winner: {best_model_name}")
            logger.info(f"   Performance: R² = {best_performance['r2_score']:.4f}")
            
            # Check if there's significant improvement
            improvement_threshold = 0.01  # 1% improvement threshold
            is_significant_improvement = True
            
            if current_champion and 'current_champion' in model_performances:
                current_r2 = model_performances['current_champion']['r2_score']
                new_r2 = best_performance['r2_score']
                improvement = new_r2 - current_r2
                
                logger.info(f"   Improvement: {improvement:+.4f} ({improvement/current_r2*100:+.1f}%)")
                
                if improvement < improvement_threshold:
                    logger.warning(f"⚠️ Improvement below threshold ({improvement_threshold:.3f})")
                    is_significant_improvement = False
                else:
                    logger.info(f"✅ Significant improvement detected")
            
            # Store performance comparison
            self.training_results["performance_comparison"] = model_performances
            self.training_results["champion_model"] = {
                "name": best_model_name,
                "performance": best_performance,
                "is_new_champion": best_model_name != 'current_champion',
                "significant_improvement": is_significant_improvement
            }
            
            # Return the winning challenger (or current champion if it wins)
            if best_model_name == 'current_champion':
                return 'current_champion', current_champion
            else:
                return best_model_name, challengers[best_model_name]
            
        except Exception as e:
            logger.error(f"❌ Error in champion selection: {str(e)}")
            return None, {}

    def save_champion_model(self, champion_name: str, champion_data: Dict, datasets: Dict) -> bool:
        """Save the new champion model and metadata"""
        logger.info(f"\n💾 Saving Champion Model: {champion_name}")
        logger.info("-" * (25 + len(champion_name)))
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare champion metadata
            champion_metadata = {
                "model_type": champion_data.get('model_type', champion_name),
                "training_date": datetime.now().isoformat(),
                "champion_name": champion_name,
                "performance": champion_data.get('performance', {}),
                "parameters": champion_data.get('parameters', {}),
                "feature_info": datasets.get('feature_info', {}),
                "training_samples": champion_data.get('training_samples', 0),
                "model_version": timestamp,
                "training_results": self.training_results
            }
            
            # Save champion model
            if champion_name == 'current_champion':
                # Keep current champion - just update metadata
                champion_model = champion_data['model']
                logger.info("ℹ️ Keeping current champion model")
            else:
                # Save new challenger as champion
                champion_model = champion_data['model']
                logger.info(f"🆕 Promoting {champion_name} to champion")
            
            # Save model file
            champion_file = os.path.join(self.trained_dir, "champion_model.pkl")
            joblib.dump(champion_model, champion_file)
            
            # Save timestamped backup
            backup_file = os.path.join(self.trained_dir, f"champion_model_{timestamp}.pkl")
            joblib.dump(champion_model, backup_file)
            
            # Save metadata
            metadata_file = os.path.join(self.metadata_dir, "champion_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(champion_metadata, f, indent=4)
            
            # Save timestamped metadata
            timestamped_metadata_file = os.path.join(self.metadata_dir, f"champion_metadata_{timestamp}.json")
            with open(timestamped_metadata_file, 'w') as f:
                json.dump(champion_metadata, f, indent=4)
            
            # Save performance results
            performance_file = os.path.join(self.performance_dir, f"training_performance_{timestamp}.json")
            with open(performance_file, 'w') as f:
                json.dump(self.training_results, f, indent=4)
            
            # Save latest performance
            latest_performance_file = os.path.join(self.performance_dir, "latest_performance.json")
            with open(latest_performance_file, 'w') as f:
                json.dump(self.training_results, f, indent=4)
            
            logger.info("✅ Champion model saved successfully:")
            logger.info(f"   Model: {champion_file}")
            logger.info(f"   Metadata: {metadata_file}")
            logger.info(f"   Performance: {performance_file}")
            logger.info(f"   Backup: {backup_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving champion model: {str(e)}")
            return False

    def run_champion_challenger_training(self) -> bool:
        """Run complete champion/challenger training process"""
        logger.info("\n🚀 STARTING CHAMPION/CHALLENGER TRAINING")
        logger.info("=" * 50)
        
        try:
            # Step 1: Load training datasets
            datasets = self.load_training_datasets()
            if not datasets:
                return False
            
            # Step 2: Load current champion
            current_champion = self.load_current_champion()
            
            # Step 3: Train challenger models
            challengers = self.train_all_challengers(datasets)
            if not challengers:
                logger.error("❌ No challengers trained successfully")
                return False
            
            # Step 4: Select new champion
            champion_name, champion_data = self.select_champion(challengers, current_champion, datasets)
            if not champion_data:
                logger.error("❌ Champion selection failed")
                return False
            
            # Step 5: Save champion model
            save_success = self.save_champion_model(champion_name, champion_data, datasets)
            if not save_success:
                return False
            
            logger.info("\n🎉 CHAMPION/CHALLENGER TRAINING COMPLETED!")
            logger.info(f"🏆 New Champion: {champion_name}")
            
            # Log final summary
            champion_performance = self.training_results["champion_model"]["performance"]
            logger.info(f"📊 Champion Performance:")
            logger.info(f"   R² Score: {champion_performance['r2_score']:.4f}")
            logger.info(f"   RMSE: {champion_performance['rmse']:.3f}")
            logger.info(f"   MAE: {champion_performance['mae']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Champion/Challenger training failed: {str(e)}")
            return False

def main():
    """Main function for champion/challenger training"""
    trainer = ChampionChallengerTrainer()
    success = trainer.run_champion_challenger_training()
    
    if success:
        print("\n🎯 CHAMPION/CHALLENGER TRAINING SUCCESS!")
        print("🏆 New champion model selected and saved")
    else:
        print("\n❌ Champion/Challenger training failed")
        print("📋 Check logs and training data")

if __name__ == "__main__":
    main()
