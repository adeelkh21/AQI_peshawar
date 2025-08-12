"""
Continuous Model Retraining Trigger System
==========================================

This script determines whether model retraining should be triggered based on:
- Model performance degradation
- Data freshness and volume
- Feature drift detection
- Time since last retraining
- Manual override conditions

Author: Data Science Team
Date: August 12, 2025
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainingTrigger:
    """Intelligent model retraining trigger system"""
    
    def __init__(self):
        """Initialize retraining trigger system"""
        logger.info("🔍 MODEL RETRAINING TRIGGER SYSTEM")
        logger.info("=" * 40)
        
        # Configuration
        self.config = {
            "performance_thresholds": {
                "r2_degradation_percent": 5.0,  # 5% performance drop triggers retraining
                "mae_increase_percent": 10.0,   # 10% MAE increase triggers retraining
                "min_r2_threshold": 0.85        # Absolute minimum R² before forced retraining
            },
            "data_requirements": {
                "min_new_records": 72,          # Minimum 72 hours (3 days) of new data
                "max_hours_since_training": 168, # Force retrain after 1 week
                "min_total_records": 1000       # Minimum records for reliable training
            },
            "drift_thresholds": {
                "feature_drift_percent": 15.0,  # 15% feature drift triggers retraining
                "data_quality_threshold": 0.90  # 90% data quality required
            },
            "override_conditions": {
                "force_retrain_env": "FORCE_RETRAIN",
                "manual_trigger_file": "force_retrain.flag"
            }
        }
        
        # Directories
        self.models_dir = "data_repositories/models"
        self.features_dir = "data_repositories/features"
        self.reports_dir = "data_repositories/training/reports"
        
        # Create directories
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.trigger_results = {
            "timestamp": datetime.now().isoformat(),
            "decision": "no_retrain",
            "reasons": [],
            "metrics": {},
            "next_check": None
        }

    def check_override_conditions(self) -> Tuple[bool, List[str]]:
        """Check for manual override conditions"""
        logger.info("\n🔧 Checking Override Conditions")
        logger.info("-" * 32)
        
        override_reasons = []
        
        # Check environment variable
        force_env = os.getenv(self.config["override_conditions"]["force_retrain_env"], "false").lower()
        if force_env in ["true", "1", "yes"]:
            override_reasons.append("Environment variable FORCE_RETRAIN is set")
            logger.info("✅ Force retrain environment variable detected")
        
        # Check manual trigger file
        trigger_file = self.config["override_conditions"]["manual_trigger_file"]
        if os.path.exists(trigger_file):
            override_reasons.append("Manual trigger file found")
            logger.info(f"✅ Manual trigger file detected: {trigger_file}")
            # Remove the file after detection
            try:
                os.remove(trigger_file)
                logger.info("🗑️ Manual trigger file removed")
            except:
                pass
        
        has_override = len(override_reasons) > 0
        
        if has_override:
            logger.info(f"🚨 Override conditions detected: {len(override_reasons)} reasons")
        else:
            logger.info("ℹ️ No override conditions found")
        
        return has_override, override_reasons

    def check_model_performance(self) -> Tuple[bool, List[str], Dict]:
        """Check current model performance against thresholds"""
        logger.info("\n📊 Checking Model Performance")
        logger.info("-" * 30)
        
        performance_reasons = []
        performance_metrics = {}
        
        try:
            # Find latest performance file
            performance_dir = os.path.join(self.models_dir, "performance")
            if not os.path.exists(performance_dir):
                performance_reasons.append("No performance history found - first training")
                logger.info("ℹ️ No performance history - triggering first training")
                return True, performance_reasons, {}
            
            performance_files = [f for f in os.listdir(performance_dir) if f.endswith('.json')]
            if not performance_files:
                performance_reasons.append("No performance files found")
                logger.info("ℹ️ No performance files - triggering training")
                return True, performance_reasons, {}
            
            # Load latest performance
            latest_file = sorted(performance_files)[-1]
            with open(os.path.join(performance_dir, latest_file), 'r') as f:
                latest_performance = json.load(f)
            
            # Extract current metrics
            current_metrics = latest_performance.get('validation_metrics', {})
            current_r2 = current_metrics.get('r2_score', 0)
            current_mae = current_metrics.get('mae', float('inf'))
            
            performance_metrics = {
                "current_r2": current_r2,
                "current_mae": current_mae,
                "performance_file": latest_file
            }
            
            logger.info(f"📈 Current Model Performance:")
            logger.info(f"   R² Score: {current_r2:.4f}")
            logger.info(f"   MAE: {current_mae:.2f}")
            
            # Check absolute thresholds
            min_r2 = self.config["performance_thresholds"]["min_r2_threshold"]
            if current_r2 < min_r2:
                performance_reasons.append(f"R² below minimum threshold ({current_r2:.3f} < {min_r2})")
                logger.warning(f"⚠️ R² below minimum threshold: {current_r2:.3f} < {min_r2}")
            
            # Check historical degradation if we have baseline
            if 'baseline_metrics' in latest_performance:
                baseline = latest_performance['baseline_metrics']
                baseline_r2 = baseline.get('r2_score', current_r2)
                baseline_mae = baseline.get('mae', current_mae)
                
                # R² degradation check
                r2_degradation = ((baseline_r2 - current_r2) / baseline_r2) * 100
                r2_threshold = self.config["performance_thresholds"]["r2_degradation_percent"]
                
                if r2_degradation > r2_threshold:
                    performance_reasons.append(f"R² degraded by {r2_degradation:.1f}% (threshold: {r2_threshold}%)")
                    logger.warning(f"⚠️ Performance degradation detected: {r2_degradation:.1f}%")
                
                # MAE increase check
                mae_increase = ((current_mae - baseline_mae) / baseline_mae) * 100
                mae_threshold = self.config["performance_thresholds"]["mae_increase_percent"]
                
                if mae_increase > mae_threshold:
                    performance_reasons.append(f"MAE increased by {mae_increase:.1f}% (threshold: {mae_threshold}%)")
                    logger.warning(f"⚠️ MAE increase detected: {mae_increase:.1f}%")
                
                performance_metrics.update({
                    "baseline_r2": baseline_r2,
                    "baseline_mae": baseline_mae,
                    "r2_degradation_percent": r2_degradation,
                    "mae_increase_percent": mae_increase
                })
            
            needs_retraining = len(performance_reasons) > 0
            
            if needs_retraining:
                logger.warning(f"⚠️ Performance issues detected: {len(performance_reasons)} reasons")
            else:
                logger.info("✅ Model performance within acceptable thresholds")
            
            return needs_retraining, performance_reasons, performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Error checking performance: {str(e)}")
            performance_reasons.append(f"Performance check failed: {str(e)}")
            return True, performance_reasons, {}

    def check_data_freshness(self) -> Tuple[bool, List[str], Dict]:
        """Check data freshness and volume requirements"""
        logger.info("\n📅 Checking Data Freshness")
        logger.info("-" * 26)
        
        data_reasons = []
        data_metrics = {}
        
        try:
            # Check latest feature data
            feature_file = os.path.join(self.features_dir, "engineered", "realtime_features.csv")
            
            if not os.path.exists(feature_file):
                data_reasons.append("No feature data available")
                logger.warning("⚠️ No feature data found")
                return False, data_reasons, {}
            
            # Load and analyze feature data
            df = pd.read_csv(feature_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Data freshness metrics
            latest_timestamp = df['timestamp'].max()
            hours_since_latest = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            data_metrics = {
                "total_records": len(df),
                "latest_timestamp": latest_timestamp.isoformat(),
                "hours_since_latest": hours_since_latest,
                "date_range_hours": (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            }
            
            logger.info(f"📊 Data Status:")
            logger.info(f"   Total records: {len(df):,}")
            logger.info(f"   Latest data: {latest_timestamp}")
            logger.info(f"   Hours since latest: {hours_since_latest:.1f}")
            
            # Check minimum total records
            min_records = self.config["data_requirements"]["min_total_records"]
            if len(df) < min_records:
                data_reasons.append(f"Insufficient total records ({len(df)} < {min_records})")
                logger.warning(f"⚠️ Insufficient data: {len(df)} < {min_records}")
            
            # Check data freshness (don't train on very stale data)
            if hours_since_latest > 12:  # More than 12 hours old
                data_reasons.append(f"Data is too stale ({hours_since_latest:.1f} hours old)")
                logger.warning(f"⚠️ Stale data: {hours_since_latest:.1f} hours old")
                return False, data_reasons, data_metrics
            
            # Check for new data since last training
            last_training_file = os.path.join(self.reports_dir, "last_training_timestamp.txt")
            
            if os.path.exists(last_training_file):
                with open(last_training_file, 'r') as f:
                    last_training_str = f.read().strip()
                
                try:
                    last_training = datetime.fromisoformat(last_training_str)
                    hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
                    
                    data_metrics["hours_since_last_training"] = hours_since_training
                    
                    logger.info(f"   Hours since last training: {hours_since_training:.1f}")
                    
                    # Force retrain if too much time has passed
                    max_hours = self.config["data_requirements"]["max_hours_since_training"]
                    if hours_since_training > max_hours:
                        data_reasons.append(f"Maximum time exceeded since last training ({hours_since_training:.1f}h > {max_hours}h)")
                        logger.warning(f"⚠️ Training overdue: {hours_since_training:.1f}h > {max_hours}h")
                    
                    # Check for sufficient new data
                    new_data = df[df['timestamp'] > last_training]
                    min_new = self.config["data_requirements"]["min_new_records"]
                    
                    data_metrics["new_records_since_training"] = len(new_data)
                    
                    if len(new_data) >= min_new:
                        data_reasons.append(f"Sufficient new data available ({len(new_data)} >= {min_new} records)")
                        logger.info(f"✅ Sufficient new data: {len(new_data)} records")
                    else:
                        logger.info(f"ℹ️ Insufficient new data: {len(new_data)} < {min_new} records")
                
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing last training time: {str(e)}")
                    data_reasons.append("Cannot determine last training time - triggering training")
            else:
                data_reasons.append("No previous training record found - first training")
                logger.info("ℹ️ No training history - triggering first training")
            
            needs_training = len(data_reasons) > 0
            
            if needs_training:
                logger.info(f"✅ Data conditions support retraining: {len(data_reasons)} reasons")
            else:
                logger.info("ℹ️ Data conditions do not require immediate retraining")
            
            return needs_training, data_reasons, data_metrics
            
        except Exception as e:
            logger.error(f"❌ Error checking data freshness: {str(e)}")
            data_reasons.append(f"Data check failed: {str(e)}")
            return False, data_reasons, {}

    def check_feature_drift(self) -> Tuple[bool, List[str], Dict]:
        """Check for feature drift that might require retraining"""
        logger.info("\n🌊 Checking Feature Drift")
        logger.info("-" * 22)
        
        drift_reasons = []
        drift_metrics = {}
        
        try:
            # This is a simplified drift detection
            # In production, you'd have more sophisticated drift detection
            
            feature_file = os.path.join(self.features_dir, "engineered", "realtime_features.csv")
            
            if not os.path.exists(feature_file):
                logger.info("ℹ️ No feature data for drift analysis")
                return False, [], {}
            
            df = pd.read_csv(feature_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Simple drift detection based on recent data statistics
            recent_hours = 72  # Last 3 days
            cutoff_time = datetime.now() - timedelta(hours=recent_hours)
            recent_data = df[df['timestamp'] >= cutoff_time]
            
            if len(recent_data) < 50:  # Need minimum data for drift detection
                logger.info("ℹ️ Insufficient recent data for drift analysis")
                return False, [], {}
            
            # Analyze numerical columns for statistical drift
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if not col.startswith('timestamp')]
            
            high_variance_features = 0
            for col in numeric_cols[:20]:  # Check first 20 features to avoid timeout
                try:
                    recent_std = recent_data[col].std()
                    recent_mean = recent_data[col].mean()
                    
                    # High coefficient of variation indicates potential drift
                    if recent_mean != 0:
                        cv = recent_std / abs(recent_mean)
                        if cv > 2.0:  # High variability threshold
                            high_variance_features += 1
                except:
                    continue
            
            drift_metrics = {
                "recent_data_points": len(recent_data),
                "high_variance_features": high_variance_features,
                "total_features_checked": min(20, len(numeric_cols))
            }
            
            # Determine if drift is significant
            drift_threshold = self.config["drift_thresholds"]["feature_drift_percent"] / 100
            variance_ratio = high_variance_features / max(drift_metrics["total_features_checked"], 1)
            
            if variance_ratio > drift_threshold:
                drift_reasons.append(f"High feature variance detected ({high_variance_features}/{drift_metrics['total_features_checked']} features)")
                logger.warning(f"⚠️ Feature drift detected: {variance_ratio:.2%} variance ratio")
            
            logger.info(f"📊 Drift Analysis:")
            logger.info(f"   Recent data points: {len(recent_data)}")
            logger.info(f"   High variance features: {high_variance_features}/{drift_metrics['total_features_checked']}")
            logger.info(f"   Variance ratio: {variance_ratio:.2%}")
            
            has_drift = len(drift_reasons) > 0
            
            if has_drift:
                logger.warning(f"⚠️ Feature drift detected: {len(drift_reasons)} issues")
            else:
                logger.info("✅ No significant feature drift detected")
            
            return has_drift, drift_reasons, drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Error checking feature drift: {str(e)}")
            return False, [], {"error": str(e)}

    def make_retraining_decision(self) -> bool:
        """Make intelligent retraining decision based on all factors"""
        logger.info("\n🎯 MAKING RETRAINING DECISION")
        logger.info("=" * 35)
        
        # Check all conditions
        has_override, override_reasons = self.check_override_conditions()
        has_performance_issues, performance_reasons, performance_metrics = self.check_model_performance()
        has_data_conditions, data_reasons, data_metrics = self.check_data_freshness()
        has_drift, drift_reasons, drift_metrics = self.check_feature_drift()
        
        # Compile results
        all_reasons = []
        if has_override:
            all_reasons.extend([f"Override: {r}" for r in override_reasons])
        if has_performance_issues:
            all_reasons.extend([f"Performance: {r}" for r in performance_reasons])
        if has_data_conditions:
            all_reasons.extend([f"Data: {r}" for r in data_reasons])
        if has_drift:
            all_reasons.extend([f"Drift: {r}" for r in drift_reasons])
        
        # Decision logic
        should_retrain = False
        
        if has_override:
            should_retrain = True
            logger.info("🚨 Retraining triggered by OVERRIDE conditions")
        elif has_performance_issues:
            should_retrain = True
            logger.info("📉 Retraining triggered by PERFORMANCE degradation")
        elif has_data_conditions and (has_drift or len(data_reasons) > 1):
            should_retrain = True
            logger.info("📊 Retraining triggered by DATA conditions + drift")
        else:
            logger.info("ℹ️ No retraining required at this time")
        
        # Update trigger results
        self.trigger_results.update({
            "decision": "retrain" if should_retrain else "no_retrain",
            "reasons": all_reasons,
            "metrics": {
                "performance": performance_metrics,
                "data": data_metrics,
                "drift": drift_metrics
            },
            "next_check": (datetime.now() + timedelta(hours=3)).isoformat()
        })
        
        # Log decision summary
        logger.info(f"\n📋 DECISION SUMMARY:")
        logger.info(f"   Decision: {'RETRAIN' if should_retrain else 'NO RETRAIN'}")
        logger.info(f"   Total reasons: {len(all_reasons)}")
        logger.info(f"   Override conditions: {has_override}")
        logger.info(f"   Performance issues: {has_performance_issues}")
        logger.info(f"   Data conditions: {has_data_conditions}")
        logger.info(f"   Feature drift: {has_drift}")
        
        if all_reasons:
            logger.info(f"\n📝 Reasons:")
            for reason in all_reasons:
                logger.info(f"   - {reason}")
        
        return should_retrain

    def save_decision(self, decision: bool) -> None:
        """Save retraining decision and results"""
        try:
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.reports_dir, f"trigger_decision_{timestamp}.json")
            
            with open(results_file, 'w') as f:
                json.dump(self.trigger_results, f, indent=4)
            
            # Save simple decision for GitHub Actions
            with open("trigger_result.txt", 'w') as f:
                f.write("retrain" if decision else "no_retrain")
            
            logger.info(f"📁 Decision saved: {results_file}")
            logger.info(f"🎯 Result: {'RETRAIN' if decision else 'NO_RETRAIN'}")
            
        except Exception as e:
            logger.error(f"❌ Error saving decision: {str(e)}")

def main():
    """Main function for retraining trigger system"""
    trigger = ModelRetrainingTrigger()
    
    try:
        decision = trigger.make_retraining_decision()
        trigger.save_decision(decision)
        
        if decision:
            print("\n🚀 RETRAINING TRIGGERED!")
            print("📈 Model retraining will proceed")
        else:
            print("\n✅ NO RETRAINING NEEDED")
            print("📊 Model performance is acceptable")
        
    except Exception as e:
        logger.error(f"❌ Trigger system error: {str(e)}")
        # Default to no retrain on errors to prevent unnecessary training
        with open("trigger_result.txt", 'w') as f:
            f.write("no_retrain")
        print("\n❌ Trigger system error - defaulting to no retrain")

if __name__ == "__main__":
    main()
