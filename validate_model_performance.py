"""
Model Performance Validation System
===================================

This script validates trained models to ensure they meet quality thresholds
before deployment. Includes comprehensive performance analysis, regression
checks, and production readiness validation.

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
# Use headless backend for CI environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ModelPerformanceValidator:
    """Comprehensive model performance validation system"""
    
    def __init__(self):
        """Initialize model performance validator"""
        logger.info("✅ MODEL PERFORMANCE VALIDATION")
        logger.info("=" * 35)
        
        # Directories
        self.models_dir = "data_repositories/models"
        self.performance_dir = os.path.join(self.models_dir, "performance")
        self.validation_dir = os.path.join("data_repositories", "training", "validation")
        self.datasets_dir = os.path.join("data_repositories", "training", "datasets")
        
        # Create directories
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Validation thresholds
        self.thresholds = {
            "minimum_r2": 0.80,           # Minimum R² for production
            "maximum_mae": 15.0,          # Maximum MAE (AQI points)
            "maximum_mape": 25.0,         # Maximum MAPE (%)
            "minimum_improvement": 0.01,   # Minimum improvement over baseline
            "regression_tolerance": 0.05,  # Allowable performance regression
            "prediction_range": {
                "min_aqi": 0,
                "max_aqi": 500
            }
        }
        
        # Validation results
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_passed": False,
            "performance_metrics": {},
            "threshold_checks": {},
            "regression_analysis": {},
            "production_readiness": {},
            "recommendations": []
        }

    def load_champion_model(self) -> Optional[Dict]:
        """Load the current champion model and metadata"""
        logger.info("\n🏆 Loading Champion Model")
        logger.info("-" * 24)
        
        try:
            # Load champion model
            champion_file = os.path.join(self.models_dir, "trained", "champion_model.pkl")
            metadata_file = os.path.join(self.models_dir, "metadata", "champion_metadata.json")
            
            if not os.path.exists(champion_file):
                logger.error("❌ Champion model not found")
                return None
            
            model = joblib.load(champion_file)
            
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"✅ Champion model loaded: {metadata.get('model_type', 'Unknown')}")
            logger.info(f"   Training date: {metadata.get('training_date', 'Unknown')}")
            
            return {
                'model': model,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading champion model: {str(e)}")
            return None

    def load_test_dataset(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Load test dataset for validation"""
        logger.info("\n📊 Loading Test Dataset")
        logger.info("-" * 23)
        
        try:
            X_test_file = os.path.join(self.datasets_dir, "X_test.csv")
            y_test_file = os.path.join(self.datasets_dir, "y_test.csv")
            
            if not os.path.exists(X_test_file) or not os.path.exists(y_test_file):
                logger.error("❌ Test dataset files not found")
                return None, None
            
            X_test = pd.read_csv(X_test_file)
            y_test = pd.read_csv(y_test_file).iloc[:, 0]  # First column is target
            
            logger.info(f"✅ Test dataset loaded: {len(X_test)} samples")
            logger.info(f"   Features: {len(X_test.columns)}")
            logger.info(f"   Target range: {y_test.min():.1f} to {y_test.max():.1f}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"❌ Error loading test dataset: {str(e)}")
            return None, None

    def calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        logger.info("\n📈 Calculating Performance Metrics")
        logger.info("-" * 33)
        
        try:
            # Basic regression metrics
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Additional metrics
            try:
                mape = mean_absolute_percentage_error(y_true, y_pred)
            except:
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Custom AQI-specific metrics
            residuals = y_true - y_pred
            abs_residuals = np.abs(residuals)
            
            # Accuracy within different AQI ranges
            within_5_aqi = (abs_residuals <= 5).mean() * 100
            within_10_aqi = (abs_residuals <= 10).mean() * 100
            within_15_aqi = (abs_residuals <= 15).mean() * 100
            
            # Bias analysis
            mean_bias = np.mean(residuals)
            std_bias = np.std(residuals)
            
            # Prediction range validation
            pred_min, pred_max = y_pred.min(), y_pred.max()
            true_min, true_max = y_true.min(), y_true.max()
            
            metrics = {
                "r2_score": float(r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "mean_bias": float(mean_bias),
                "std_bias": float(std_bias),
                "accuracy_within_5_aqi": float(within_5_aqi),
                "accuracy_within_10_aqi": float(within_10_aqi),
                "accuracy_within_15_aqi": float(within_15_aqi),
                "prediction_range": {
                    "min_predicted": float(pred_min),
                    "max_predicted": float(pred_max),
                    "min_actual": float(true_min),
                    "max_actual": float(true_max)
                },
                "sample_count": len(y_true)
            }
            
            logger.info("📊 Performance Metrics:")
            logger.info(f"   R² Score: {r2:.4f}")
            logger.info(f"   RMSE: {rmse:.3f}")
            logger.info(f"   MAE: {mae:.3f}")
            logger.info(f"   MAPE: {mape:.1f}%")
            logger.info(f"   Within ±5 AQI: {within_5_aqi:.1f}%")
            logger.info(f"   Within ±10 AQI: {within_10_aqi:.1f}%")
            logger.info(f"   Mean Bias: {mean_bias:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {str(e)}")
            return {}

    def validate_against_thresholds(self, metrics: Dict) -> Dict:
        """Validate model performance against production thresholds"""
        logger.info("\n🎯 Validating Against Thresholds")
        logger.info("-" * 32)
        
        threshold_results = {}
        passed_checks = 0
        total_checks = 0
        
        # R² threshold check
        r2_check = metrics["r2_score"] >= self.thresholds["minimum_r2"]
        threshold_results["r2_threshold"] = {
            "passed": r2_check,
            "value": metrics["r2_score"],
            "threshold": self.thresholds["minimum_r2"],
            "message": f"R² {metrics['r2_score']:.4f} {'≥' if r2_check else '<'} {self.thresholds['minimum_r2']}"
        }
        passed_checks += r2_check
        total_checks += 1
        
        # MAE threshold check
        mae_check = metrics["mae"] <= self.thresholds["maximum_mae"]
        threshold_results["mae_threshold"] = {
            "passed": mae_check,
            "value": metrics["mae"],
            "threshold": self.thresholds["maximum_mae"],
            "message": f"MAE {metrics['mae']:.3f} {'≤' if mae_check else '>'} {self.thresholds['maximum_mae']}"
        }
        passed_checks += mae_check
        total_checks += 1
        
        # MAPE threshold check
        mape_check = metrics["mape"] <= self.thresholds["maximum_mape"]
        threshold_results["mape_threshold"] = {
            "passed": mape_check,
            "value": metrics["mape"],
            "threshold": self.thresholds["maximum_mape"],
            "message": f"MAPE {metrics['mape']:.1f}% {'≤' if mape_check else '>'} {self.thresholds['maximum_mape']}%"
        }
        passed_checks += mape_check
        total_checks += 1
        
        # Prediction range check
        pred_range = metrics["prediction_range"]
        range_check = (pred_range["min_predicted"] >= self.thresholds["prediction_range"]["min_aqi"] and
                      pred_range["max_predicted"] <= self.thresholds["prediction_range"]["max_aqi"])
        threshold_results["prediction_range"] = {
            "passed": range_check,
            "value": f"[{pred_range['min_predicted']:.1f}, {pred_range['max_predicted']:.1f}]",
            "threshold": f"[{self.thresholds['prediction_range']['min_aqi']}, {self.thresholds['prediction_range']['max_aqi']}]",
            "message": f"Predictions within valid AQI range: {'Yes' if range_check else 'No'}"
        }
        passed_checks += range_check
        total_checks += 1
        
        # Overall threshold validation
        all_passed = passed_checks == total_checks
        
        logger.info("🎯 Threshold Validation Results:")
        for check_name, result in threshold_results.items():
            status = "✅" if result["passed"] else "❌"
            logger.info(f"   {status} {result['message']}")
        
        logger.info(f"\n📊 Threshold Summary: {passed_checks}/{total_checks} checks passed")
        
        threshold_summary = {
            "checks": threshold_results,
            "passed_count": passed_checks,
            "total_count": total_checks,
            "all_passed": all_passed,
            "pass_rate": passed_checks / total_checks
        }
        
        return threshold_summary

    def analyze_regression_risk(self, current_metrics: Dict, champion_data: Dict) -> Dict:
        """Analyze if there's performance regression compared to previous champion"""
        logger.info("\n📉 Analyzing Regression Risk")
        logger.info("-" * 27)
        
        regression_analysis = {
            "has_baseline": False,
            "performance_changes": {},
            "regression_risk": "unknown",
            "recommendations": []
        }
        
        try:
            # Get baseline performance from previous champion
            baseline_performance = champion_data.get('metadata', {}).get('performance', {})
            
            if not baseline_performance:
                logger.info("ℹ️ No baseline performance found - first model deployment")
                regression_analysis["regression_risk"] = "none"
                return regression_analysis
            
            regression_analysis["has_baseline"] = True
            
            # Compare key metrics
            key_metrics = ['r2_score', 'mae', 'rmse', 'mape']
            
            for metric in key_metrics:
                if metric in baseline_performance and metric in current_metrics:
                    baseline_value = baseline_performance[metric]
                    current_value = current_metrics[metric]
                    
                    # Calculate change (positive is improvement for R², negative for errors)
                    if metric == 'r2_score':
                        change = current_value - baseline_value
                        improvement = change > 0
                    else:  # Error metrics (lower is better)
                        change = baseline_value - current_value
                        improvement = change > 0
                    
                    change_percent = (abs(change) / abs(baseline_value)) * 100 if baseline_value != 0 else 0
                    
                    regression_analysis["performance_changes"][metric] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "change": change,
                        "change_percent": change_percent,
                        "improvement": improvement
                    }
            
            # Assess overall regression risk
            r2_change = regression_analysis["performance_changes"].get('r2_score', {})
            mae_change = regression_analysis["performance_changes"].get('mae', {})
            
            # Check for significant regression
            regression_tolerance = self.thresholds["regression_tolerance"]
            
            significant_regression = False
            if r2_change:
                r2_regression = r2_change["change"] < -regression_tolerance
                if r2_regression:
                    significant_regression = True
                    regression_analysis["recommendations"].append(f"R² decreased by {abs(r2_change['change']):.3f}")
            
            if mae_change:
                mae_regression = mae_change["change"] < -0.5  # 0.5 AQI point regression threshold
                if mae_regression:
                    significant_regression = True
                    regression_analysis["recommendations"].append(f"MAE increased by {abs(mae_change['change']):.3f}")
            
            if significant_regression:
                regression_analysis["regression_risk"] = "high"
                regression_analysis["recommendations"].append("Review model training process")
                regression_analysis["recommendations"].append("Consider additional hyperparameter tuning")
            else:
                regression_analysis["regression_risk"] = "low"
                regression_analysis["recommendations"].append("Performance maintained or improved")
            
            # Log regression analysis
            logger.info("📊 Performance Comparison:")
            for metric, changes in regression_analysis["performance_changes"].items():
                direction = "↗️" if changes["improvement"] else "↘️"
                logger.info(f"   {metric}: {changes['baseline']:.3f} → {changes['current']:.3f} {direction}")
                logger.info(f"      Change: {changes['change']:+.3f} ({changes['change_percent']:+.1f}%)")
            
            logger.info(f"\n🚨 Regression Risk: {regression_analysis['regression_risk'].upper()}")
            
        except Exception as e:
            logger.error(f"❌ Error in regression analysis: {str(e)}")
            regression_analysis["regression_risk"] = "unknown"
        
        return regression_analysis

    def assess_production_readiness(self, metrics: Dict, threshold_results: Dict, 
                                  regression_analysis: Dict) -> Dict:
        """Assess overall production readiness"""
        logger.info("\n🚀 Assessing Production Readiness")
        logger.info("-" * 33)
        
        readiness_score = 0
        max_score = 10
        
        readiness_assessment = {
            "overall_score": 0,
            "max_score": max_score,
            "readiness_level": "not_ready",
            "blocking_issues": [],
            "warnings": [],
            "strengths": []
        }
        
        # Performance thresholds (4 points)
        if threshold_results["all_passed"]:
            readiness_score += 4
            readiness_assessment["strengths"].append("All performance thresholds met")
        else:
            failed_checks = threshold_results["total_count"] - threshold_results["passed_count"]
            readiness_assessment["blocking_issues"].append(f"{failed_checks} performance thresholds failed")
        
        # Regression risk (2 points)
        if regression_analysis["regression_risk"] == "low":
            readiness_score += 2
            readiness_assessment["strengths"].append("Low regression risk")
        elif regression_analysis["regression_risk"] == "high":
            readiness_assessment["warnings"].append("High regression risk detected")
        else:
            readiness_score += 1
            readiness_assessment["warnings"].append("Unknown regression risk")
        
        # Model accuracy (2 points)
        if metrics["accuracy_within_10_aqi"] >= 80:
            readiness_score += 2
            readiness_assessment["strengths"].append("High prediction accuracy")
        elif metrics["accuracy_within_10_aqi"] >= 70:
            readiness_score += 1
            readiness_assessment["warnings"].append("Moderate prediction accuracy")
        else:
            readiness_assessment["blocking_issues"].append("Low prediction accuracy")
        
        # Bias analysis (1 point)
        if abs(metrics["mean_bias"]) <= 2.0:  # Low bias
            readiness_score += 1
            readiness_assessment["strengths"].append("Low prediction bias")
        else:
            readiness_assessment["warnings"].append("Notable prediction bias detected")
        
        # Sample size adequacy (1 point)
        if metrics["sample_count"] >= 500:  # Adequate test set
            readiness_score += 1
            readiness_assessment["strengths"].append("Adequate validation sample size")
        else:
            readiness_assessment["warnings"].append("Small validation sample size")
        
        # Determine readiness level
        readiness_assessment["overall_score"] = readiness_score
        
        if readiness_score >= 8 and len(readiness_assessment["blocking_issues"]) == 0:
            readiness_assessment["readiness_level"] = "ready"
        elif readiness_score >= 6 and len(readiness_assessment["blocking_issues"]) == 0:
            readiness_assessment["readiness_level"] = "ready_with_monitoring"
        elif readiness_score >= 4:
            readiness_assessment["readiness_level"] = "needs_improvement"
        else:
            readiness_assessment["readiness_level"] = "not_ready"
        
        # Log readiness assessment
        logger.info(f"📊 Production Readiness Score: {readiness_score}/{max_score}")
        logger.info(f"🚦 Readiness Level: {readiness_assessment['readiness_level'].upper()}")
        
        if readiness_assessment["strengths"]:
            logger.info("✅ Strengths:")
            for strength in readiness_assessment["strengths"]:
                logger.info(f"   • {strength}")
        
        if readiness_assessment["warnings"]:
            logger.info("⚠️ Warnings:")
            for warning in readiness_assessment["warnings"]:
                logger.info(f"   • {warning}")
        
        if readiness_assessment["blocking_issues"]:
            logger.info("❌ Blocking Issues:")
            for issue in readiness_assessment["blocking_issues"]:
                logger.info(f"   • {issue}")
        
        return readiness_assessment

    def save_validation_results(self, validation_passed: bool) -> bool:
        """Save validation results and recommendations"""
        logger.info("\n💾 Saving Validation Results")
        logger.info("-" * 27)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Update final validation status
            self.validation_results["validation_passed"] = validation_passed
            
            # Generate recommendations based on results
            if validation_passed:
                self.validation_results["recommendations"] = [
                    "Model meets all production requirements",
                    "Deploy model to production",
                    "Monitor performance for first 24 hours",
                    "Set up automated performance tracking"
                ]
            else:
                self.validation_results["recommendations"] = [
                    "Model does not meet production requirements",
                    "Review training process and hyperparameters",
                    "Increase training data if available",
                    "Consider ensemble methods for better performance"
                ]
            
            # Save detailed validation results
            validation_file = os.path.join(self.validation_dir, f"validation_results_{timestamp}.json")
            with open(validation_file, 'w') as f:
                json.dump(self.validation_results, f, indent=4)
            
            # Save latest validation results
            latest_validation_file = os.path.join(self.validation_dir, "latest_validation.json")
            with open(latest_validation_file, 'w') as f:
                json.dump(self.validation_results, f, indent=4)
            
            # Save validation status for deployment pipeline
            status_file = "validation_status.txt"
            with open(status_file, 'w') as f:
                f.write("passed" if validation_passed else "failed")
            
            logger.info(f"✅ Validation results saved:")
            logger.info(f"   Detailed: {validation_file}")
            logger.info(f"   Latest: {latest_validation_file}")
            logger.info(f"   Status: {status_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving validation results: {str(e)}")
            return False

    def run_model_validation(self) -> bool:
        """Run complete model validation process"""
        logger.info("\n🚀 STARTING MODEL VALIDATION")
        logger.info("=" * 35)
        
        try:
            # Step 1: Load champion model
            champion_data = self.load_champion_model()
            if not champion_data:
                return False
            
            # Step 2: Load test dataset
            X_test, y_test = self.load_test_dataset()
            if X_test is None or y_test is None:
                return False
            
            # Step 3: Generate predictions
            logger.info("\n🔮 Generating Predictions")
            logger.info("-" * 23)
            y_pred = champion_data['model'].predict(X_test)
            logger.info(f"✅ Predictions generated for {len(y_test)} samples")
            
            # Step 4: Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred)
            if not metrics:
                return False
            
            self.validation_results["performance_metrics"] = metrics
            
            # Step 5: Validate against thresholds
            threshold_results = self.validate_against_thresholds(metrics)
            self.validation_results["threshold_checks"] = threshold_results
            
            # Step 6: Analyze regression risk
            regression_analysis = self.analyze_regression_risk(metrics, champion_data)
            self.validation_results["regression_analysis"] = regression_analysis
            
            # Step 7: Assess production readiness
            readiness = self.assess_production_readiness(metrics, threshold_results, regression_analysis)
            self.validation_results["production_readiness"] = readiness
            
            # Step 8: Make final validation decision
            validation_passed = (
                threshold_results["all_passed"] and
                regression_analysis["regression_risk"] in ["low", "none"] and
                readiness["readiness_level"] in ["ready", "ready_with_monitoring"]
            )
            
            # Step 9: Save validation results
            save_success = self.save_validation_results(validation_passed)
            if not save_success:
                return False
            
            # Final summary
            if validation_passed:
                logger.info("\n🎉 MODEL VALIDATION PASSED!")
                logger.info("✅ Model is ready for production deployment")
            else:
                logger.info("\n❌ MODEL VALIDATION FAILED!")
                logger.info("🔄 Model requires improvements before deployment")
            
            logger.info(f"📊 Validation Summary:")
            logger.info(f"   Performance Score: {readiness['overall_score']}/{readiness['max_score']}")
            logger.info(f"   Readiness Level: {readiness['readiness_level']}")
            logger.info(f"   Threshold Checks: {threshold_results['passed_count']}/{threshold_results['total_count']}")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"\n❌ Model validation failed: {str(e)}")
            return False

def main():
    """Main function for model validation"""
    validator = ModelPerformanceValidator()
    validation_passed = validator.run_model_validation()
    
    if validation_passed:
        print("\n🎯 MODEL VALIDATION SUCCESS!")
        print("✅ Model approved for production deployment")
    else:
        print("\n❌ Model validation failed")
        print("📋 Check validation results and improve model")

if __name__ == "__main__":
    main()
