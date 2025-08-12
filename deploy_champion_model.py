"""
Automated Model Deployment System
=================================

This script handles automated deployment of validated champion models with:
- Hot-swapping of models without service interruption
- Rollback capability in case of issues
- Health checks and validation
- Integration with prediction API
- Deployment logging and monitoring

Author: Data Science Team
Date: August 12, 2025
"""

import os
import shutil
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ModelDeploymentManager:
    """Automated model deployment with hot-swapping and rollback"""
    
    def __init__(self):
        """Initialize model deployment manager"""
        logger.info("🚀 AUTOMATED MODEL DEPLOYMENT")
        logger.info("=" * 32)
        
        # Directories
        self.models_dir = "data_repositories/models"
        self.deployment_dir = os.path.join("data_repositories", "deployment")
        self.staging_dir = os.path.join(self.deployment_dir, "staging")
        self.production_dir = os.path.join(self.deployment_dir, "production")
        self.rollback_dir = os.path.join(self.deployment_dir, "rollback")
        self.logs_dir = os.path.join(self.deployment_dir, "logs")
        
        # Create directories
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.production_dir, exist_ok=True)
        os.makedirs(self.rollback_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Deployment configuration
        self.config = {
            "api_endpoint": "http://localhost:8000",  # FastAPI endpoint
            "health_check_timeout": 30,
            "validation_samples": 10,
            "rollback_timeout": 60,
            "deployment_strategy": "blue_green",  # blue_green or rolling
            "max_retry_attempts": 3
        }
        
        # Deployment results
        self.deployment_results = {
            "timestamp": datetime.now().isoformat(),
            "deployment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "validation_status": None,
            "staging_status": None,
            "production_status": None,
            "health_checks": {},
            "rollback_available": False
        }

    def check_validation_status(self) -> bool:
        """Check if model passed validation"""
        logger.info("\n✅ Checking Validation Status")
        logger.info("-" * 27)
        
        try:
            # Check validation status file
            validation_file = "validation_status.txt"
            
            if not os.path.exists(validation_file):
                logger.error("❌ Validation status file not found")
                return False
            
            with open(validation_file, 'r') as f:
                status = f.read().strip()
            
            validation_passed = status.lower() == "passed"
            
            if validation_passed:
                logger.info("✅ Model validation passed - proceeding with deployment")
            else:
                logger.error("❌ Model validation failed - deployment blocked")
            
            self.deployment_results["validation_status"] = status
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"❌ Error checking validation status: {str(e)}")
            return False

    def prepare_staging_deployment(self) -> bool:
        """Prepare model for staging deployment"""
        logger.info("\n🎬 Preparing Staging Deployment")
        logger.info("-" * 31)
        
        try:
            # Load champion model and metadata
            champion_file = os.path.join(self.models_dir, "trained", "champion_model.pkl")
            metadata_file = os.path.join(self.models_dir, "metadata", "champion_metadata.json")
            scaler_file = os.path.join("data_repositories", "training", "datasets", "feature_scaler.pkl")
            feature_info_file = os.path.join("data_repositories", "training", "datasets", "feature_info.json")
            
            # Check required files
            required_files = {
                "champion_model.pkl": champion_file,
                "champion_metadata.json": metadata_file,
                "feature_scaler.pkl": scaler_file,
                "feature_info.json": feature_info_file
            }
            
            missing_files = []
            for file_name, file_path in required_files.items():
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
                    logger.error(f"❌ Missing: {file_name}")
                else:
                    logger.info(f"✅ Found: {file_name}")
            
            if missing_files:
                logger.error(f"❌ Missing required files: {missing_files}")
                return False
            
            # Copy files to staging
            deployment_id = self.deployment_results["deployment_id"]
            staging_model_dir = os.path.join(self.staging_dir, deployment_id)
            os.makedirs(staging_model_dir, exist_ok=True)
            
            for file_name, file_path in required_files.items():
                staging_path = os.path.join(staging_model_dir, file_name)
                shutil.copy2(file_path, staging_path)
                logger.info(f"📄 Copied {file_name} to staging")
            
            # Create deployment manifest
            manifest = {
                "deployment_id": deployment_id,
                "timestamp": datetime.now().isoformat(),
                "model_files": list(required_files.keys()),
                "deployment_strategy": self.config["deployment_strategy"],
                "status": "staged"
            }
            
            manifest_file = os.path.join(staging_model_dir, "deployment_manifest.json")
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=4)
            
            logger.info(f"✅ Staging deployment prepared: {staging_model_dir}")
            
            self.deployment_results["staging_status"] = "prepared"
            self.deployment_results["staging_path"] = staging_model_dir
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error preparing staging deployment: {str(e)}")
            return False

    def validate_staging_model(self) -> bool:
        """Validate staged model with test predictions"""
        logger.info("\n🧪 Validating Staging Model")
        logger.info("-" * 25)
        
        try:
            staging_path = self.deployment_results["staging_path"]
            
            # Load staged model and components
            model = joblib.load(os.path.join(staging_path, "champion_model.pkl"))
            scaler = joblib.load(os.path.join(staging_path, "feature_scaler.pkl"))
            
            with open(os.path.join(staging_path, "feature_info.json"), 'r') as f:
                feature_info = json.load(f)
            
            with open(os.path.join(staging_path, "champion_metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"✅ Loaded staged model: {metadata.get('model_type', 'Unknown')}")
            
            # Generate test predictions
            logger.info("🔮 Testing model predictions...")
            
            # Create sample test data (using feature info)
            feature_columns = feature_info["feature_columns"]
            n_samples = self.config["validation_samples"]
            
            # Generate realistic test features (simplified)
            np.random.seed(42)
            test_data = np.random.randn(n_samples, len(feature_columns))
            test_df = pd.DataFrame(test_data, columns=feature_columns)
            
            # Scale features
            test_scaled = scaler.transform(test_df)
            
            # Generate predictions
            predictions = model.predict(test_scaled)
            
            # Validate predictions
            valid_predictions = True
            issues = []
            
            # Check for valid prediction range
            if np.any(np.isnan(predictions)):
                valid_predictions = False
                issues.append("NaN predictions detected")
            
            if np.any(np.isinf(predictions)):
                valid_predictions = False
                issues.append("Infinite predictions detected")
            
            # Check AQI range
            if np.any(predictions < 0) or np.any(predictions > 500):
                valid_predictions = False
                issues.append("Predictions outside valid AQI range (0-500)")
            
            if valid_predictions:
                logger.info(f"✅ Model validation passed:")
                logger.info(f"   Test samples: {n_samples}")
                logger.info(f"   Prediction range: {predictions.min():.1f} to {predictions.max():.1f}")
                logger.info(f"   Mean prediction: {predictions.mean():.1f}")
            else:
                logger.error("❌ Model validation failed:")
                for issue in issues:
                    logger.error(f"   • {issue}")
            
            self.deployment_results["staging_status"] = "validated" if valid_predictions else "validation_failed"
            
            return valid_predictions
            
        except Exception as e:
            logger.error(f"❌ Error validating staging model: {str(e)}")
            self.deployment_results["staging_status"] = "validation_error"
            return False

    def backup_current_production(self) -> bool:
        """Backup current production model for rollback"""
        logger.info("\n💾 Backing Up Current Production Model")
        logger.info("-" * 37)
        
        try:
            # Check if production model exists
            production_model_file = os.path.join(self.production_dir, "champion_model.pkl")
            
            if not os.path.exists(production_model_file):
                logger.info("ℹ️ No existing production model - first deployment")
                return True
            
            # Create rollback backup
            rollback_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            rollback_path = os.path.join(self.rollback_dir, rollback_id)
            os.makedirs(rollback_path, exist_ok=True)
            
            # Copy all production files
            production_files = [
                "champion_model.pkl",
                "champion_metadata.json", 
                "feature_scaler.pkl",
                "feature_info.json",
                "deployment_manifest.json"
            ]
            
            backed_up_files = []
            for file_name in production_files:
                prod_file = os.path.join(self.production_dir, file_name)
                if os.path.exists(prod_file):
                    rollback_file = os.path.join(rollback_path, file_name)
                    shutil.copy2(prod_file, rollback_file)
                    backed_up_files.append(file_name)
                    logger.info(f"📄 Backed up: {file_name}")
            
            # Create rollback manifest
            rollback_manifest = {
                "rollback_id": rollback_id,
                "backup_timestamp": datetime.now().isoformat(),
                "backed_up_files": backed_up_files,
                "original_deployment": self.deployment_results["deployment_id"]
            }
            
            rollback_manifest_file = os.path.join(rollback_path, "rollback_manifest.json")
            with open(rollback_manifest_file, 'w') as f:
                json.dump(rollback_manifest, f, indent=4)
            
            logger.info(f"✅ Production model backed up: {rollback_path}")
            
            self.deployment_results["rollback_available"] = True
            self.deployment_results["rollback_path"] = rollback_path
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error backing up production model: {str(e)}")
            return False

    def deploy_to_production(self) -> bool:
        """Deploy staged model to production"""
        logger.info("\n🚀 Deploying to Production")
        logger.info("-" * 25)
        
        try:
            staging_path = self.deployment_results["staging_path"]
            
            # Copy staged files to production
            staging_files = [
                "champion_model.pkl",
                "champion_metadata.json",
                "feature_scaler.pkl", 
                "feature_info.json",
                "deployment_manifest.json"
            ]
            
            deployed_files = []
            for file_name in staging_files:
                staging_file = os.path.join(staging_path, file_name)
                production_file = os.path.join(self.production_dir, file_name)
                
                if os.path.exists(staging_file):
                    shutil.copy2(staging_file, production_file)
                    deployed_files.append(file_name)
                    logger.info(f"📄 Deployed: {file_name}")
            
            # Update deployment manifest
            manifest_file = os.path.join(self.production_dir, "deployment_manifest.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                manifest.update({
                    "production_deployment_time": datetime.now().isoformat(),
                    "status": "deployed",
                    "deployed_files": deployed_files
                })
                
                with open(manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=4)
            
            logger.info(f"✅ Model deployed to production")
            logger.info(f"   Deployed files: {len(deployed_files)}")
            
            self.deployment_results["production_status"] = "deployed"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deploying to production: {str(e)}")
            self.deployment_results["production_status"] = "deployment_failed"
            return False

    def perform_health_checks(self) -> bool:
        """Perform health checks on deployed model"""
        logger.info("\n🏥 Performing Health Checks")
        logger.info("-" * 26)
        
        health_results = {
            "api_health": False,
            "model_predictions": False,
            "response_time": None,
            "prediction_samples": {}
        }
        
        try:
            # Check API health endpoint
            logger.info("🔍 Checking API health...")
            
            api_url = self.config["api_endpoint"]
            timeout = self.config["health_check_timeout"]
            
            try:
                start_time = time.time()
                health_response = requests.get(f"{api_url}/health", timeout=timeout)
                response_time = (time.time() - start_time) * 1000  # milliseconds
                
                if health_response.status_code == 200:
                    health_results["api_health"] = True
                    health_results["response_time"] = response_time
                    logger.info(f"✅ API health check passed ({response_time:.1f}ms)")
                else:
                    logger.error(f"❌ API health check failed: {health_response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ API health check failed: {str(e)}")
            
            # Test model predictions through API
            logger.info("🔮 Testing model predictions...")
            
            try:
                # Sample prediction request
                test_location = {
                    "latitude": 34.0151,
                    "longitude": 71.5249,
                    "city": "Peshawar",
                    "country": "Pakistan"
                }
                
                prediction_payload = {
                    "location": test_location,
                    "include_confidence": True
                }
                
                pred_start_time = time.time()
                prediction_response = requests.post(
                    f"{api_url}/predict/current",
                    json=prediction_payload,
                    timeout=timeout
                )
                pred_response_time = (time.time() - pred_start_time) * 1000
                
                if prediction_response.status_code == 200:
                    pred_data = prediction_response.json()
                    
                    # Validate prediction response
                    if 'prediction' in pred_data and 'aqi' in pred_data['prediction']:
                        aqi_value = pred_data['prediction']['aqi']
                        
                        if 0 <= aqi_value <= 500:  # Valid AQI range
                            health_results["model_predictions"] = True
                            health_results["prediction_samples"]["current_aqi"] = aqi_value
                            health_results["prediction_samples"]["response_time"] = pred_response_time
                            logger.info(f"✅ Model prediction test passed (AQI: {aqi_value:.1f})")
                        else:
                            logger.error(f"❌ Invalid AQI prediction: {aqi_value}")
                    else:
                        logger.error("❌ Invalid prediction response format")
                else:
                    logger.error(f"❌ Prediction test failed: {prediction_response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Model prediction test failed: {str(e)}")
            
            # Overall health assessment
            overall_health = health_results["api_health"] and health_results["model_predictions"]
            
            logger.info(f"\n🏥 Health Check Summary:")
            logger.info(f"   API Health: {'✅' if health_results['api_health'] else '❌'}")
            logger.info(f"   Model Predictions: {'✅' if health_results['model_predictions'] else '❌'}")
            logger.info(f"   Overall Status: {'✅ HEALTHY' if overall_health else '❌ UNHEALTHY'}")
            
            self.deployment_results["health_checks"] = health_results
            
            return overall_health
            
        except Exception as e:
            logger.error(f"❌ Error in health checks: {str(e)}")
            return False

    def save_deployment_log(self, deployment_success: bool) -> bool:
        """Save deployment log and results"""
        logger.info("\n📝 Saving Deployment Log")
        logger.info("-" * 23)
        
        try:
            # Update final deployment results
            self.deployment_results.update({
                "deployment_success": deployment_success,
                "completion_timestamp": datetime.now().isoformat()
            })
            
            # Save detailed deployment log
            deployment_id = self.deployment_results["deployment_id"]
            log_file = os.path.join(self.logs_dir, f"deployment_log_{deployment_id}.json")
            
            with open(log_file, 'w') as f:
                json.dump(self.deployment_results, f, indent=4)
            
            # Save latest deployment status
            latest_log_file = os.path.join(self.logs_dir, "latest_deployment.json")
            with open(latest_log_file, 'w') as f:
                json.dump(self.deployment_results, f, indent=4)
            
            # Save deployment status for CICD
            status_file = "deployment_status.txt"
            with open(status_file, 'w') as f:
                f.write("success" if deployment_success else "failed")
            
            logger.info(f"✅ Deployment log saved:")
            logger.info(f"   Detailed: {log_file}")
            logger.info(f"   Latest: {latest_log_file}")
            logger.info(f"   Status: {status_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving deployment log: {str(e)}")
            return False

    def run_model_deployment(self) -> bool:
        """Run complete model deployment process"""
        logger.info("\n🚀 STARTING MODEL DEPLOYMENT")
        logger.info("=" * 35)
        
        try:
            # Step 1: Check validation status
            if not self.check_validation_status():
                return False
            
            # Step 2: Prepare staging deployment
            if not self.prepare_staging_deployment():
                return False
            
            # Step 3: Validate staging model
            if not self.validate_staging_model():
                return False
            
            # Step 4: Backup current production
            if not self.backup_current_production():
                return False
            
            # Step 5: Deploy to production
            if not self.deploy_to_production():
                return False
            
            # Step 6: Perform health checks
            health_passed = self.perform_health_checks()
            
            # Step 7: Save deployment log
            save_success = self.save_deployment_log(health_passed)
            
            if health_passed:
                logger.info("\n🎉 MODEL DEPLOYMENT COMPLETED SUCCESSFULLY!")
                logger.info("✅ New model is live in production")
                logger.info("🏥 Health checks passed")
            else:
                logger.warning("\n⚠️ Model deployed but health checks failed")
                logger.warning("🔄 Consider rollback if issues persist")
            
            return health_passed
            
        except Exception as e:
            logger.error(f"\n❌ Model deployment failed: {str(e)}")
            self.save_deployment_log(False)
            return False

def main():
    """Main function for model deployment"""
    deployer = ModelDeploymentManager()
    deployment_success = deployer.run_model_deployment()
    
    if deployment_success:
        print("\n🎯 MODEL DEPLOYMENT SUCCESS!")
        print("🚀 New model deployed and operational")
    else:
        print("\n❌ Model deployment failed")
        print("📋 Check deployment logs and system health")

if __name__ == "__main__":
    main()
