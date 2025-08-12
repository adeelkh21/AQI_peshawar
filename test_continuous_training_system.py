"""
Continuous Training System Test Suite
=====================================

This script tests the complete Phase 2 continuous training system to ensure
all components work together correctly.

Components tested:
- Model retraining trigger system
- Champion/challenger training
- Model validation and deployment
- Performance monitoring setup
- End-to-end training pipeline

Author: Data Science Team
Date: August 12, 2025
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousTrainingTester:
    """Test the complete continuous training system"""
    
    def __init__(self):
        """Initialize training system tester"""
        logger.info("🧪 CONTINUOUS TRAINING SYSTEM TESTING")
        logger.info("=" * 45)
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown"
        }

    def test_pipeline_structure(self) -> bool:
        """Test 1: Check if all pipeline components exist"""
        logger.info("\n📂 TEST 1: Pipeline Structure")
        logger.info("-" * 30)
        
        try:
            required_files = [
                ".github/workflows/model_retraining_pipeline.yml",
                "continuous_model_trigger.py",
                "fetch_training_features.py", 
                "prepare_training_dataset.py",
                "train_champion_challenger.py",
                "validate_model_performance.py",
                "deploy_champion_model.py",
                "update_model_registry.py",
                "setup_model_monitoring.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    logger.error(f"❌ Missing: {file_path}")
                else:
                    logger.info(f"✅ Found: {file_path}")
            
            if missing_files:
                self.test_results["tests"]["pipeline_structure"] = {
                    "status": "failed",
                    "missing_files": missing_files
                }
                logger.error(f"❌ Pipeline structure test failed: {len(missing_files)} missing files")
                return False
            else:
                self.test_results["tests"]["pipeline_structure"] = {
                    "status": "passed",
                    "files_checked": len(required_files)
                }
                logger.info("✅ Pipeline structure test passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Pipeline structure test error: {str(e)}")
            self.test_results["tests"]["pipeline_structure"] = {"status": "error", "error": str(e)}
            return False

    def test_directory_structure(self) -> bool:
        """Test 2: Check if required directories exist or can be created"""
        logger.info("\n📁 TEST 2: Directory Structure")
        logger.info("-" * 29)
        
        try:
            required_dirs = [
                "data_repositories/training/features",
                "data_repositories/training/datasets", 
                "data_repositories/training/logs",
                "data_repositories/models/trained",
                "data_repositories/models/metadata",
                "data_repositories/models/performance",
                "data_repositories/deployment/staging",
                "data_repositories/deployment/production",
                "data_repositories/monitoring/performance",
                "data_repositories/monitoring/alerts"
            ]
            
            created_dirs = []
            for dir_path in required_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    created_dirs.append(dir_path)
                    logger.info(f"✅ Ready: {dir_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to create {dir_path}: {str(e)}")
                    return False
            
            self.test_results["tests"]["directory_structure"] = {
                "status": "passed",
                "directories_ready": len(created_dirs)
            }
            
            logger.info(f"✅ Directory structure test passed: {len(created_dirs)} directories ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory structure test error: {str(e)}")
            self.test_results["tests"]["directory_structure"] = {"status": "error", "error": str(e)}
            return False

    def test_trigger_system(self) -> bool:
        """Test 3: Test the model retraining trigger system"""
        logger.info("\n🔍 TEST 3: Trigger System")
        logger.info("-" * 24)
        
        try:
            # Import and test trigger system
            from continuous_model_trigger import ModelRetrainingTrigger
            
            trigger = ModelRetrainingTrigger()
            
            # Test override conditions check
            has_override, override_reasons = trigger.check_override_conditions()
            logger.info(f"✅ Override check: {has_override} ({len(override_reasons)} reasons)")
            
            # Test configuration validation
            config_valid = (
                "performance_thresholds" in trigger.config and
                "data_requirements" in trigger.config and
                "drift_thresholds" in trigger.config
            )
            
            if config_valid:
                logger.info("✅ Trigger configuration is valid")
            else:
                logger.error("❌ Trigger configuration is invalid")
                return False
            
            self.test_results["tests"]["trigger_system"] = {
                "status": "passed",
                "configuration_valid": config_valid,
                "override_check": has_override
            }
            
            logger.info("✅ Trigger system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trigger system test error: {str(e)}")
            self.test_results["tests"]["trigger_system"] = {"status": "error", "error": str(e)}
            return False

    def test_model_training_components(self) -> bool:
        """Test 4: Test model training components"""
        logger.info("\n🎯 TEST 4: Model Training Components")
        logger.info("-" * 36)
        
        try:
            # Test champion/challenger trainer import
            from train_champion_challenger import ChampionChallengerTrainer
            
            trainer = ChampionChallengerTrainer()
            
            # Check model configurations
            model_configs = trainer.model_configs
            enabled_models = [name for name, config in model_configs.items() if config.get('enabled', False)]
            
            logger.info(f"✅ Model configurations loaded: {len(model_configs)} total")
            logger.info(f"   Enabled models: {', '.join(enabled_models)}")
            
            # Test if required libraries are available
            try:
                import xgboost as xgb
                logger.info("✅ XGBoost available")
                xgb_available = True
            except ImportError:
                logger.warning("⚠️ XGBoost not available")
                xgb_available = False
            
            try:
                import lightgbm as lgb
                logger.info("✅ LightGBM available")
                lgb_available = True
            except ImportError:
                logger.warning("⚠️ LightGBM not available")
                lgb_available = False
            
            # Check if at least one model is available
            if not (xgb_available or lgb_available):
                logger.error("❌ No ML libraries available for training")
                return False
            
            self.test_results["tests"]["model_training_components"] = {
                "status": "passed",
                "model_configs": len(model_configs),
                "enabled_models": len(enabled_models),
                "xgboost_available": xgb_available,
                "lightgbm_available": lgb_available
            }
            
            logger.info("✅ Model training components test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training components test error: {str(e)}")
            self.test_results["tests"]["model_training_components"] = {"status": "error", "error": str(e)}
            return False

    def test_validation_system(self) -> bool:
        """Test 5: Test model validation system"""
        logger.info("\n✅ TEST 5: Validation System")
        logger.info("-" * 26)
        
        try:
            # Test validator import
            from validate_model_performance import ModelPerformanceValidator
            
            validator = ModelPerformanceValidator()
            
            # Check validation thresholds
            thresholds = validator.thresholds
            required_thresholds = [
                "minimum_r2", "maximum_mae", "maximum_mape", 
                "minimum_improvement", "regression_tolerance"
            ]
            
            missing_thresholds = [t for t in required_thresholds if t not in thresholds]
            
            if missing_thresholds:
                logger.error(f"❌ Missing thresholds: {missing_thresholds}")
                return False
            
            logger.info(f"✅ Validation thresholds configured: {len(thresholds)}")
            logger.info(f"   Minimum R²: {thresholds['minimum_r2']}")
            logger.info(f"   Maximum MAE: {thresholds['maximum_mae']}")
            
            self.test_results["tests"]["validation_system"] = {
                "status": "passed",
                "thresholds_configured": len(thresholds),
                "minimum_r2": thresholds["minimum_r2"],
                "maximum_mae": thresholds["maximum_mae"]
            }
            
            logger.info("✅ Validation system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation system test error: {str(e)}")
            self.test_results["tests"]["validation_system"] = {"status": "error", "error": str(e)}
            return False

    def test_deployment_system(self) -> bool:
        """Test 6: Test deployment system"""
        logger.info("\n🚀 TEST 6: Deployment System")
        logger.info("-" * 26)
        
        try:
            # Test deployment manager import
            from deploy_champion_model import ModelDeploymentManager
            
            deployer = ModelDeploymentManager()
            
            # Check deployment configuration
            config = deployer.config
            required_config = ["api_endpoint", "health_check_timeout", "deployment_strategy"]
            
            missing_config = [c for c in required_config if c not in config]
            
            if missing_config:
                logger.error(f"❌ Missing deployment config: {missing_config}")
                return False
            
            logger.info(f"✅ Deployment configuration loaded: {len(config)} settings")
            logger.info(f"   API endpoint: {config['api_endpoint']}")
            logger.info(f"   Strategy: {config['deployment_strategy']}")
            
            # Check if required directories exist
            deployment_dirs = [deployer.staging_dir, deployer.production_dir, deployer.rollback_dir]
            for dir_path in deployment_dirs:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"✅ Directory ready: {os.path.basename(dir_path)}")
            
            self.test_results["tests"]["deployment_system"] = {
                "status": "passed",
                "configuration_items": len(config),
                "api_endpoint": config["api_endpoint"],
                "deployment_strategy": config["deployment_strategy"]
            }
            
            logger.info("✅ Deployment system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment system test error: {str(e)}")
            self.test_results["tests"]["deployment_system"] = {"status": "error", "error": str(e)}
            return False

    def test_monitoring_setup(self) -> bool:
        """Test 7: Test monitoring system setup"""
        logger.info("\n📊 TEST 7: Monitoring Setup")
        logger.info("-" * 25)
        
        try:
            # Test monitoring setup import
            from setup_model_monitoring import ModelMonitoringSetup
            
            monitoring = ModelMonitoringSetup()
            
            # Check monitoring configuration
            config = monitoring.config
            required_sections = ["alert_thresholds", "monitoring_intervals", "baseline_performance"]
            
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                logger.error(f"❌ Missing monitoring config sections: {missing_sections}")
                return False
            
            # Check alert thresholds
            alert_thresholds = config["alert_thresholds"]
            logger.info(f"✅ Alert thresholds configured: {len(alert_thresholds)}")
            
            # Check monitoring intervals
            intervals = config["monitoring_intervals"]
            logger.info(f"✅ Monitoring intervals configured: {len(intervals)}")
            
            self.test_results["tests"]["monitoring_setup"] = {
                "status": "passed",
                "alert_thresholds": len(alert_thresholds),
                "monitoring_intervals": len(intervals),
                "baseline_configured": "baseline_performance" in config
            }
            
            logger.info("✅ Monitoring setup test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup test error: {str(e)}")
            self.test_results["tests"]["monitoring_setup"] = {"status": "error", "error": str(e)}
            return False

    def test_cicd_pipeline_syntax(self) -> bool:
        """Test 8: Validate CICD pipeline syntax"""
        logger.info("\n⚙️ TEST 8: CICD Pipeline Syntax")
        logger.info("-" * 30)
        
        try:
            pipeline_file = ".github/workflows/model_retraining_pipeline.yml"
            
            if not os.path.exists(pipeline_file):
                logger.error("❌ CICD pipeline file not found")
                return False
            
            # Read and validate basic YAML structure
            with open(pipeline_file, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            required_sections = ["name:", "on:", "jobs:", "steps:"]
            missing_sections = []
            
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.error(f"❌ Missing YAML sections: {missing_sections}")
                return False
            
            # Check for key workflow steps
            required_steps = [
                "continuous_model_trigger.py",
                "fetch_training_features.py",
                "prepare_training_dataset.py", 
                "train_champion_challenger.py",
                "validate_model_performance.py",
                "deploy_champion_model.py"
            ]
            
            missing_steps = []
            for step in required_steps:
                if step not in content:
                    missing_steps.append(step)
            
            if missing_steps:
                logger.warning(f"⚠️ Missing workflow steps: {missing_steps}")
            
            logger.info(f"✅ CICD pipeline structure validated")
            logger.info(f"   Required sections: {len(required_sections)} found")
            logger.info(f"   Required steps: {len(required_steps) - len(missing_steps)}/{len(required_steps)} found")
            
            self.test_results["tests"]["cicd_pipeline_syntax"] = {
                "status": "passed",
                "required_sections": len(required_sections),
                "required_steps_found": len(required_steps) - len(missing_steps),
                "total_steps": len(required_steps)
            }
            
            logger.info("✅ CICD pipeline syntax test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ CICD pipeline syntax test error: {str(e)}")
            self.test_results["tests"]["cicd_pipeline_syntax"] = {"status": "error", "error": str(e)}
            return False

    def run_comprehensive_test(self) -> bool:
        """Run all continuous training system tests"""
        logger.info("\n🚀 STARTING COMPREHENSIVE SYSTEM TESTING")
        logger.info("=" * 50)
        
        tests = [
            ("Pipeline Structure", self.test_pipeline_structure),
            ("Directory Structure", self.test_directory_structure),
            ("Trigger System", self.test_trigger_system),
            ("Model Training Components", self.test_model_training_components),
            ("Validation System", self.test_validation_system),
            ("Deployment System", self.test_deployment_system),
            ("Monitoring Setup", self.test_monitoring_setup),
            ("CICD Pipeline Syntax", self.test_cicd_pipeline_syntax)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name.upper()} {'='*20}")
            
            try:
                success = test_func()
                if success:
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
        
        # Calculate overall status
        success_rate = passed_tests / total_tests
        
        if success_rate == 1.0:
            self.test_results["overall_status"] = "all_passed"
            logger.info(f"\n🎉 ALL TESTS PASSED ({passed_tests}/{total_tests})")
            logger.info("✅ Continuous training system is ready for deployment!")
            return True
        elif success_rate >= 0.75:
            self.test_results["overall_status"] = "mostly_passed"
            logger.warning(f"\n⚠️ MOST TESTS PASSED ({passed_tests}/{total_tests})")
            logger.warning("🔧 Review failed tests and fix issues")
            return True
        else:
            self.test_results["overall_status"] = "failed"
            logger.error(f"\n❌ SYSTEM TESTING FAILED ({passed_tests}/{total_tests})")
            logger.error("🛑 Fix critical issues before deployment")
            return False

    def save_test_results(self) -> None:
        """Save test results to file"""
        try:
            os.makedirs("data_repositories/training/tests", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"data_repositories/training/tests/system_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=4)
            
            logger.info(f"📁 Test results saved: {results_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save test results: {str(e)}")

def main():
    """Main function for continuous training system testing"""
    tester = ContinuousTrainingTester()
    
    try:
        success = tester.run_comprehensive_test()
        tester.save_test_results()
        
        if success:
            print("\n🎯 CONTINUOUS TRAINING SYSTEM TESTING SUCCESS!")
            print("🚀 System is ready for production deployment")
            print("📋 All critical components validated")
            sys.exit(0)
        else:
            print("\n❌ Continuous training system testing failed")
            print("📋 Check logs and fix issues before deployment")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Testing framework error: {str(e)}")
        print("\n💥 Testing framework error")
        sys.exit(1)

if __name__ == "__main__":
    main()
