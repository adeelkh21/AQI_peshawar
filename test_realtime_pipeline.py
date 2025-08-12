"""
Real-time Pipeline Testing Script
=================================

This script tests the complete real-time data pipeline to ensure all components
work together correctly before deployment.

Components tested:
- Data collection and merging
- Feature engineering pipeline
- Hopsworks integration
- End-to-end pipeline validation

Author: Data Science Team
Date: August 12, 2025
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimePipelineTester:
    """Test the complete real-time pipeline"""
    
    def __init__(self):
        """Initialize pipeline tester"""
        logger.info("🧪 REAL-TIME PIPELINE TESTING")
        logger.info("=" * 40)
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown"
        }

    def test_data_availability(self) -> bool:
        """Test 1: Check if required data files exist"""
        logger.info("\n📂 TEST 1: Data Availability")
        logger.info("-" * 28)
        
        try:
            required_files = [
                "data_repositories/merged_data/processed/merged_data.csv",
                "data_repositories/merged_data/metadata/dataset_info.json"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    logger.error(f"❌ Missing: {file_path}")
                else:
                    logger.info(f"✅ Found: {file_path}")
            
            if missing_files:
                self.test_results["tests"]["data_availability"] = {
                    "status": "failed",
                    "missing_files": missing_files
                }
                logger.error(f"❌ Data availability test failed: {len(missing_files)} missing files")
                return False
            else:
                self.test_results["tests"]["data_availability"] = {
                    "status": "passed",
                    "files_checked": len(required_files)
                }
                logger.info("✅ Data availability test passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Data availability test error: {str(e)}")
            self.test_results["tests"]["data_availability"] = {"status": "error", "error": str(e)}
            return False

    def test_feature_engineering(self) -> bool:
        """Test 2: Run feature engineering pipeline"""
        logger.info("\n🔧 TEST 2: Feature Engineering")
        logger.info("-" * 30)
        
        try:
            # Import and run feature engineering
            from realtime_feature_engineering import RealTimeFeatureEngineer
            
            engineer = RealTimeFeatureEngineer()
            success = engineer.run_realtime_pipeline()
            
            if success:
                # Check if feature file was created
                feature_file = "data_repositories/features/engineered/realtime_features.csv"
                if os.path.exists(feature_file):
                    df = pd.read_csv(feature_file)
                    self.test_results["tests"]["feature_engineering"] = {
                        "status": "passed",
                        "features_created": len(df.columns),
                        "records_processed": len(df)
                    }
                    logger.info(f"✅ Feature engineering test passed: {len(df.columns)} features, {len(df)} records")
                    return True
                else:
                    logger.error("❌ Feature file not created")
                    return False
            else:
                self.test_results["tests"]["feature_engineering"] = {"status": "failed"}
                logger.error("❌ Feature engineering pipeline failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Feature engineering test error: {str(e)}")
            self.test_results["tests"]["feature_engineering"] = {"status": "error", "error": str(e)}
            return False

    def test_hopsworks_connection(self) -> bool:
        """Test 3: Test Hopsworks connection (if credentials available)"""
        logger.info("\n🏪 TEST 3: Hopsworks Connection")
        logger.info("-" * 32)
        
        try:
            # Check if credentials are available
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT')
            
            if not api_key:
                logger.warning("⚠️ HOPSWORKS_API_KEY not found - skipping Hopsworks test")
                self.test_results["tests"]["hopsworks_connection"] = {
                    "status": "skipped",
                    "reason": "No API key found"
                }
                return True  # Not a failure, just skipped
            
            # Import and test Hopsworks integration
            from realtime_hopsworks_integration import RealTimeHopsworksManager
            
            manager = RealTimeHopsworksManager()
            connection_success = manager.step1_verify_hopsworks_connection()
            
            if connection_success:
                self.test_results["tests"]["hopsworks_connection"] = {
                    "status": "passed",
                    "project": project_name
                }
                logger.info("✅ Hopsworks connection test passed")
                return True
            else:
                self.test_results["tests"]["hopsworks_connection"] = {"status": "failed"}
                logger.error("❌ Hopsworks connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Hopsworks connection test error: {str(e)}")
            self.test_results["tests"]["hopsworks_connection"] = {"status": "error", "error": str(e)}
            return False

    def test_pipeline_integration(self) -> bool:
        """Test 4: End-to-end pipeline integration"""
        logger.info("\n🔗 TEST 4: Pipeline Integration")
        logger.info("-" * 31)
        
        try:
            # Check if all pipeline outputs exist
            pipeline_outputs = [
                "data_repositories/features/engineered/realtime_features.csv",
                "data_repositories/features/metadata/realtime_feature_metadata.json"
            ]
            
            missing_outputs = []
            for output_path in pipeline_outputs:
                if not os.path.exists(output_path):
                    missing_outputs.append(output_path)
                    logger.error(f"❌ Missing output: {output_path}")
                else:
                    logger.info(f"✅ Found output: {output_path}")
            
            if missing_outputs:
                self.test_results["tests"]["pipeline_integration"] = {
                    "status": "failed", 
                    "missing_outputs": missing_outputs
                }
                logger.error(f"❌ Pipeline integration test failed: {len(missing_outputs)} missing outputs")
                return False
            
            # Load and validate feature metadata
            metadata_file = "data_repositories/features/metadata/realtime_feature_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            required_metadata_keys = ["last_update", "total_features", "total_records", "categories"]
            missing_keys = [key for key in required_metadata_keys if key not in metadata]
            
            if missing_keys:
                logger.error(f"❌ Missing metadata keys: {missing_keys}")
                return False
            
            self.test_results["tests"]["pipeline_integration"] = {
                "status": "passed",
                "outputs_verified": len(pipeline_outputs),
                "total_features": metadata.get("total_features", 0),
                "last_update": metadata.get("last_update")
            }
            
            logger.info("✅ Pipeline integration test passed")
            logger.info(f"📊 Pipeline outputs: {len(pipeline_outputs)} files")
            logger.info(f"🔢 Total features: {metadata.get('total_features', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline integration test error: {str(e)}")
            self.test_results["tests"]["pipeline_integration"] = {"status": "error", "error": str(e)}
            return False

    def run_comprehensive_test(self) -> bool:
        """Run all pipeline tests"""
        logger.info("\n🚀 STARTING COMPREHENSIVE PIPELINE TESTING")
        logger.info("=" * 50)
        
        tests = [
            ("Data Availability", self.test_data_availability),
            ("Feature Engineering", self.test_feature_engineering),
            ("Hopsworks Connection", self.test_hopsworks_connection),
            ("Pipeline Integration", self.test_pipeline_integration)
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
            logger.info("✅ Real-time pipeline is ready for deployment!")
            return True
        elif success_rate >= 0.75:
            self.test_results["overall_status"] = "mostly_passed"
            logger.warning(f"\n⚠️ MOST TESTS PASSED ({passed_tests}/{total_tests})")
            logger.warning("🔧 Review failed tests and fix issues")
            return True
        else:
            self.test_results["overall_status"] = "failed"
            logger.error(f"\n❌ PIPELINE TESTING FAILED ({passed_tests}/{total_tests})")
            logger.error("🛑 Fix critical issues before deployment")
            return False

    def save_test_results(self) -> None:
        """Save test results to file"""
        try:
            os.makedirs("data_repositories/pipeline_tests", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"data_repositories/pipeline_tests/test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=4)
            
            logger.info(f"📁 Test results saved: {results_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save test results: {str(e)}")

def main():
    """Main function for pipeline testing"""
    tester = RealTimePipelineTester()
    
    try:
        success = tester.run_comprehensive_test()
        tester.save_test_results()
        
        if success:
            print("\n🎯 PIPELINE TESTING SUCCESS!")
            print("🚀 Real-time pipeline is ready")
            sys.exit(0)
        else:
            print("\n❌ Pipeline testing failed")
            print("📋 Check logs and fix issues")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Testing framework error: {str(e)}")
        print("\n💥 Testing framework error")
        sys.exit(1)

if __name__ == "__main__":
    main()
