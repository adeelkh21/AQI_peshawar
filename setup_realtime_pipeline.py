"""
Real-time Pipeline Setup Script
===============================

This script sets up the complete real-time pipeline infrastructure, creates
necessary directories, validates configurations, and prepares the system
for continuous operation.

Author: Data Science Team
Date: August 12, 2025
"""

import os
import json
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimePipelineSetup:
    """Setup real-time pipeline infrastructure"""
    
    def __init__(self):
        """Initialize pipeline setup"""
        logger.info("🏗️ REAL-TIME PIPELINE SETUP")
        logger.info("=" * 35)
        
        self.setup_results = {
            "timestamp": datetime.now().isoformat(),
            "setup_version": "1.0",
            "components": {}
        }

    def create_directory_structure(self) -> bool:
        """Create required directory structure"""
        logger.info("\n📁 Creating Directory Structure")
        logger.info("-" * 32)
        
        try:
            directories = [
                # Core data repositories
                "data_repositories/hourly_data/raw",
                "data_repositories/hourly_data/processed", 
                "data_repositories/hourly_data/metadata",
                "data_repositories/merged_data/raw",
                "data_repositories/merged_data/processed",
                "data_repositories/merged_data/metadata",
                "data_repositories/features/engineered",
                "data_repositories/features/metadata",
                
                # Hopsworks integration
                "data_repositories/hopsworks/updates",
                "data_repositories/hopsworks/logs",
                "data_repositories/hopsworks/backups",
                
                # Pipeline monitoring
                "data_repositories/pipeline_reports",
                "data_repositories/pipeline_tests",
                "data_repositories/quality_reports",
                
                # Model artifacts (for future phases)
                "data_repositories/models/trained",
                "data_repositories/models/metadata",
                "data_repositories/models/performance"
            ]
            
            created_dirs = []
            for directory in directories:
                try:
                    os.makedirs(directory, exist_ok=True)
                    created_dirs.append(directory)
                    logger.info(f"✅ Created: {directory}")
                except Exception as e:
                    logger.error(f"❌ Failed to create {directory}: {str(e)}")
                    return False
            
            self.setup_results["components"]["directory_structure"] = {
                "status": "success",
                "directories_created": len(created_dirs),
                "total_directories": len(directories)
            }
            
            logger.info(f"✅ Directory structure created: {len(created_dirs)} directories")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating directory structure: {str(e)}")
            self.setup_results["components"]["directory_structure"] = {"status": "error", "error": str(e)}
            return False

    def create_configuration_files(self) -> bool:
        """Create configuration files for the pipeline"""
        logger.info("\n⚙️ Creating Configuration Files")
        logger.info("-" * 33)
        
        try:
            # Pipeline configuration
            pipeline_config = {
                "pipeline_version": "1.0",
                "created": datetime.now().isoformat(),
                "components": {
                    "data_collection": {
                        "frequency": "hourly",
                        "timeout_minutes": 10,
                        "retry_attempts": 3
                    },
                    "feature_engineering": {
                        "target_features": 215,
                        "lag_hours": [1, 3, 6, 12, 24, 48, 72],
                        "rolling_windows": [3, 6, 12, 24],
                        "validation_enabled": True
                    },
                    "hopsworks_integration": {
                        "update_frequency": "hourly",
                        "feature_categories": [
                            "pollution", "weather", "temporal", 
                            "lag_features", "rolling_stats", "advanced"
                        ],
                        "online_enabled": False
                    }
                },
                "monitoring": {
                    "quality_checks": True,
                    "performance_tracking": True,
                    "alert_thresholds": {
                        "data_freshness_hours": 2,
                        "missing_data_percent": 10,
                        "feature_count_deviation": 5
                    }
                }
            }
            
            # Save pipeline configuration
            config_file = "data_repositories/pipeline_config.json"
            with open(config_file, 'w') as f:
                json.dump(pipeline_config, f, indent=4)
            logger.info(f"✅ Pipeline config created: {config_file}")
            
            # Feature engineering configuration
            feature_config = {
                "feature_engineering_version": "2.0",
                "base_features": [
                    "timestamp", "pm2_5", "pm10", "no2", "o3", "aqi_numeric",
                    "temperature", "relative_humidity", "wind_speed", "pressure"
                ],
                "temporal_features": [
                    "hour", "day_of_week", "month", "is_weekend",
                    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
                    "is_morning_rush", "is_evening_rush", "is_night"
                ],
                "feature_categories": {
                    "pollution": {"priority": "high", "update_frequency": "hourly"},
                    "weather": {"priority": "high", "update_frequency": "hourly"},
                    "temporal": {"priority": "medium", "update_frequency": "hourly"},
                    "lag_features": {"priority": "high", "update_frequency": "hourly"},
                    "rolling_stats": {"priority": "medium", "update_frequency": "hourly"},
                    "advanced": {"priority": "low", "update_frequency": "hourly"}
                },
                "validation_rules": {
                    "max_missing_percent": 10,
                    "max_infinite_values": 0,
                    "expected_feature_count": 215,
                    "required_date_range_hours": 168
                }
            }
            
            # Save feature configuration
            feature_config_file = "data_repositories/features/feature_config.json"
            with open(feature_config_file, 'w') as f:
                json.dump(feature_config, f, indent=4)
            logger.info(f"✅ Feature config created: {feature_config_file}")
            
            self.setup_results["components"]["configuration_files"] = {
                "status": "success",
                "files_created": ["pipeline_config.json", "feature_config.json"]
            }
            
            logger.info("✅ Configuration files created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating configuration files: {str(e)}")
            self.setup_results["components"]["configuration_files"] = {"status": "error", "error": str(e)}
            return False

    def create_monitoring_templates(self) -> bool:
        """Create monitoring and logging templates"""
        logger.info("\n📊 Creating Monitoring Templates")
        logger.info("-" * 34)
        
        try:
            # Pipeline status template
            status_template = {
                "pipeline_id": "realtime_aqi_pipeline",
                "last_run": None,
                "status": "initialized",
                "components": {
                    "data_collection": {"status": "ready", "last_success": None},
                    "feature_engineering": {"status": "ready", "last_success": None},
                    "hopsworks_integration": {"status": "ready", "last_success": None}
                },
                "metrics": {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "average_runtime_minutes": 0
                },
                "alerts": []
            }
            
            # Save status template
            status_file = "data_repositories/pipeline_reports/pipeline_status.json"
            with open(status_file, 'w') as f:
                json.dump(status_template, f, indent=4)
            logger.info(f"✅ Status template created: {status_file}")
            
            # Quality monitoring template
            quality_template = {
                "monitoring_version": "1.0",
                "last_check": None,
                "data_quality": {
                    "freshness_status": "unknown",
                    "completeness_status": "unknown", 
                    "accuracy_status": "unknown"
                },
                "feature_quality": {
                    "feature_count": 0,
                    "missing_features": [],
                    "invalid_features": []
                },
                "thresholds": {
                    "max_data_age_hours": 2,
                    "min_completeness_percent": 90,
                    "max_missing_features": 5
                },
                "history": []
            }
            
            # Save quality template
            quality_file = "data_repositories/quality_reports/quality_monitoring.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_template, f, indent=4)
            logger.info(f"✅ Quality template created: {quality_file}")
            
            self.setup_results["components"]["monitoring_templates"] = {
                "status": "success",
                "templates_created": ["pipeline_status.json", "quality_monitoring.json"]
            }
            
            logger.info("✅ Monitoring templates created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating monitoring templates: {str(e)}")
            self.setup_results["components"]["monitoring_templates"] = {"status": "error", "error": str(e)}
            return False

    def validate_dependencies(self) -> bool:
        """Validate required dependencies and configurations"""
        logger.info("\n🔍 Validating Dependencies")
        logger.info("-" * 26)
        
        try:
            validation_results = {
                "python_packages": {},
                "data_files": {},
                "environment_variables": {},
                "issues": []
            }
            
            # Check Python packages
            required_packages = [
                "pandas", "numpy", "requests", "meteostat", 
                "scikit-learn", "xgboost", "lightgbm", "hopsworks"
            ]
            
            for package in required_packages:
                try:
                    __import__(package)
                    validation_results["python_packages"][package] = "available"
                    logger.info(f"✅ Package available: {package}")
                except ImportError:
                    validation_results["python_packages"][package] = "missing"
                    validation_results["issues"].append(f"Missing package: {package}")
                    logger.warning(f"⚠️ Package missing: {package}")
            
            # Check critical data files
            required_data_files = [
                "data_repositories/merged_data/processed/merged_data.csv",
                "data_repositories/features/final_features.csv"
            ]
            
            for file_path in required_data_files:
                if os.path.exists(file_path):
                    validation_results["data_files"][file_path] = "exists"
                    logger.info(f"✅ Data file exists: {os.path.basename(file_path)}")
                else:
                    validation_results["data_files"][file_path] = "missing"
                    validation_results["issues"].append(f"Missing data file: {file_path}")
                    logger.warning(f"⚠️ Data file missing: {file_path}")
            
            # Check environment variables
            env_vars = ["OPENWEATHER_API_KEY", "HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT"]
            
            for var in env_vars:
                if os.getenv(var):
                    validation_results["environment_variables"][var] = "set"
                    logger.info(f"✅ Environment variable set: {var}")
                else:
                    validation_results["environment_variables"][var] = "missing"
                    validation_results["issues"].append(f"Missing environment variable: {var}")
                    logger.warning(f"⚠️ Environment variable missing: {var}")
            
            # Save validation results
            validation_file = "data_repositories/pipeline_validation.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=4)
            
            # Overall validation status
            if len(validation_results["issues"]) == 0:
                logger.info("✅ All dependencies validated successfully")
                self.setup_results["components"]["dependency_validation"] = {
                    "status": "passed",
                    "issues_count": 0
                }
                return True
            else:
                logger.warning(f"⚠️ Dependency validation completed with {len(validation_results['issues'])} issues")
                self.setup_results["components"]["dependency_validation"] = {
                    "status": "warnings",
                    "issues_count": len(validation_results["issues"]),
                    "issues": validation_results["issues"]
                }
                return True  # Don't fail setup for warnings
                
        except Exception as e:
            logger.error(f"❌ Error validating dependencies: {str(e)}")
            self.setup_results["components"]["dependency_validation"] = {"status": "error", "error": str(e)}
            return False

    def create_documentation(self) -> bool:
        """Create setup documentation"""
        logger.info("\n📖 Creating Documentation")
        logger.info("-" * 25)
        
        try:
            readme_content = """# Real-time AQI Pipeline Setup

## Overview
This directory contains the real-time AQI prediction pipeline that continuously:
1. Collects hourly weather and pollution data
2. Engineers features using the validated 215-feature pipeline
3. Updates Hopsworks feature store
4. Prepares data for model retraining every 3 hours

## Directory Structure
```
data_repositories/
├── hourly_data/           # Hourly data collection
├── merged_data/           # Merged weather + pollution data
├── features/              # Engineered features
├── hopsworks/            # Hopsworks integration logs
├── pipeline_reports/     # Pipeline execution reports
├── quality_reports/      # Data quality monitoring
└── models/              # Model artifacts (Phase 2)
```

## Pipeline Components

### 1. Data Collection (`phase1_data_collection.py`)
- Runs every hour via GitHub Actions
- Collects weather data from Meteostat
- Collects pollution data from OpenWeatherMap
- Validates and stores raw data

### 2. Feature Engineering (`realtime_feature_engineering.py`)
- Processes latest merged data
- Creates 215 engineered features
- Maintains consistency with training pipeline
- Validates feature quality

### 3. Hopsworks Integration (`realtime_hopsworks_integration.py`)
- Updates feature store with latest features
- Organizes features by category
- Maintains feature versioning
- Provides update monitoring

### 4. Pipeline Testing (`test_realtime_pipeline.py`)
- Validates end-to-end pipeline
- Tests all components
- Provides quality assurance

## Configuration Files

### `pipeline_config.json`
Main pipeline configuration including component settings, monitoring thresholds, and operational parameters.

### `features/feature_config.json`  
Feature engineering configuration with feature definitions, validation rules, and quality thresholds.

## Monitoring

### Pipeline Status
Check `pipeline_reports/pipeline_status.json` for current pipeline health and execution metrics.

### Data Quality
Monitor `quality_reports/quality_monitoring.json` for data freshness, completeness, and accuracy metrics.

## Environment Variables Required
- `OPENWEATHER_API_KEY`: For pollution data collection
- `HOPSWORKS_API_KEY`: For feature store integration  
- `HOPSWORKS_PROJECT`: Hopsworks project name

## Running the Pipeline

### Manual Testing
```bash
python test_realtime_pipeline.py
```

### Individual Components
```bash
python realtime_feature_engineering.py
python realtime_hopsworks_integration.py
```

### GitHub Actions
The pipeline runs automatically every hour via `.github/workflows/realtime_data_pipeline.yml`

## Next Steps (Phase 2)
1. Model retraining every 3 hours
2. Automated model deployment
3. Real-time prediction updates
4. Performance monitoring

---
Generated: {}
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Save README
            readme_file = "data_repositories/README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            logger.info(f"✅ Documentation created: {readme_file}")
            
            self.setup_results["components"]["documentation"] = {
                "status": "success",
                "files_created": ["README.md"]
            }
            
            logger.info("✅ Documentation created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating documentation: {str(e)}")
            self.setup_results["components"]["documentation"] = {"status": "error", "error": str(e)}
            return False

    def run_complete_setup(self) -> bool:
        """Run complete pipeline setup"""
        logger.info("\n🚀 STARTING COMPLETE PIPELINE SETUP")
        logger.info("=" * 45)
        
        setup_steps = [
            ("Directory Structure", self.create_directory_structure),
            ("Configuration Files", self.create_configuration_files),
            ("Monitoring Templates", self.create_monitoring_templates),
            ("Dependency Validation", self.validate_dependencies),
            ("Documentation", self.create_documentation)
        ]
        
        successful_steps = 0
        total_steps = len(setup_steps)
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n{'='*15} {step_name.upper()} {'='*15}")
            
            try:
                success = step_func()
                if success:
                    successful_steps += 1
                    logger.info(f"✅ {step_name}: COMPLETED")
                else:
                    logger.error(f"❌ {step_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {step_name}: ERROR - {str(e)}")
        
        # Save setup results
        self.save_setup_results()
        
        # Calculate success rate
        success_rate = successful_steps / total_steps
        
        if success_rate == 1.0:
            logger.info(f"\n🎉 SETUP COMPLETED SUCCESSFULLY ({successful_steps}/{total_steps})")
            logger.info("✅ Real-time pipeline is ready for deployment!")
            return True
        elif success_rate >= 0.8:
            logger.warning(f"\n⚠️ SETUP MOSTLY COMPLETED ({successful_steps}/{total_steps})")
            logger.warning("🔧 Review any warnings and proceed carefully")
            return True
        else:
            logger.error(f"\n❌ SETUP FAILED ({successful_steps}/{total_steps})")
            logger.error("🛑 Fix critical issues before proceeding")
            return False

    def save_setup_results(self) -> None:
        """Save setup results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"data_repositories/pipeline_setup_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.setup_results, f, indent=4)
            
            logger.info(f"📁 Setup results saved: {results_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save setup results: {str(e)}")

def main():
    """Main function for pipeline setup"""
    setup = RealTimePipelineSetup()
    
    try:
        success = setup.run_complete_setup()
        
        if success:
            print("\n🎯 PIPELINE SETUP SUCCESS!")
            print("🚀 Real-time pipeline is ready")
            print("📋 Check data_repositories/README.md for documentation")
        else:
            print("\n❌ Pipeline setup failed")
            print("📋 Check logs and fix issues")
            
    except Exception as e:
        logger.error(f"❌ Setup framework error: {str(e)}")
        print("\n💥 Setup framework error")

if __name__ == "__main__":
    main()
