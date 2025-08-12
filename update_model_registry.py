"""
Model Registry Update System
===========================

This script maintains a comprehensive model registry that tracks:
- Model versions and metadata
- Performance history and comparisons
- Deployment status and timestamps
- Rollback information and lineage
- Model lifecycle management

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Comprehensive model registry management system"""
    
    def __init__(self):
        """Initialize model registry"""
        logger.info("📝 MODEL REGISTRY UPDATE")
        logger.info("=" * 25)
        
        # Directories
        self.models_dir = "data_repositories/models"
        self.registry_dir = os.path.join(self.models_dir, "registry")
        
        # Create registry directory
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Registry files
        self.registry_file = os.path.join(self.registry_dir, "model_registry.json")
        self.history_file = os.path.join(self.registry_dir, "model_history.json")
        self.performance_file = os.path.join(self.registry_dir, "performance_history.json")

    def load_current_registry(self) -> Dict:
        """Load current model registry"""
        logger.info("\n📖 Loading Current Registry")
        logger.info("-" * 26)
        
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                logger.info(f"✅ Registry loaded: {len(registry.get('models', {}))} models")
            else:
                logger.info("ℹ️ No existing registry - creating new")
                registry = {
                    "registry_version": "1.0",
                    "created": datetime.now().isoformat(),
                    "models": {},
                    "current_champion": None,
                    "last_updated": None
                }
            
            return registry
            
        except Exception as e:
            logger.error(f"❌ Error loading registry: {str(e)}")
            return {"models": {}, "current_champion": None}

    def get_model_metadata(self) -> Optional[Dict]:
        """Get latest model metadata"""
        logger.info("\n📊 Gathering Model Metadata")
        logger.info("-" * 28)
        
        try:
            # Load champion metadata
            metadata_file = os.path.join(self.models_dir, "metadata", "champion_metadata.json")
            if not os.path.exists(metadata_file):
                logger.error("❌ Champion metadata not found")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load performance data
            performance_file = os.path.join(self.models_dir, "performance", "latest_performance.json")
            performance_data = {}
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
            
            # Load deployment status
            deployment_file = os.path.join("data_repositories", "deployment", "logs", "latest_deployment.json")
            deployment_data = {}
            if os.path.exists(deployment_file):
                with open(deployment_file, 'r') as f:
                    deployment_data = json.load(f)
            
            logger.info(f"✅ Model metadata gathered:")
            logger.info(f"   Model type: {metadata.get('model_type', 'Unknown')}")
            logger.info(f"   Training date: {metadata.get('training_date', 'Unknown')}")
            logger.info(f"   Performance available: {'Yes' if performance_data else 'No'}")
            logger.info(f"   Deployment status: {'Yes' if deployment_data else 'No'}")
            
            return {
                "metadata": metadata,
                "performance": performance_data,
                "deployment": deployment_data
            }
            
        except Exception as e:
            logger.error(f"❌ Error gathering metadata: {str(e)}")
            return None

    def create_model_entry(self, model_data: Dict) -> Dict:
        """Create new model registry entry"""
        logger.info("\n📝 Creating Model Entry")
        logger.info("-" * 22)
        
        try:
            metadata = model_data["metadata"]
            performance = model_data["performance"]
            deployment = model_data["deployment"]
            
            # Generate model ID
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract key information
            model_entry = {
                "model_id": model_id,
                "model_type": metadata.get("model_type", "Unknown"),
                "model_version": metadata.get("model_version", "1.0"),
                "created_timestamp": datetime.now().isoformat(),
                "training_timestamp": metadata.get("training_date"),
                
                # Performance metrics
                "performance": {
                    "r2_score": None,
                    "mae": None,
                    "rmse": None,
                    "validation_metrics": {}
                },
                
                # Model configuration
                "configuration": {
                    "parameters": metadata.get("parameters", {}),
                    "feature_count": metadata.get("feature_info", {}).get("n_features", 0),
                    "training_samples": metadata.get("training_samples", 0)
                },
                
                # Deployment information
                "deployment": {
                    "status": "unknown",
                    "deployment_timestamp": None,
                    "deployment_id": None,
                    "health_status": "unknown"
                },
                
                # Model lifecycle
                "lifecycle": {
                    "status": "trained",  # trained, validated, deployed, retired
                    "is_champion": False,
                    "previous_champion": None,
                    "retirement_date": None
                },
                
                # Lineage and comparison
                "lineage": {
                    "training_run_id": metadata.get("model_version"),
                    "baseline_model": None,
                    "improvement_metrics": {}
                }
            }
            
            # Extract performance metrics if available
            if performance:
                perf_comparison = performance.get("performance_comparison", {})
                champion_model = performance.get("champion_model", {})
                
                if champion_model and "performance" in champion_model:
                    perf_metrics = champion_model["performance"]
                    model_entry["performance"].update({
                        "r2_score": perf_metrics.get("r2_score"),
                        "mae": perf_metrics.get("mae"),
                        "rmse": perf_metrics.get("rmse"),
                        "validation_metrics": perf_metrics
                    })
                
                # Add improvement metrics if baseline exists
                if perf_comparison:
                    for model_name, metrics in perf_comparison.items():
                        if model_name == "current_champion" and "r2_score" in metrics:
                            # Calculate improvement
                            current_r2 = model_entry["performance"]["r2_score"]
                            baseline_r2 = metrics["r2_score"]
                            
                            if current_r2 and baseline_r2:
                                improvement = ((current_r2 - baseline_r2) / baseline_r2) * 100
                                model_entry["lineage"]["improvement_metrics"] = {
                                    "r2_improvement_percent": improvement,
                                    "baseline_r2": baseline_r2,
                                    "current_r2": current_r2
                                }
            
            # Extract deployment information
            if deployment:
                model_entry["deployment"].update({
                    "status": deployment.get("production_status", "unknown"),
                    "deployment_timestamp": deployment.get("timestamp"),
                    "deployment_id": deployment.get("deployment_id"),
                    "health_status": "healthy" if deployment.get("health_checks", {}).get("api_health") else "unknown"
                })
                
                # Update lifecycle based on deployment
                if deployment.get("deployment_success"):
                    model_entry["lifecycle"]["status"] = "deployed"
                    model_entry["lifecycle"]["is_champion"] = True
            
            logger.info(f"✅ Model entry created:")
            logger.info(f"   Model ID: {model_id}")
            logger.info(f"   Type: {model_entry['model_type']}")
            logger.info(f"   R² Score: {model_entry['performance']['r2_score']}")
            logger.info(f"   Status: {model_entry['lifecycle']['status']}")
            
            return model_entry
            
        except Exception as e:
            logger.error(f"❌ Error creating model entry: {str(e)}")
            return {}

    def update_registry(self, registry: Dict, new_model: Dict) -> Dict:
        """Update registry with new model"""
        logger.info("\n🔄 Updating Registry")
        logger.info("-" * 18)
        
        try:
            model_id = new_model["model_id"]
            
            # Update previous champion status
            if registry.get("current_champion"):
                prev_champion_id = registry["current_champion"]
                if prev_champion_id in registry["models"]:
                    registry["models"][prev_champion_id]["lifecycle"]["is_champion"] = False
                    new_model["lineage"]["previous_champion"] = prev_champion_id
                    logger.info(f"📈 Previous champion: {prev_champion_id}")
            
            # Add new model to registry
            registry["models"][model_id] = new_model
            
            # Update current champion if model is deployed
            if new_model["lifecycle"]["is_champion"]:
                registry["current_champion"] = model_id
                logger.info(f"🏆 New champion: {model_id}")
            
            # Update registry metadata
            registry.update({
                "last_updated": datetime.now().isoformat(),
                "total_models": len(registry["models"]),
                "version": "1.0"
            })
            
            logger.info(f"✅ Registry updated:")
            logger.info(f"   Total models: {len(registry['models'])}")
            logger.info(f"   Current champion: {registry.get('current_champion', 'None')}")
            
            return registry
            
        except Exception as e:
            logger.error(f"❌ Error updating registry: {str(e)}")
            return registry

    def save_registry(self, registry: Dict) -> bool:
        """Save updated registry"""
        logger.info("\n💾 Saving Registry")
        logger.info("-" * 16)
        
        try:
            # Save main registry
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=4)
            logger.info(f"✅ Registry saved: {self.registry_file}")
            
            # Save model history (simplified view)
            history = {
                "last_updated": datetime.now().isoformat(),
                "model_timeline": []
            }
            
            # Create chronological timeline
            for model_id, model_data in registry["models"].items():
                history["model_timeline"].append({
                    "model_id": model_id,
                    "timestamp": model_data["created_timestamp"],
                    "model_type": model_data["model_type"],
                    "r2_score": model_data["performance"]["r2_score"],
                    "status": model_data["lifecycle"]["status"],
                    "is_champion": model_data["lifecycle"]["is_champion"]
                })
            
            # Sort by timestamp
            history["model_timeline"].sort(key=lambda x: x["timestamp"])
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
            logger.info(f"✅ History saved: {self.history_file}")
            
            # Save performance tracking
            performance_tracking = {
                "last_updated": datetime.now().isoformat(),
                "performance_evolution": [],
                "champion_history": []
            }
            
            # Track performance evolution
            for model_data in history["model_timeline"]:
                if model_data["r2_score"]:
                    performance_tracking["performance_evolution"].append({
                        "timestamp": model_data["timestamp"],
                        "model_id": model_data["model_id"],
                        "r2_score": model_data["r2_score"],
                        "model_type": model_data["model_type"]
                    })
                
                # Track champion changes
                if model_data["is_champion"]:
                    performance_tracking["champion_history"].append({
                        "timestamp": model_data["timestamp"],
                        "model_id": model_data["model_id"],
                        "model_type": model_data["model_type"],
                        "r2_score": model_data["r2_score"]
                    })
            
            with open(self.performance_file, 'w') as f:
                json.dump(performance_tracking, f, indent=4)
            logger.info(f"✅ Performance tracking saved: {self.performance_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving registry: {str(e)}")
            return False

    def generate_registry_report(self, registry: Dict) -> None:
        """Generate registry summary report"""
        logger.info("\n📊 Registry Summary Report")
        logger.info("-" * 26)
        
        try:
            models = registry.get("models", {})
            
            if not models:
                logger.info("ℹ️ No models in registry")
                return
            
            # Overall statistics
            total_models = len(models)
            deployed_models = sum(1 for m in models.values() if m["lifecycle"]["status"] == "deployed")
            current_champion = registry.get("current_champion")
            
            logger.info(f"📈 Registry Statistics:")
            logger.info(f"   Total models: {total_models}")
            logger.info(f"   Deployed models: {deployed_models}")
            logger.info(f"   Current champion: {current_champion or 'None'}")
            
            # Performance analysis
            r2_scores = [m["performance"]["r2_score"] for m in models.values() 
                        if m["performance"]["r2_score"] is not None]
            
            if r2_scores:
                logger.info(f"📊 Performance Analysis:")
                logger.info(f"   Best R² score: {max(r2_scores):.4f}")
                logger.info(f"   Average R² score: {sum(r2_scores)/len(r2_scores):.4f}")
                logger.info(f"   Latest R² score: {r2_scores[-1]:.4f}")
            
            # Model type distribution
            model_types = {}
            for model_data in models.values():
                model_type = model_data["model_type"]
                model_types[model_type] = model_types.get(model_type, 0) + 1
            
            logger.info(f"🔧 Model Types:")
            for model_type, count in model_types.items():
                logger.info(f"   {model_type}: {count} models")
            
            # Recent activity
            latest_model = max(models.values(), key=lambda x: x["created_timestamp"])
            logger.info(f"⏰ Latest Activity:")
            logger.info(f"   Latest model: {latest_model['model_id']}")
            logger.info(f"   Created: {latest_model['created_timestamp']}")
            logger.info(f"   Status: {latest_model['lifecycle']['status']}")
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {str(e)}")

    def run_registry_update(self) -> bool:
        """Run complete registry update process"""
        logger.info("\n🚀 STARTING REGISTRY UPDATE")
        logger.info("=" * 30)
        
        try:
            # Step 1: Load current registry
            registry = self.load_current_registry()
            
            # Step 2: Get model metadata
            model_data = self.get_model_metadata()
            if not model_data:
                return False
            
            # Step 3: Create model entry
            new_model = self.create_model_entry(model_data)
            if not new_model:
                return False
            
            # Step 4: Update registry
            updated_registry = self.update_registry(registry, new_model)
            
            # Step 5: Save registry
            if not self.save_registry(updated_registry):
                return False
            
            # Step 6: Generate report
            self.generate_registry_report(updated_registry)
            
            logger.info("\n🎉 REGISTRY UPDATE COMPLETED!")
            logger.info("✅ Model registry updated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Registry update failed: {str(e)}")
            return False

def main():
    """Main function for registry update"""
    registry = ModelRegistry()
    success = registry.run_registry_update()
    
    if success:
        print("\n🎯 REGISTRY UPDATE SUCCESS!")
        print("📝 Model registry updated with latest champion")
    else:
        print("\n❌ Registry update failed")
        print("📋 Check model metadata and registry files")

if __name__ == "__main__":
    main()
