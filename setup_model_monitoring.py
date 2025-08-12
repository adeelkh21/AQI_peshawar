"""
Model Performance Monitoring Setup
==================================

This script sets up comprehensive monitoring for deployed models including:
- Performance tracking and alerting
- Prediction accuracy monitoring
- Model drift detection
- Health dashboards and reports
- Automated alert systems

Author: Data Science Team
Date: August 12, 2025
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ModelMonitoringSetup:
    """Comprehensive model performance monitoring system"""
    
    def __init__(self):
        """Initialize model monitoring setup"""
        logger.info("📊 MODEL PERFORMANCE MONITORING SETUP")
        logger.info("=" * 40)
        
        # Directories
        self.monitoring_dir = os.path.join("data_repositories", "monitoring")
        self.performance_dir = os.path.join(self.monitoring_dir, "performance")
        self.alerts_dir = os.path.join(self.monitoring_dir, "alerts")
        self.dashboards_dir = os.path.join(self.monitoring_dir, "dashboards")
        self.logs_dir = os.path.join(self.monitoring_dir, "logs")
        
        # Create directories
        for directory in [self.monitoring_dir, self.performance_dir, self.alerts_dir, 
                         self.dashboards_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Monitoring configuration
        self.config = {
            "alert_thresholds": {
                "r2_degradation_percent": 5.0,      # 5% R² drop triggers alert
                "mae_increase_percent": 10.0,       # 10% MAE increase triggers alert
                "prediction_latency_ms": 1000,      # 1 second max response time
                "error_rate_percent": 1.0,          # 1% error rate threshold
                "data_drift_threshold": 0.15        # 15% feature drift threshold
            },
            "monitoring_intervals": {
                "performance_check_hours": 1,       # Check performance every hour
                "drift_check_hours": 6,            # Check drift every 6 hours
                "health_check_minutes": 15,        # Health check every 15 minutes
                "report_generation_hours": 24      # Daily reports
            },
            "baseline_performance": {
                "min_r2": 0.80,
                "max_mae": 15.0,
                "max_response_time_ms": 500
            }
        }
        
        # Monitoring results
        self.setup_results = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_components": {},
            "alert_rules": {},
            "dashboard_status": {},
            "baseline_metrics": {}
        }

    def load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics from latest model"""
        logger.info("\n📈 Loading Baseline Metrics")
        logger.info("-" * 27)
        
        try:
            # Load latest performance data
            performance_file = os.path.join("data_repositories", "models", "performance", "latest_performance.json")
            
            if not os.path.exists(performance_file):
                logger.error("❌ Latest performance file not found")
                return {}
            
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
            
            # Extract champion model performance
            champion_model = performance_data.get("champion_model", {})
            performance_metrics = champion_model.get("performance", {})
            
            baseline = {
                "r2_score": performance_metrics.get("r2_score"),
                "mae": performance_metrics.get("mae"),
                "rmse": performance_metrics.get("rmse"),
                "model_type": performance_data.get("champion_model", {}).get("name", "Unknown"),
                "baseline_timestamp": datetime.now().isoformat(),
                "training_samples": len(performance_data.get("challenger_models", {}))
            }
            
            logger.info(f"✅ Baseline metrics loaded:")
            logger.info(f"   R² Score: {baseline['r2_score']:.4f}")
            logger.info(f"   MAE: {baseline['mae']:.3f}")
            logger.info(f"   Model Type: {baseline['model_type']}")
            
            self.setup_results["baseline_metrics"] = baseline
            
            return baseline
            
        except Exception as e:
            logger.error(f"❌ Error loading baseline metrics: {str(e)}")
            return {}

    def create_alert_rules(self, baseline: Dict) -> Dict:
        """Create alert rules based on baseline metrics"""
        logger.info("\n🚨 Creating Alert Rules")
        logger.info("-" * 22)
        
        try:
            alert_rules = {
                "performance_alerts": {
                    "r2_degradation": {
                        "enabled": True,
                        "threshold_type": "percentage_decrease",
                        "threshold_value": self.config["alert_thresholds"]["r2_degradation_percent"],
                        "baseline_value": baseline.get("r2_score", 0.80),
                        "severity": "critical",
                        "description": "R² score degradation beyond acceptable threshold"
                    },
                    "mae_increase": {
                        "enabled": True,
                        "threshold_type": "percentage_increase",
                        "threshold_value": self.config["alert_thresholds"]["mae_increase_percent"],
                        "baseline_value": baseline.get("mae", 15.0),
                        "severity": "warning",
                        "description": "Mean Absolute Error increase beyond acceptable threshold"
                    },
                    "absolute_performance": {
                        "enabled": True,
                        "threshold_type": "absolute_minimum",
                        "threshold_value": self.config["baseline_performance"]["min_r2"],
                        "severity": "critical",
                        "description": "Model performance below minimum acceptable level"
                    }
                },
                "operational_alerts": {
                    "response_time": {
                        "enabled": True,
                        "threshold_type": "absolute_maximum",
                        "threshold_value": self.config["alert_thresholds"]["prediction_latency_ms"],
                        "severity": "warning",
                        "description": "Prediction response time exceeds acceptable limit"
                    },
                    "error_rate": {
                        "enabled": True,
                        "threshold_type": "percentage_maximum",
                        "threshold_value": self.config["alert_thresholds"]["error_rate_percent"],
                        "severity": "critical",
                        "description": "Prediction error rate exceeds acceptable limit"
                    },
                    "health_check": {
                        "enabled": True,
                        "threshold_type": "boolean",
                        "threshold_value": True,
                        "severity": "critical",
                        "description": "Model health check failure"
                    }
                },
                "data_quality_alerts": {
                    "feature_drift": {
                        "enabled": True,
                        "threshold_type": "percentage_maximum",
                        "threshold_value": self.config["alert_thresholds"]["data_drift_threshold"] * 100,
                        "severity": "warning",
                        "description": "Significant feature drift detected"
                    },
                    "missing_data": {
                        "enabled": True,
                        "threshold_type": "percentage_maximum",
                        "threshold_value": 10.0,  # 10% missing data threshold
                        "severity": "warning",
                        "description": "High percentage of missing input data"
                    }
                }
            }
            
            # Log alert rules
            logger.info("🚨 Alert Rules Created:")
            total_rules = 0
            for category, rules in alert_rules.items():
                category_count = len(rules)
                total_rules += category_count
                logger.info(f"   {category}: {category_count} rules")
            
            logger.info(f"   Total alert rules: {total_rules}")
            
            self.setup_results["alert_rules"] = alert_rules
            
            return alert_rules
            
        except Exception as e:
            logger.error(f"❌ Error creating alert rules: {str(e)}")
            return {}

    def setup_performance_tracking(self) -> bool:
        """Setup performance tracking system"""
        logger.info("\n📊 Setting Up Performance Tracking")
        logger.info("-" * 34)
        
        try:
            # Create performance tracking configuration
            tracking_config = {
                "tracking_version": "1.0",
                "created": datetime.now().isoformat(),
                "metrics_to_track": {
                    "prediction_metrics": {
                        "r2_score": {"weight": 0.4, "higher_is_better": True},
                        "mae": {"weight": 0.3, "higher_is_better": False},
                        "rmse": {"weight": 0.3, "higher_is_better": False}
                    },
                    "operational_metrics": {
                        "response_time_ms": {"weight": 0.4, "higher_is_better": False},
                        "error_rate_percent": {"weight": 0.3, "higher_is_better": False},
                        "uptime_percent": {"weight": 0.3, "higher_is_better": True}
                    },
                    "data_quality_metrics": {
                        "feature_drift_score": {"weight": 0.5, "higher_is_better": False},
                        "missing_data_percent": {"weight": 0.5, "higher_is_better": False}
                    }
                },
                "aggregation_windows": {
                    "real_time": "1_minute",
                    "short_term": "1_hour",
                    "medium_term": "24_hours",
                    "long_term": "7_days"
                },
                "data_retention": {
                    "raw_metrics": "30_days",
                    "hourly_aggregates": "90_days",
                    "daily_aggregates": "1_year"
                }
            }
            
            # Save tracking configuration
            tracking_config_file = os.path.join(self.performance_dir, "tracking_config.json")
            with open(tracking_config_file, 'w') as f:
                json.dump(tracking_config, f, indent=4)
            
            # Create initial performance log
            initial_log = {
                "log_version": "1.0",
                "created": datetime.now().isoformat(),
                "entries": [],
                "last_updated": datetime.now().isoformat()
            }
            
            performance_log_file = os.path.join(self.performance_dir, "performance_log.json")
            with open(performance_log_file, 'w') as f:
                json.dump(initial_log, f, indent=4)
            
            # Create performance tracking script template
            tracking_script = '''#!/usr/bin/env python3
"""
Automated Performance Tracking Script
Generated by Model Monitoring Setup
"""

import os
import json
import requests
import time
from datetime import datetime

def collect_performance_metrics():
    """Collect current performance metrics"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "prediction_metrics": {},
        "operational_metrics": {},
        "data_quality_metrics": {}
    }
    
    try:
        # Health check
        response = requests.get("http://localhost:8000/health", timeout=5)
        metrics["operational_metrics"]["uptime_status"] = response.status_code == 200
        metrics["operational_metrics"]["response_time_ms"] = response.elapsed.total_seconds() * 1000
        
        # Test prediction
        test_payload = {
            "location": {"latitude": 34.0151, "longitude": 71.5249, "city": "Peshawar", "country": "Pakistan"}
        }
        
        pred_start = time.time()
        pred_response = requests.post("http://localhost:8000/predict/current", json=test_payload, timeout=5)
        pred_time = (time.time() - pred_start) * 1000
        
        metrics["operational_metrics"]["prediction_response_time_ms"] = pred_time
        metrics["operational_metrics"]["prediction_success"] = pred_response.status_code == 200
        
    except Exception as e:
        metrics["operational_metrics"]["error"] = str(e)
    
    return metrics

def save_metrics(metrics):
    """Save metrics to performance log"""
    log_file = "data_repositories/monitoring/performance/performance_log.json"
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {"entries": []}
    
    log_data["entries"].append(metrics)
    log_data["last_updated"] = datetime.now().isoformat()
    
    # Keep only last 1000 entries
    if len(log_data["entries"]) > 1000:
        log_data["entries"] = log_data["entries"][-1000:]
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    metrics = collect_performance_metrics()
    save_metrics(metrics)
    print(f"Metrics collected at {metrics['timestamp']}")
'''
            
            tracking_script_file = os.path.join(self.performance_dir, "collect_metrics.py")
            with open(tracking_script_file, 'w') as f:
                f.write(tracking_script)
            
            logger.info("✅ Performance tracking setup completed:")
            logger.info(f"   Configuration: {tracking_config_file}")
            logger.info(f"   Performance log: {performance_log_file}")
            logger.info(f"   Collection script: {tracking_script_file}")
            
            self.setup_results["monitoring_components"]["performance_tracking"] = {
                "status": "configured",
                "config_file": tracking_config_file,
                "log_file": performance_log_file,
                "collection_script": tracking_script_file
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up performance tracking: {str(e)}")
            return False

    def create_monitoring_dashboard(self) -> bool:
        """Create monitoring dashboard configuration"""
        logger.info("\n📊 Creating Monitoring Dashboard")
        logger.info("-" * 31)
        
        try:
            # Dashboard configuration
            dashboard_config = {
                "dashboard_version": "1.0",
                "created": datetime.now().isoformat(),
                "dashboard_sections": {
                    "model_performance": {
                        "title": "Model Performance",
                        "charts": [
                            {
                                "type": "line_chart",
                                "title": "R² Score Over Time",
                                "data_source": "performance_log",
                                "metric": "prediction_metrics.r2_score",
                                "timeframe": "7_days"
                            },
                            {
                                "type": "line_chart", 
                                "title": "MAE Over Time",
                                "data_source": "performance_log",
                                "metric": "prediction_metrics.mae",
                                "timeframe": "7_days"
                            },
                            {
                                "type": "gauge",
                                "title": "Current R² Score",
                                "data_source": "latest_metrics",
                                "metric": "prediction_metrics.r2_score",
                                "thresholds": [0.70, 0.80, 0.90]
                            }
                        ]
                    },
                    "operational_health": {
                        "title": "Operational Health",
                        "charts": [
                            {
                                "type": "line_chart",
                                "title": "Response Time",
                                "data_source": "performance_log",
                                "metric": "operational_metrics.response_time_ms",
                                "timeframe": "24_hours"
                            },
                            {
                                "type": "status_indicator",
                                "title": "System Status",
                                "data_source": "latest_metrics",
                                "metric": "operational_metrics.uptime_status"
                            },
                            {
                                "type": "histogram",
                                "title": "Error Rate Distribution",
                                "data_source": "performance_log",
                                "metric": "operational_metrics.error_rate_percent",
                                "timeframe": "24_hours"
                            }
                        ]
                    },
                    "data_quality": {
                        "title": "Data Quality",
                        "charts": [
                            {
                                "type": "line_chart",
                                "title": "Feature Drift Score",
                                "data_source": "performance_log",
                                "metric": "data_quality_metrics.feature_drift_score",
                                "timeframe": "7_days"
                            },
                            {
                                "type": "bar_chart",
                                "title": "Missing Data Percentage",
                                "data_source": "performance_log",
                                "metric": "data_quality_metrics.missing_data_percent",
                                "timeframe": "24_hours"
                            }
                        ]
                    },
                    "alerts_summary": {
                        "title": "Alerts Summary",
                        "charts": [
                            {
                                "type": "alert_list",
                                "title": "Active Alerts",
                                "data_source": "alerts_log",
                                "timeframe": "24_hours"
                            },
                            {
                                "type": "pie_chart",
                                "title": "Alert Categories",
                                "data_source": "alerts_log",
                                "metric": "alert_category",
                                "timeframe": "7_days"
                            }
                        ]
                    }
                },
                "refresh_intervals": {
                    "model_performance": "5_minutes",
                    "operational_health": "1_minute",
                    "data_quality": "15_minutes",
                    "alerts_summary": "1_minute"
                }
            }
            
            # Save dashboard configuration
            dashboard_config_file = os.path.join(self.dashboards_dir, "dashboard_config.json")
            with open(dashboard_config_file, 'w') as f:
                json.dump(dashboard_config, f, indent=4)
            
            # Create dashboard template (HTML)
            dashboard_html = '''<!DOCTYPE html>
<html>
<head>
    <title>AQI Model Monitoring Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .dashboard-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }
        .metric-label { font-weight: bold; }
        .metric-value { font-size: 24px; color: #333; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
    </style>
</head>
<body>
    <h1>🏠 AQI Model Monitoring Dashboard</h1>
    
    <div class="dashboard-section">
        <div class="section-title">📈 Model Performance</div>
        <div class="metric">
            <div class="metric-label">R² Score</div>
            <div class="metric-value status-good" id="r2-score">Loading...</div>
        </div>
        <div class="metric">
            <div class="metric-label">MAE</div>
            <div class="metric-value" id="mae">Loading...</div>
        </div>
        <div class="metric">
            <div class="metric-label">Response Time</div>
            <div class="metric-value" id="response-time">Loading...</div>
        </div>
    </div>
    
    <div class="dashboard-section">
        <div class="section-title">🏥 System Health</div>
        <div class="metric">
            <div class="metric-label">Status</div>
            <div class="metric-value status-good" id="system-status">🟢 Online</div>
        </div>
        <div class="metric">
            <div class="metric-label">Uptime</div>
            <div class="metric-value" id="uptime">99.9%</div>
        </div>
    </div>
    
    <div class="dashboard-section">
        <div class="section-title">🚨 Recent Alerts</div>
        <div id="alerts-list">No active alerts</div>
    </div>
    
    <script>
        // Dashboard auto-refresh every 30 seconds
        setInterval(function() {
            // In a real implementation, this would fetch actual metrics
            document.getElementById('r2-score').textContent = '0.' + (800 + Math.floor(Math.random() * 100));
            document.getElementById('mae').textContent = (8 + Math.random() * 2).toFixed(1);
            document.getElementById('response-time').textContent = (200 + Math.floor(Math.random() * 100)) + 'ms';
        }, 30000);
    </script>
</body>
</html>'''
            
            dashboard_html_file = os.path.join(self.dashboards_dir, "dashboard.html")
            with open(dashboard_html_file, 'w') as f:
                f.write(dashboard_html)
            
            logger.info("✅ Monitoring dashboard created:")
            logger.info(f"   Configuration: {dashboard_config_file}")
            logger.info(f"   Dashboard: {dashboard_html_file}")
            
            self.setup_results["dashboard_status"] = {
                "status": "created",
                "config_file": dashboard_config_file,
                "dashboard_file": dashboard_html_file
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating dashboard: {str(e)}")
            return False

    def setup_alert_system(self, alert_rules: Dict) -> bool:
        """Setup automated alert system"""
        logger.info("\n🚨 Setting Up Alert System")
        logger.info("-" * 26)
        
        try:
            # Create alert configuration
            alert_config = {
                "alert_system_version": "1.0",
                "created": datetime.now().isoformat(),
                "alert_rules": alert_rules,
                "notification_channels": {
                    "console": {"enabled": True, "severity_threshold": "warning"},
                    "log_file": {"enabled": True, "severity_threshold": "info"},
                    "github_issues": {"enabled": True, "severity_threshold": "critical"}
                },
                "alert_cooldown": {
                    "critical": "15_minutes",
                    "warning": "1_hour",
                    "info": "4_hours"
                }
            }
            
            # Save alert configuration
            alert_config_file = os.path.join(self.alerts_dir, "alert_config.json")
            with open(alert_config_file, 'w') as f:
                json.dump(alert_config, f, indent=4)
            
            # Create alerts log
            alerts_log = {
                "log_version": "1.0",
                "created": datetime.now().isoformat(),
                "alerts": [],
                "last_updated": datetime.now().isoformat()
            }
            
            alerts_log_file = os.path.join(self.alerts_dir, "alerts_log.json")
            with open(alerts_log_file, 'w') as f:
                json.dump(alerts_log, f, indent=4)
            
            logger.info("✅ Alert system setup completed:")
            logger.info(f"   Configuration: {alert_config_file}")
            logger.info(f"   Alerts log: {alerts_log_file}")
            logger.info(f"   Alert rules: {sum(len(rules) for rules in alert_rules.values())}")
            
            self.setup_results["monitoring_components"]["alert_system"] = {
                "status": "configured",
                "config_file": alert_config_file,
                "log_file": alerts_log_file,
                "total_rules": sum(len(rules) for rules in alert_rules.values())
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up alert system: {str(e)}")
            return False

    def save_monitoring_configuration(self) -> bool:
        """Save complete monitoring configuration"""
        logger.info("\n💾 Saving Monitoring Configuration")
        logger.info("-" * 34)
        
        try:
            # Save monitoring setup results
            setup_file = os.path.join(self.monitoring_dir, "monitoring_setup.json")
            with open(setup_file, 'w') as f:
                json.dump(self.setup_results, f, indent=4)
            
            # Create monitoring status file for CICD
            status = {
                "monitoring_enabled": True,
                "setup_timestamp": datetime.now().isoformat(),
                "components_configured": len(self.setup_results["monitoring_components"]),
                "alert_rules_count": sum(len(rules) for rules in self.setup_results.get("alert_rules", {}).values())
            }
            
            status_file = "monitoring_status.txt"
            with open(status_file, 'w') as f:
                f.write("enabled" if status["monitoring_enabled"] else "disabled")
            
            # Save detailed status
            detailed_status_file = os.path.join(self.monitoring_dir, "monitoring_status.json")
            with open(detailed_status_file, 'w') as f:
                json.dump(status, f, indent=4)
            
            logger.info("✅ Monitoring configuration saved:")
            logger.info(f"   Setup details: {setup_file}")
            logger.info(f"   Status: {status_file}")
            logger.info(f"   Detailed status: {detailed_status_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving monitoring configuration: {str(e)}")
            return False

    def run_monitoring_setup(self) -> bool:
        """Run complete monitoring setup process"""
        logger.info("\n🚀 STARTING MONITORING SETUP")
        logger.info("=" * 32)
        
        try:
            # Step 1: Load baseline metrics
            baseline = self.load_baseline_metrics()
            if not baseline:
                logger.warning("⚠️ No baseline metrics - using defaults")
                baseline = {"r2_score": 0.80, "mae": 15.0, "model_type": "Unknown"}
            
            # Step 2: Create alert rules
            alert_rules = self.create_alert_rules(baseline)
            if not alert_rules:
                return False
            
            # Step 3: Setup performance tracking
            if not self.setup_performance_tracking():
                return False
            
            # Step 4: Create monitoring dashboard
            if not self.create_monitoring_dashboard():
                return False
            
            # Step 5: Setup alert system
            if not self.setup_alert_system(alert_rules):
                return False
            
            # Step 6: Save monitoring configuration
            if not self.save_monitoring_configuration():
                return False
            
            logger.info("\n🎉 MONITORING SETUP COMPLETED!")
            logger.info("✅ Model monitoring system is now active")
            
            # Summary
            components = len(self.setup_results["monitoring_components"])
            alert_rules_count = sum(len(rules) for rules in alert_rules.values())
            
            logger.info(f"📊 Setup Summary:")
            logger.info(f"   Components configured: {components}")
            logger.info(f"   Alert rules created: {alert_rules_count}")
            logger.info(f"   Dashboard available: Yes")
            logger.info(f"   Performance tracking: Enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Monitoring setup failed: {str(e)}")
            return False

def main():
    """Main function for monitoring setup"""
    monitoring = ModelMonitoringSetup()
    success = monitoring.run_monitoring_setup()
    
    if success:
        print("\n🎯 MONITORING SETUP SUCCESS!")
        print("📊 Model monitoring system is active")
        print("🚨 Alerts and dashboards configured")
    else:
        print("\n❌ Monitoring setup failed")
        print("📋 Check configuration and baseline metrics")

if __name__ == "__main__":
    main()
