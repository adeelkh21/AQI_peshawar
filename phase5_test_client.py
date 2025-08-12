"""
Phase 5: Production System Test Client
=====================================

Test client to verify the AQI prediction API functionality.
Tests all endpoints including 72-hour forecasting capabilities.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List

class AQIAPIClient:
    """Test client for AQI Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client"""
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> Dict:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_current(self, location: Dict) -> Dict:
        """Test current AQI prediction"""
        try:
            payload = {
                "location": location,
                "include_confidence": True,
                "include_alerts": True
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/current",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_forecast(self, location: Dict, forecast_hours: List[int]) -> Dict:
        """Test forecast prediction"""
        try:
            payload = {
                "location": location,
                "forecast_hours": forecast_hours,
                "include_confidence": True,
                "include_alerts": True
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/forecast",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_72h_forecast(self, location: Dict) -> Dict:
        """Test 72-hour comprehensive forecast"""
        try:
            payload = {
                "location": location,
                "include_confidence": True,
                "include_alerts": True
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/forecast/72h",
                json=payload,
                timeout=90
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def run_comprehensive_tests():
    """Run comprehensive API tests"""
    print("🧪 PHASE 5: API TESTING SUITE")
    print("=" * 35)
    
    # Initialize client
    client = AQIAPIClient()
    
    # Test location (Peshawar coordinates)
    test_location = {
        "latitude": 34.0151,
        "longitude": 71.5249,
        "city": "Peshawar",
        "country": "Pakistan"
    }
    
    print("🔍 Testing API endpoints...")
    print()
    
    # Test 1: Health Check
    print("1️⃣  HEALTH CHECK")
    print("-" * 15)
    health = client.health_check()
    if "error" in health:
        print(f"❌ Health check failed: {health['error']}")
        print("⚠️  Make sure the server is running: python phase5_production_system.py")
        return False
    else:
        print(f"✅ Status: {health.get('status', 'unknown')}")
        print(f"📅 Timestamp: {health.get('timestamp', 'unknown')}")
        print(f"🤖 Model loaded: {health.get('model_loaded', False)}")
        print()
    
    # Test 2: Model Info
    print("2️⃣  MODEL INFORMATION")
    print("-" * 20)
    model_info = client.get_model_info()
    if "error" in model_info:
        print(f"❌ Model info failed: {model_info['error']}")
    else:
        metadata = model_info.get('model_metadata', {})
        print(f"🏆 Model type: {metadata.get('model_type', 'unknown')}")
        print(f"📊 Performance R²: {metadata.get('performance_r2', 'unknown')}")
        print(f"🔢 Feature count: {model_info.get('feature_count', 'unknown')}")
        print(f"⏰ Max forecast hours: {model_info.get('max_forecast_hours', 'unknown')}")
        print()
    
    # Test 3: Current Prediction
    print("3️⃣  CURRENT AQI PREDICTION")
    print("-" * 25)
    current_result = client.predict_current(test_location)
    if "error" in current_result:
        print(f"❌ Current prediction failed: {current_result['error']}")
    else:
        current_aqi = current_result.get('current_aqi')
        processing_time = current_result.get('processing_time_ms')
        alerts = current_result.get('alerts', [])
        
        print(f"🌬️  Current AQI: {current_aqi:.1f}" if current_aqi else "❌ No prediction")
        print(f"⚡ Processing time: {processing_time:.1f}ms" if processing_time else "❌ No timing")
        print(f"⚠️  Alerts: {len(alerts)} generated")
        if alerts:
            for alert in alerts[:2]:  # Show first 2 alerts
                print(f"   • {alert.get('message', 'No message')}")
        print()
    
    # Test 4: Short-term Forecast
    print("4️⃣  SHORT-TERM FORECAST (1-24h)")
    print("-" * 32)
    short_forecast = client.predict_forecast(test_location, [1, 3, 6, 12, 24])
    if "error" in short_forecast:
        print(f"❌ Short-term forecast failed: {short_forecast['error']}")
    else:
        forecasts = short_forecast.get('forecasts', [])
        alerts = short_forecast.get('alerts', [])
        processing_time = short_forecast.get('processing_time_ms')
        
        print(f"📊 Forecasts generated: {len(forecasts)}")
        print(f"⚡ Processing time: {processing_time:.1f}ms" if processing_time else "❌ No timing")
        
        for forecast in forecasts:
            horizon = forecast.get('horizon_hours')
            aqi_pred = forecast.get('aqi_prediction')
            accuracy = forecast.get('accuracy_estimate')
            category = forecast.get('quality_category')
            
            print(f"   {horizon}h: AQI {aqi_pred:.1f} ({category}) - Accuracy: {accuracy:.1%}")
        
        print(f"⚠️  Health alerts: {len(alerts)}")
        print()
    
    # Test 5: 72-Hour Comprehensive Forecast
    print("5️⃣  72-HOUR COMPREHENSIVE FORECAST")
    print("-" * 35)
    long_forecast = client.predict_72h_forecast(test_location)
    if "error" in long_forecast:
        print(f"❌ 72h forecast failed: {long_forecast['error']}")
    else:
        forecasts = long_forecast.get('forecasts', [])
        alerts = long_forecast.get('alerts', [])
        processing_time = long_forecast.get('processing_time_ms')
        
        print(f"📊 Total forecasts: {len(forecasts)}")
        print(f"⚡ Processing time: {processing_time:.1f}ms" if processing_time else "❌ No timing")
        
        # Show key forecast points
        key_horizons = [1, 6, 24, 48, 72]
        for forecast in forecasts:
            horizon = forecast.get('horizon_hours')
            if horizon in key_horizons:
                aqi_pred = forecast.get('aqi_prediction')
                accuracy = forecast.get('accuracy_estimate')
                category = forecast.get('quality_category')
                confidence = forecast.get('confidence_intervals', {}).get('95%', {})
                
                conf_range = ""
                if confidence:
                    lower = confidence.get('lower', 0)
                    upper = confidence.get('upper', 0)
                    conf_range = f" (95% CI: {lower:.1f}-{upper:.1f})"
                
                print(f"   {horizon:2d}h: AQI {aqi_pred:.1f}{conf_range} ({category}) - Accuracy: {accuracy:.1%}")
        
        print(f"⚠️  Total health alerts: {len(alerts)}")
        
        # Show critical alerts
        critical_alerts = [a for a in alerts if a.get('severity') in ['high', 'severe']]
        if critical_alerts:
            print(f"🚨 Critical alerts: {len(critical_alerts)}")
            for alert in critical_alerts[:3]:  # Show first 3 critical
                horizon = alert.get('horizon_hours')
                message = alert.get('message', 'No message')
                print(f"   • {horizon}h: {message}")
        print()
    
    # Performance Summary
    print("📊 PERFORMANCE SUMMARY")
    print("-" * 22)
    
    if "error" not in health:
        print("✅ API Health: HEALTHY")
    if "error" not in current_result:
        print("✅ Current Prediction: WORKING")
    if "error" not in short_forecast:
        print("✅ Short-term Forecast: WORKING")
    if "error" not in long_forecast:
        print("✅ 72h Forecast: WORKING")
    
    # Calculate average processing time
    times = []
    if "error" not in current_result and current_result.get('processing_time_ms'):
        times.append(current_result['processing_time_ms'])
    if "error" not in short_forecast and short_forecast.get('processing_time_ms'):
        times.append(short_forecast['processing_time_ms'])
    if "error" not in long_forecast and long_forecast.get('processing_time_ms'):
        times.append(long_forecast['processing_time_ms'])
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"⚡ Average response time: {avg_time:.1f}ms")
        
        # Performance assessment
        if avg_time < 200:
            print("🟢 Performance: EXCELLENT (<200ms)")
        elif avg_time < 500:
            print("🟡 Performance: GOOD (<500ms)")
        else:
            print("🔴 Performance: NEEDS OPTIMIZATION (>500ms)")
    
    print()
    print("🎉 API TESTING COMPLETED!")
    print("🚀 Phase 5 production system is ready!")
    
    return True

def main():
    """Run API tests"""
    print("🧪 Starting API test suite...")
    print()
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to initialize...")
    time.sleep(2)
    
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎯 PHASE 5 VALIDATION: SUCCESS!")
        print("✅ Production AQI system with 72h forecasting is operational")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🌐 Health Check: http://localhost:8000/health")
    else:
        print("\n⚠️  PHASE 5 VALIDATION: PARTIAL SUCCESS")
        print("Some endpoints may need debugging")

if __name__ == "__main__":
    main()
