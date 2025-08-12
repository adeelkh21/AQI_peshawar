"""
Test Streamlit Integration with Phase 5 API
==========================================

Simple test to verify the integration between Streamlit dashboard and the Phase 5 API.
"""

import requests
import json
from datetime import datetime

def test_api_connection():
    """Test Phase 5 API connection"""
    print("🧪 Testing Phase 5 API Integration")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health: CONNECTED")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model Loaded: {health_data.get('model_loaded')}")
        else:
            print("❌ API Health: FAILED")
            return False
    except Exception as e:
        print(f"❌ API Connection Failed: {str(e)}")
        return False
    
    # Test 2: Current prediction
    try:
        payload = {
            "location": {
                "latitude": 34.0151,
                "longitude": 71.5249,
                "city": "Peshawar",
                "country": "Pakistan"
            }
        }
        
        response = requests.post(f"{base_url}/predict/current", json=payload, timeout=10)
        if response.status_code == 200:
            current_data = response.json()
            aqi = current_data.get('current_aqi', 0)
            processing_time = current_data.get('processing_time_ms', 0)
            print("✅ Current Prediction: WORKING")
            print(f"   AQI: {aqi:.1f}")
            print(f"   Response Time: {processing_time:.1f}ms")
        else:
            print("❌ Current Prediction: FAILED")
            return False
    except Exception as e:
        print(f"❌ Current Prediction Failed: {str(e)}")
        return False
    
    # Test 3: 72h forecast
    try:
        response = requests.post(f"{base_url}/predict/forecast/72h", json=payload, timeout=15)
        if response.status_code == 200:
            forecast_data = response.json()
            forecasts = forecast_data.get('forecasts', [])
            print("✅ 72h Forecast: WORKING")
            print(f"   Forecast Points: {len(forecasts)}")
            if forecasts:
                print(f"   1h Prediction: {forecasts[0].get('aqi_prediction', 0):.1f}")
                print(f"   72h Prediction: {forecasts[-1].get('aqi_prediction', 0):.1f}")
        else:
            print("❌ 72h Forecast: FAILED")
            return False
    except Exception as e:
        print(f"❌ 72h Forecast Failed: {str(e)}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Phase 5 API is ready for Streamlit integration")
    return True

def test_streamlit_config():
    """Test Streamlit configuration"""
    print("\n🎛️ Testing Streamlit Configuration")
    print("=" * 35)
    
    try:
        from streamlit_config import StreamlitConfig
        
        print("✅ Configuration Loaded")
        print(f"   App Title: {StreamlitConfig.APP_TITLE}")
        print(f"   API URL: {StreamlitConfig.API_BASE_URL}")
        print(f"   Default Location: {StreamlitConfig.DEFAULT_LOCATION['city']}")
        print(f"   AQI Categories: {len(StreamlitConfig.AQI_CATEGORIES)}")
        
        # Test AQI category function
        test_aqi = 85
        category_info = StreamlitConfig.get_aqi_category(test_aqi)
        print(f"   Test AQI {test_aqi}: {category_info['category']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration Failed: {str(e)}")
        return False

def main():
    """Run integration tests"""
    print("🚀 STREAMLIT DASHBOARD INTEGRATION TEST")
    print("=" * 45)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test API
    api_ok = test_api_connection()
    
    # Test Streamlit config
    config_ok = test_streamlit_config()
    
    print("\n📊 TEST SUMMARY")
    print("=" * 16)
    
    if api_ok and config_ok:
        print("🎉 ALL SYSTEMS READY!")
        print("✅ Phase 5 API: Connected")
        print("✅ Streamlit Config: Loaded")
        print("✅ Integration: Ready")
        print()
        print("🌐 Access your dashboard:")
        print("   Streamlit Dashboard: http://localhost:8501")
        print("   API Documentation: http://localhost:8000/docs")
        print("   API Health: http://localhost:8000/health")
        print()
        print("🎯 Your end-to-end AQI prediction system is operational!")
        
    else:
        print("⚠️  SOME TESTS FAILED")
        if not api_ok:
            print("❌ API Issues: Start Phase 5 server with 'python phase5_production_system.py'")
        if not config_ok:
            print("❌ Config Issues: Check streamlit_config.py")

if __name__ == "__main__":
    main()
