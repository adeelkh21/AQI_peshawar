"""
Direct API Testing
==================
Test the API endpoints directly to see exactly what's being returned.
"""

import requests
import json

def test_current():
    """Test current AQI endpoint"""
    print("🧪 TESTING CURRENT AQI ENDPOINT")
    print("=" * 35)
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/current",
            json={
                "location": {
                    "latitude": 34.0151,
                    "longitude": 71.5249,
                    "city": "Peshawar",
                    "country": "Pakistan"
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Current AQI: {data.get('current_aqi', 'N/A')}")
            print(f"⚡ Processing time: {data.get('processing_time_ms', 'N/A')}ms")
            print(f"🏠 Location: {data.get('location', {}).get('city', 'N/A')}")
            return data.get('current_aqi')
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def test_forecast():
    """Test forecast endpoint"""
    print("\n🔮 TESTING FORECAST ENDPOINT")
    print("=" * 30)
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/forecast",
            json={
                "location": {
                    "latitude": 34.0151,
                    "longitude": 71.5249,
                    "city": "Peshawar",
                    "country": "Pakistan"
                },
                "forecast_hours": [1, 6, 24, 72]
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            forecasts = data.get('forecasts', [])
            
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Number of forecasts: {len(forecasts)}")
            
            for forecast in forecasts:
                horizon = forecast.get('horizon_hours', 0)
                aqi = forecast.get('aqi_prediction', 0)
                category = forecast.get('quality_category', 'Unknown')
                accuracy = forecast.get('accuracy_estimate', 0) * 100
                
                print(f"   {horizon:2d}h: AQI {aqi:5.1f} ({category}) - {accuracy:.0f}% accuracy")
            
            return forecasts
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def test_root():
    """Test root endpoint"""
    print("\n🏠 TESTING ROOT ENDPOINT")
    print("=" * 25)
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data.get('service', 'N/A')}")
            print(f"📊 Status: {data.get('status', 'N/A')}")
            print(f"🎯 Current AQI Range: {data.get('current_aqi_range', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print("🚀 DIRECT API TESTING")
    print("=" * 22)
    
    # Test root
    root_ok = test_root()
    
    # Test current
    current_aqi = test_current()
    
    # Test forecast
    forecasts = test_forecast()
    
    print("\n📋 SUMMARY")
    print("=" * 11)
    
    if root_ok:
        print("✅ Root endpoint: Working")
    else:
        print("❌ Root endpoint: Failed")
    
    if current_aqi and 120 <= current_aqi <= 160:
        print(f"✅ Current AQI: {current_aqi:.1f} (REALISTIC)")
    else:
        print(f"❌ Current AQI: {current_aqi} (UNREALISTIC)")
    
    if forecasts and len(forecasts) > 0:
        first_forecast = forecasts[0].get('aqi_prediction', 0)
        if 90 <= first_forecast <= 160:
            print(f"✅ Forecasts: Starting at {first_forecast:.1f} (REALISTIC)")
        else:
            print(f"❌ Forecasts: Starting at {first_forecast:.1f} (UNREALISTIC)")
    else:
        print("❌ Forecasts: No data received")

if __name__ == "__main__":
    main()
