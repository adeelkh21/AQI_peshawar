"""
Test Realistic Predictions
=========================

Test the API to see if it's now showing realistic AQI values for Peshawar.
"""

import requests
import json

def test_current_aqi():
    """Test current AQI prediction"""
    
    payload = {
        "location": {
            "latitude": 34.0151,
            "longitude": 71.5249,
            "city": "Peshawar",
            "country": "Pakistan"
        },
        "include_confidence": True,
        "include_alerts": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/current",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            current_aqi = data.get('current_aqi', 0)
            
            print("🌬️ CURRENT AQI TEST")
            print("=" * 20)
            print(f"Predicted AQI: {current_aqi:.1f}")
            print(f"Expected AQI: ~134 (user reported)")
            print(f"Processing time: {data.get('processing_time_ms', 0):.1f}ms")
            
            if 100 <= current_aqi <= 160:
                print("✅ REALISTIC: AQI is in expected range for Peshawar")
            else:
                print("❌ UNREALISTIC: AQI should be around 100-160 for Peshawar")
            
            return current_aqi
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def test_forecast_consistency():
    """Test forecast consistency"""
    
    payload = {
        "location": {
            "latitude": 34.0151,
            "longitude": 71.5249,
            "city": "Peshawar",
            "country": "Pakistan"
        },
        "forecast_hours": [1, 6, 24, 72],
        "include_confidence": True,
        "include_alerts": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/forecast",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            forecasts = data.get('forecasts', [])
            
            print("\n🔮 FORECAST TEST")
            print("=" * 16)
            
            for forecast in forecasts:
                horizon = forecast['horizon_hours']
                aqi_pred = forecast['aqi_prediction']
                category = forecast['quality_category']
                
                print(f"{horizon:2d}h: AQI {aqi_pred:5.1f} ({category})")
            
            # Check if forecasts are consistent
            if len(forecasts) >= 2:
                first_aqi = forecasts[0]['aqi_prediction']
                if 80 <= first_aqi <= 160:
                    print("✅ REALISTIC: Forecast starts in expected range")
                else:
                    print("❌ UNREALISTIC: Forecast should start in range 80-160")
            
            return forecasts
        else:
            print(f"❌ Forecast API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Forecast request failed: {str(e)}")
        return None

def main():
    """Run realistic prediction tests"""
    print("🧪 TESTING REALISTIC PREDICTIONS")
    print("=" * 35)
    print("Expected: Current AQI ~134 for Peshawar")
    print()
    
    # Test current prediction
    current_aqi = test_current_aqi()
    
    # Test forecasts
    forecasts = test_forecast_consistency()
    
    print("\n📊 SUMMARY")
    print("=" * 10)
    
    if current_aqi and 100 <= current_aqi <= 160:
        print("✅ Current AQI: REALISTIC")
    else:
        print("❌ Current AQI: NEEDS FIXING")
        print("   The model should predict 100-160 AQI for current Peshawar conditions")
    
    if forecasts and len(forecasts) > 0:
        first_forecast = forecasts[0]['aqi_prediction']
        if 80 <= first_forecast <= 160:
            print("✅ Forecasts: REALISTIC")
        else:
            print("❌ Forecasts: NEED FIXING")
    
    print("\n💡 NEXT STEPS:")
    if not current_aqi or current_aqi < 120:
        print("1. Fix the feature engineering to reflect actual Peshawar conditions")
        print("2. Adjust model inputs to produce realistic AQI around 134")
        print("3. Ensure predictions are based on real environmental factors")

if __name__ == "__main__":
    main()
