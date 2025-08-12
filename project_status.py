"""
Project Status Check
===================
Check if all services are running correctly.
"""

import requests
import time

def check_api_status():
    """Check if the realistic API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ REALISTIC API: RUNNING")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Realistic Calibration: {data.get('realistic_calibration', False)}")
            return True
        else:
            print(f"❌ REALISTIC API: ERROR ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ REALISTIC API: NOT RUNNING")
        print(f"   Error: {str(e)}")
        return False

def check_streamlit_status():
    """Check if Streamlit is accessible"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ STREAMLIT DASHBOARD: RUNNING")
            print("   URL: http://localhost:8501")
            return True
        else:
            print(f"❌ STREAMLIT DASHBOARD: ERROR ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ STREAMLIT DASHBOARD: NOT RUNNING")
        print(f"   Error: {str(e)}")
        return False

def test_current_aqi():
    """Test current AQI prediction"""
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
            current_aqi = data.get('current_aqi', 0)
            print(f"✅ CURRENT AQI: {current_aqi:.1f}")
            
            if 100 <= current_aqi <= 200:
                print("   ✅ REALISTIC for Peshawar")
            else:
                print("   ⚠️  May be outside expected range")
            
            return current_aqi
        else:
            print(f"❌ AQI PREDICTION: ERROR ({response.status_code})")
            return None
            
    except Exception as e:
        print(f"❌ AQI PREDICTION: FAILED")
        print(f"   Error: {str(e)}")
        return None

def main():
    """Run project status check"""
    print("🚀 PROJECT STATUS CHECK")
    print("=" * 25)
    print()
    
    # Check API
    api_running = check_api_status()
    print()
    
    # Check Streamlit
    streamlit_running = check_streamlit_status()
    print()
    
    # Test AQI if API is running
    if api_running:
        print("🧪 TESTING REALISTIC AQI:")
        current_aqi = test_current_aqi()
        print()
    
    # Summary
    print("📋 SUMMARY")
    print("=" * 11)
    
    if api_running:
        print("✅ Realistic API Server: OPERATIONAL")
    else:
        print("❌ Realistic API Server: NEEDS RESTART")
        print("   Fix: python phase5_realistic_api.py")
    
    if streamlit_running:
        print("✅ Streamlit Dashboard: OPERATIONAL")
    else:
        print("❌ Streamlit Dashboard: NEEDS RESTART")
        print("   Fix: streamlit run streamlit_app.py --server.port 8501")
    
    print()
    if api_running and streamlit_running:
        print("🎉 PROJECT IS FULLY OPERATIONAL!")
        print("📱 Access your dashboard: http://localhost:8501")
        print("🔧 API documentation: http://localhost:8000/docs")
    else:
        print("⚠️  Some services need to be restarted")

if __name__ == "__main__":
    main()

