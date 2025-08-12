"""
Real Data Integration for AQI Dashboard
======================================

Connect to real weather and pollution APIs to get actual live data
instead of dummy/simulated data.

APIs to integrate:
1. OpenWeatherMap - Current weather and pollution
2. IQAir - Real-time AQI data
3. Government pollution monitoring APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import os

logger = logging.getLogger(__name__)

class RealDataCollector:
    """Collect real weather and pollution data from live APIs"""
    
    def __init__(self):
        """Initialize with API keys"""
        # Get API keys from environment variables
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY', 'your_openweather_key_here')
        self.iqair_api_key = os.getenv('IQAIR_API_KEY', 'your_iqair_key_here')
        
        # API endpoints
        self.openweather_current_url = "http://api.openweathermap.org/data/2.5/weather"
        self.openweather_pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.openweather_forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        self.iqair_current_url = "http://api.airvisual.com/v2/city"
        
        logger.info("🌐 Real Data Collector initialized")
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather data from OpenWeatherMap"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.openweather_current_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'wind_direction': data.get('wind', {}).get('deg', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'clouds': data.get('clouds', {}).get('all', 0),
                    'weather_description': data['weather'][0]['description'],
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Weather data retrieved: {weather_data['temperature']}°C, {weather_data['humidity']}%")
                return weather_data
            
            else:
                logger.error(f"❌ Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting weather data: {str(e)}")
            return None
    
    def get_current_air_pollution(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current air pollution data from OpenWeatherMap"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key
            }
            
            response = requests.get(self.openweather_pollution_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'list' in data and len(data['list']) > 0:
                    pollution_data = data['list'][0]
                    components = pollution_data['components']
                    
                    # Calculate AQI from components (simplified US AQI calculation)
                    pm2_5 = components.get('pm2_5', 0)
                    pm10 = components.get('pm10', 0)
                    no2 = components.get('no2', 0)
                    o3 = components.get('o3', 0)
                    
                    # Simple AQI calculation based on PM2.5 (most critical)
                    # This is a simplified version - real AQI calculation is more complex
                    if pm2_5 <= 12:
                        aqi = pm2_5 * 4.17  # 0-50 range
                    elif pm2_5 <= 35.4:
                        aqi = 50 + (pm2_5 - 12) * 2.13  # 51-100 range
                    elif pm2_5 <= 55.4:
                        aqi = 100 + (pm2_5 - 35.4) * 2.5  # 101-150 range
                    elif pm2_5 <= 150.4:
                        aqi = 150 + (pm2_5 - 55.4) * 0.53  # 151-200 range
                    else:
                        aqi = min(500, 200 + (pm2_5 - 150.4) * 1.0)  # 201-500 range
                    
                    pollution_result = {
                        'aqi_calculated': round(aqi, 1),
                        'pm2_5': pm2_5,
                        'pm10': pm10,
                        'no2': no2,
                        'o3': o3,
                        'co': components.get('co', 0),
                        'so2': components.get('so2', 0),
                        'aqi_category': pollution_data['main']['aqi'],  # OpenWeather AQI (1-5 scale)
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"✅ Pollution data retrieved: AQI {pollution_result['aqi_calculated']}, PM2.5 {pm2_5}")
                    return pollution_result
                
                else:
                    logger.error("❌ No pollution data in response")
                    return None
            
            else:
                logger.error(f"❌ Pollution API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting pollution data: {str(e)}")
            return None
    
    def get_iqair_data(self, city: str, state: str, country: str) -> Optional[Dict]:
        """Get AQI data from IQAir (more accurate for AQI)"""
        try:
            params = {
                'city': city,
                'state': state,
                'country': country,
                'key': self.iqair_api_key
            }
            
            response = requests.get(self.iqair_current_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'success':
                    current = data['data']['current']
                    pollution = current['pollution']
                    weather = current['weather']
                    
                    iqair_data = {
                        'aqi_us': pollution['aqius'],  # US AQI
                        'aqi_china': pollution['aqicn'],  # China AQI
                        'main_pollutant': pollution['mainus'],
                        'temperature': weather['tp'],
                        'humidity': weather['hu'],
                        'pressure': weather['pr'],
                        'wind_speed': weather['ws'],
                        'wind_direction': weather['wd'],
                        'timestamp': pollution['ts']
                    }
                    
                    logger.info(f"✅ IQAir data retrieved: US AQI {iqair_data['aqi_us']}")
                    return iqair_data
                
                else:
                    logger.error(f"❌ IQAir API error: {data.get('message', 'Unknown error')}")
                    return None
            
            else:
                logger.error(f"❌ IQAir API HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting IQAir data: {str(e)}")
            return None
    
    def get_weather_forecast(self, lat: float, lon: float, hours: int = 120) -> Optional[List[Dict]]:
        """Get weather forecast data"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.openweather_forecast_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                forecast_list = []
                for item in data['list'][:hours//3]:  # API returns 3-hour intervals
                    forecast_item = {
                        'datetime': item['dt_txt'],
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item.get('wind', {}).get('speed', 0),
                        'clouds': item.get('clouds', {}).get('all', 0),
                        'weather': item['weather'][0]['description']
                    }
                    forecast_list.append(forecast_item)
                
                logger.info(f"✅ Weather forecast retrieved: {len(forecast_list)} periods")
                return forecast_list
            
            else:
                logger.error(f"❌ Forecast API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting weather forecast: {str(e)}")
            return None
    
    def get_comprehensive_current_data(self, location: Dict) -> Dict:
        """Get comprehensive current data from all sources"""
        lat = location['latitude']
        lon = location['longitude']
        city = location.get('city', 'Peshawar')
        country = location.get('country', 'Pakistan')
        
        result = {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'weather': None,
            'pollution': None,
            'current_aqi': None,
            'status': 'success'
        }
        
        # Try to get weather data
        weather_data = self.get_current_weather(lat, lon)
        if weather_data:
            result['weather'] = weather_data
            result['data_sources'].append('OpenWeatherMap-Weather')
        
        # Try to get pollution data from OpenWeatherMap
        pollution_data = self.get_current_air_pollution(lat, lon)
        if pollution_data:
            result['pollution'] = pollution_data
            result['current_aqi'] = pollution_data['aqi_calculated']
            result['data_sources'].append('OpenWeatherMap-Pollution')
        
        # Try to get more accurate AQI from IQAir (if available)
        if self.iqair_api_key != 'your_iqair_key_here':
            iqair_data = self.get_iqair_data(city, '', country)
            if iqair_data:
                result['iqair'] = iqair_data
                result['current_aqi'] = iqair_data['aqi_us']  # Use more accurate IQAir data
                result['data_sources'].append('IQAir')
        
        # If no real data available, provide fallback
        if not result['data_sources']:
            result['status'] = 'fallback'
            result['message'] = 'No real-time data available. Using fallback values.'
            # Provide realistic fallback data for Peshawar
            result['current_aqi'] = 134  # Based on user's observation
            result['weather'] = {
                'temperature': 28,
                'humidity': 60,
                'pressure': 1013,
                'wind_speed': 5.2
            }
        
        logger.info(f"📊 Comprehensive data collected: AQI {result.get('current_aqi', 'N/A')}")
        return result

def test_real_data():
    """Test real data collection"""
    print("🧪 Testing Real Data Collection")
    print("=" * 32)
    
    collector = RealDataCollector()
    
    # Test with Peshawar coordinates
    peshawar_location = {
        'latitude': 34.0151,
        'longitude': 71.5249,
        'city': 'Peshawar',
        'country': 'Pakistan'
    }
    
    print("🌍 Testing Peshawar data collection...")
    data = collector.get_comprehensive_current_data(peshawar_location)
    
    print(f"📊 Results:")
    print(f"   Current AQI: {data.get('current_aqi', 'N/A')}")
    print(f"   Data Sources: {', '.join(data.get('data_sources', ['None']))}")
    print(f"   Status: {data.get('status', 'unknown')}")
    
    if data.get('weather'):
        weather = data['weather']
        print(f"   Temperature: {weather.get('temperature', 'N/A')}°C")
        print(f"   Humidity: {weather.get('humidity', 'N/A')}%")
    
    if data.get('pollution'):
        pollution = data['pollution']
        print(f"   PM2.5: {pollution.get('pm2_5', 'N/A')} μg/m³")
        print(f"   PM10: {pollution.get('pm10', 'N/A')} μg/m³")
    
    return data

if __name__ == "__main__":
    test_real_data()
