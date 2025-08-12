"""
Streamlit AQI Dashboard Configuration
====================================

Configuration settings for the Streamlit AQI prediction dashboard.
"""

import os
from typing import Dict, List

class StreamlitConfig:
    """Configuration for Streamlit AQI Dashboard"""
    
    # App Configuration
    APP_TITLE = "🌬️ AQI Prediction Dashboard"
    APP_DESCRIPTION = "Real-time Air Quality Index monitoring and 72-hour forecasting system"
    APP_VERSION = "1.0.0"
    
    # Page Configuration
    PAGE_TITLE = "AQI Dashboard"
    PAGE_ICON = "🌬️"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # API Configuration
    API_BASE_URL = "http://localhost:8000"
    API_TIMEOUT = 30
    
    # Default Location (Peshawar)
    DEFAULT_LOCATION = {
        "latitude": 34.0151,
        "longitude": 71.5249,
        "city": "Peshawar",
        "country": "Pakistan"
    }
    
    # AQI Categories and Colors
    AQI_CATEGORIES = {
        "Good": {
            "range": (0, 50),
            "color": "#00E400",
            "text_color": "#000000",
            "description": "Air quality is satisfactory"
        },
        "Moderate": {
            "range": (51, 100),
            "color": "#FFFF00",
            "text_color": "#000000",
            "description": "Air quality is acceptable for most people"
        },
        "Unhealthy for Sensitive Groups": {
            "range": (101, 150),
            "color": "#FF7E00",
            "text_color": "#000000",
            "description": "Sensitive individuals may experience problems"
        },
        "Unhealthy": {
            "range": (151, 200),
            "color": "#FF0000",
            "text_color": "#FFFFFF",
            "description": "Everyone may experience problems"
        },
        "Very Unhealthy": {
            "range": (201, 300),
            "color": "#8F3F97",
            "text_color": "#FFFFFF",
            "description": "Health alert: everyone may experience serious effects"
        },
        "Hazardous": {
            "range": (301, 500),
            "color": "#7E0023",
            "text_color": "#FFFFFF",
            "description": "Health warning: emergency conditions"
        }
    }
    
    # Chart Configuration
    CHART_HEIGHT = 400
    CHART_WIDTH = 800
    
    # Forecast Configuration
    FORECAST_HORIZONS = [1, 3, 6, 12, 24, 48, 72]
    MAX_FORECAST_HOURS = 72
    
    # Update Intervals (seconds)
    REAL_TIME_UPDATE_INTERVAL = 300  # 5 minutes
    FORECAST_UPDATE_INTERVAL = 3600  # 1 hour
    
    # Cache Configuration
    CACHE_TTL = 300  # 5 minutes
    
    # Dashboard Sections
    DASHBOARD_SECTIONS = [
        "🏠 Home",
        "📊 Real-time Data", 
        "🔮 Forecasting",
        "📈 Trends & Analysis",
        "⚠️ Alerts & Notifications",
        "⚙️ Settings"
    ]
    
    # Color Palette
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "warning": "#ff7f0e",
        "danger": "#d62728",
        "info": "#17a2b8",
        "background": "#ffffff",
        "surface": "#f8f9fa"
    }
    
    @classmethod
    def get_aqi_category(cls, aqi_value: float) -> Dict:
        """Get AQI category information for a given value"""
        for category, info in cls.AQI_CATEGORIES.items():
            min_val, max_val = info["range"]
            if min_val <= aqi_value <= max_val:
                return {
                    "category": category,
                    "color": info["color"],
                    "text_color": info["text_color"],
                    "description": info["description"]
                }
        return {
            "category": "Unknown",
            "color": "#888888",
            "text_color": "#000000", 
            "description": "AQI value out of range"
        }
    
    @classmethod
    def get_aqi_color(cls, aqi_value: float) -> str:
        """Get color for AQI value"""
        return cls.get_aqi_category(aqi_value)["color"]
