"""
Test the AQI Peshawar Feature Store
==================================

Simple test script to verify the feature store is working correctly.
"""

from aqi_peshawar_feature_store import AQIPeshawarFeatureStore
import pandas as pd

def test_feature_store():
    print("🏪 TESTING AQI PESHAWAR FEATURE STORE")
    print("=" * 40)
    
    try:
        # Initialize feature store
        print("🔄 Initializing feature store connection...")
        fs = AQIPeshawarFeatureStore()
        print("✅ Feature store connected successfully!")
        print(f"📁 Project: {fs.project.name}")
        print(f"🏪 Feature Store: {fs.fs.name}")
        
        # Test getting available feature groups
        print("\n📋 Available Feature Groups:")
        feature_categories = ['weather', 'pollution', 'temporal', 'lag_features', 'advanced_features']
        
        available_groups = []
        for category in feature_categories:
            try:
                fg = fs.get_feature_group(category)
                print(f"   ✅ {category:.<20} Available (ID: {fg.id})")
                available_groups.append(category)
            except Exception as e:
                print(f"   ❌ {category:.<20} Not found")
        
        if available_groups:
            print(f"\n📊 Found {len(available_groups)} feature groups")
            
            # Test getting features from first available group
            test_category = available_groups[0]
            print(f"\n🔍 Testing feature retrieval from '{test_category}' group...")
            
            fg = fs.get_feature_group(test_category)
            schema = fg.schema
            print(f"   📈 Features in group: {len(schema)}")
            print(f"   📅 Last updated: {fg.created}")
            
            # Show first few features
            feature_names = [field.name for field in schema]
            print(f"   🔢 Sample features: {feature_names[:5]}...")
            
            print("\n✅ Feature store test completed successfully!")
            return True
        else:
            print("\n⚠️  No feature groups found")
            return False
            
    except Exception as e:
        print(f"\n❌ Feature store test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_feature_store()
    if success:
        print("\n🎉 Feature store is working correctly!")
    else:
        print("\n⚠️  Feature store needs attention")
