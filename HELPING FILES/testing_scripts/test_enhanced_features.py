"""
Test Enhanced Features for 75% Target Achievement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_performance():
    """Test enhanced features performance"""
    
    print("🧪 TESTING ENHANCED FEATURES FOR 75% TARGET")
    print("=" * 50)
    
    # Load enhanced features
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Remove rows with too many NaN values (due to 72h lags)
    df_clean = df.dropna()
    
    print(f"📊 Enhanced Dataset:")
    print(f"   Original records: {len(df):,}")
    print(f"   Clean records: {len(df_clean):,}")
    print(f"   Total features: {len(df.columns)-2}")
    print(f"   Data coverage: {len(df_clean)/len(df)*100:.1f}%")
    
    if len(df_clean) < 1000:
        print("⚠️  Using original dataset with forward fill for NaN values")
        df_clean = df.fillna(method='ffill').fillna(method='bfill')
    
    # Prepare features
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'aqi_numeric']]
    X = df_clean[feature_cols].fillna(0)
    y = df_clean['aqi_numeric']
    
    print(f"\n🎯 Model Testing Results:")
    print("-" * 30)
    
    # 1. Random Forest with all features
    print("1. Random Forest (All Features):")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_all = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf_all.fit(X_train, y_train)
    
    y_pred_all = rf_all.predict(X_test)
    r2_all = r2_score(y_test, y_pred_all)
    mae_all = mean_absolute_error(y_test, y_pred_all)
    
    print(f"   R² Score: {r2_all:.3f} ({r2_all*100:.1f}%)")
    print(f"   MAE: {mae_all:.2f}")
    
    # 2. Random Forest with feature selection
    print("\n2. Random Forest (Top 50 Features):")
    selector = SelectKBest(f_regression, k=50)
    X_selected = selector.fit_transform(X, y)
    
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    rf_sel = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train_sel, y_train_sel)
    
    y_pred_sel = rf_sel.predict(X_test_sel)
    r2_sel = r2_score(y_test_sel, y_pred_sel)
    mae_sel = mean_absolute_error(y_test_sel, y_pred_sel)
    
    print(f"   R² Score: {r2_sel:.3f} ({r2_sel*100:.1f}%)")
    print(f"   MAE: {mae_sel:.2f}")
    
    # 3. Time Series Cross-Validation
    print("\n3. Time Series Cross-Validation:")
    tscv = TimeSeriesSplit(n_splits=3)
    r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
        X_train_ts, X_test_ts = X_selected[train_idx], X_selected[test_idx]
        y_train_ts, y_test_ts = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_ts = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        rf_ts.fit(X_train_ts, y_train_ts)
        y_pred_ts = rf_ts.predict(X_test_ts)
        
        fold_r2 = r2_score(y_test_ts, y_pred_ts)
        r2_scores.append(fold_r2)
        print(f"   Fold {fold+1}: R² = {fold_r2:.3f} ({fold_r2*100:.1f}%)")
    
    avg_r2_ts = np.mean(r2_scores)
    print(f"   Average: R² = {avg_r2_ts:.3f} ({avg_r2_ts*100:.1f}%)")
    
    # 4. Feature Importance Analysis
    print(f"\n📊 Top 15 Most Important Features:")
    feature_names = [feature_cols[i] for i in selector.get_support(indices=True)]
    importances = rf_sel.feature_importances_
    
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        print(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
    
    # 5. 3-Day Forecasting Assessment
    print(f"\n🔮 3-Day Forecasting Assessment:")
    print("-" * 35)
    
    # Check for 72h lag features
    lag_72h_features = [f for f in feature_cols if 'lag72h' in f]
    print(f"   72-hour lag features: {len(lag_72h_features)}")
    
    if len(lag_72h_features) > 0:
        print("   ✅ Ready for 3-day forecasting")
        print("   - Can use 72h lags as predictors")
        print("   - Multi-step prediction capability")
    else:
        print("   ❌ Not ready for 3-day forecasting")
    
    # 6. Progress Assessment
    print(f"\n📈 Progress to 75% Target:")
    print("-" * 30)
    
    best_r2 = max(r2_all, r2_sel, avg_r2_ts)
    gap = 75 - (best_r2 * 100)
    
    print(f"   Best R² achieved: {best_r2:.3f} ({best_r2*100:.1f}%)")
    print(f"   Target: 75.0%")
    print(f"   Remaining gap: {gap:.1f} percentage points")
    
    if gap <= 0:
        print("   🎉 TARGET ACHIEVED!")
    elif gap <= 5:
        print("   ✅ VERY CLOSE - Minor tuning needed")
    elif gap <= 15:
        print("   ⚠️  MODERATE - Need better algorithms")
    else:
        print("   🚨 SIGNIFICANT GAP - Need major improvements")
    
    # 7. Next Steps Recommendations
    print(f"\n💡 Next Steps:")
    print("-" * 15)
    
    if best_r2 >= 0.75:
        print("   🎯 PHASE 2 COMPLETE!")
        print("   - Move to Phase 3: Feature Store")
        print("   - Ready for production models")
    elif best_r2 >= 0.65:
        print("   🔧 Try Advanced Models:")
        print("   - XGBoost with hyperparameter tuning")
        print("   - LightGBM with feature selection")
        print("   - Neural Networks")
    else:
        print("   📊 More Feature Engineering:")
        print("   - External data sources (weather forecasts)")
        print("   - More complex lag patterns")
        print("   - Advanced time series features")
    
    return best_r2, gap, len(lag_72h_features)

if __name__ == "__main__":
    best_r2, gap, lag_features = test_enhanced_performance()
