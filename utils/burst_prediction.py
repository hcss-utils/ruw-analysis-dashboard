#!/usr/bin/env python
# coding: utf-8

"""
Burst prediction module implementing machine learning approaches to predict future bursts.
This module provides multiple implementations of predictive modeling for burst detection:
1. Time series forecasting using ARIMA and Prophet
2. Feature-based ML prediction using historical patterns
3. Ensemble prediction combining multiple approaches
4. Confidence scoring and validation for predictions
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Union, Tuple, Optional, Any, Set
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import joblib
import os
import json
from pathlib import Path

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

###########################################
# Time Series Forecasting
###########################################

def forecast_bursts_arima(
    time_series: pd.DataFrame,
    column: str = 'count',
    date_column: str = 'date',
    periods_ahead: int = 3,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    use_sarimax: bool = False
) -> pd.DataFrame:
    """
    Forecast future bursts using ARIMA or SARIMAX time series modeling.
    
    Args:
        time_series: DataFrame with time series data
        column: Column containing count data
        date_column: Column containing date data
        periods_ahead: Number of periods to forecast
        order: ARIMA order parameters (p, d, q)
        seasonal_order: SARIMAX seasonal order parameters (P, D, Q, s)
        use_sarimax: Whether to use SARIMAX instead of ARIMA
    
    Returns:
        pd.DataFrame: DataFrame with original data and forecasts
    """
    if time_series.empty:
        logging.warning("Empty time series provided for ARIMA forecasting")
        return pd.DataFrame()
    
    # Ensure data is sorted by date
    if date_column in time_series.columns:
        time_series = time_series.sort_values(date_column)
    
    # Create a copy of the input data
    result_df = time_series.copy()
    
    try:
        # Extract the target series
        y = result_df[column].values
        
        # Fit ARIMA or SARIMAX model
        if use_sarimax and seasonal_order is not None:
            model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
        else:
            model = ARIMA(y, order=order)
            fitted_model = model.fit()
        
        # Generate forecasts
        forecast_result = fitted_model.forecast(steps=periods_ahead)
        
        # Create forecast dates based on the date pattern
        last_date = result_df[date_column].iloc[-1]
        forecast_dates = []
        
        if isinstance(last_date, str):
            # Handle string dates - try to infer format
            try:
                last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
                for i in range(1, periods_ahead + 1):
                    next_date = last_date_obj + timedelta(days=i * 30)  # Assuming monthly periods
                    forecast_dates.append(next_date.strftime('%Y-%m-%d'))
            except:
                # If format can't be inferred, use generic period labels
                for i in range(1, periods_ahead + 1):
                    forecast_dates.append(f"Forecast Period {i}")
        elif isinstance(last_date, datetime):
            # Handle datetime objects
            for i in range(1, periods_ahead + 1):
                forecast_dates.append(last_date + timedelta(days=i * 30))  # Assuming monthly periods
        elif isinstance(last_date, (int, float)):
            # Handle numeric periods
            for i in range(1, periods_ahead + 1):
                forecast_dates.append(last_date + i)
        else:
            # Generic period labels
            for i in range(1, periods_ahead + 1):
                forecast_dates.append(f"Forecast Period {i}")
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            date_column: forecast_dates,
            column: forecast_result,
            'is_forecast': True
        })
        
        # Mark original data as not forecasts
        result_df['is_forecast'] = False
        
        # Combine original and forecast data
        combined_df = pd.concat([result_df, forecast_df], ignore_index=True)
        
        # Add confidence intervals for forecasts
        combined_df['forecast_lower'] = np.nan
        combined_df['forecast_upper'] = np.nan
        
        if use_sarimax:
            # Get prediction intervals from SARIMAX
            pred_int = fitted_model.get_forecast(steps=periods_ahead).conf_int(alpha=0.05)
            combined_df.loc[combined_df['is_forecast'], 'forecast_lower'] = pred_int.iloc[:, 0].values
            combined_df.loc[combined_df['is_forecast'], 'forecast_upper'] = pred_int.iloc[:, 1].values
        else:
            # Calculate approximate prediction intervals based on model residuals
            residuals = fitted_model.resid
            residual_std = np.std(residuals)
            z_value = stats.norm.ppf(0.975)  # 95% confidence interval
            
            # Apply simple confidence intervals
            forecast_values = combined_df.loc[combined_df['is_forecast'], column].values
            combined_df.loc[combined_df['is_forecast'], 'forecast_lower'] = forecast_values - z_value * residual_std
            combined_df.loc[combined_df['is_forecast'], 'forecast_upper'] = forecast_values + z_value * residual_std
            
            # Ensure lower bound isn't negative for count data
            combined_df.loc[combined_df['forecast_lower'] < 0, 'forecast_lower'] = 0
        
        return combined_df
    
    except Exception as e:
        logging.error(f"Error in ARIMA forecasting: {e}")
        return result_df

###########################################
# Feature-based Machine Learning Prediction
###########################################

def extract_burst_features(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    n_periods: int = 10,
    min_periods_required: int = 5
) -> pd.DataFrame:
    """
    Extract features from burst data for machine learning prediction.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        n_periods: Number of periods to consider for feature extraction
        min_periods_required: Minimum number of periods required for feature extraction
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    all_features = []
    
    for data_type, elements in burst_data.items():
        for element_name, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
                
            # Skip if not enough periods
            if len(df) < min_periods_required:
                continue
            
            # Sort by period to ensure chronological order
            sorted_df = df.sort_values('period')
            
            # Basic features
            element_features = {
                'data_type': data_type,
                'element': element_name,
                'latest_period': sorted_df['period'].iloc[-1],
                'latest_intensity': sorted_df['burst_intensity'].iloc[-1],
                'latest_count': sorted_df['count'].iloc[-1] if 'count' in sorted_df.columns else np.nan,
                'total_periods': len(sorted_df),
                'max_intensity': sorted_df['burst_intensity'].max(),
                'max_intensity_period': sorted_df.loc[sorted_df['burst_intensity'].idxmax(), 'period'] if not sorted_df.empty else None,
                'mean_intensity': sorted_df['burst_intensity'].mean(),
                'median_intensity': sorted_df['burst_intensity'].median(),
                'std_intensity': sorted_df['burst_intensity'].std(),
                'min_intensity': sorted_df['burst_intensity'].min(),
            }
            
            # Trend features
            if len(sorted_df) >= 3:
                # Calculate slope of intensities over time using numpy polyfit
                x = np.arange(len(sorted_df))
                y = sorted_df['burst_intensity'].values
                
                trend_coeffs = np.polyfit(x, y, 1)
                element_features['trend_slope'] = trend_coeffs[0]
                element_features['trend_intercept'] = trend_coeffs[1]
                
                # Calculate R-squared of the trend
                trend_line = np.polyval(trend_coeffs, x)
                ss_total = np.sum((y - np.mean(y))**2)
                ss_residual = np.sum((y - trend_line)**2)
                element_features['trend_r2'] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                
                # Calculate recent momentum (change in last 3 periods)
                if len(sorted_df) >= 3:
                    element_features['recent_momentum'] = sorted_df['burst_intensity'].iloc[-1] - sorted_df['burst_intensity'].iloc[-3]
                else:
                    element_features['recent_momentum'] = 0
                    
                # Acceleration (change in momentum)
                if len(sorted_df) >= 4:
                    prev_momentum = sorted_df['burst_intensity'].iloc[-2] - sorted_df['burst_intensity'].iloc[-4]
                    element_features['acceleration'] = element_features['recent_momentum'] - prev_momentum
                else:
                    element_features['acceleration'] = 0
                
                # Seasonality features
                if len(sorted_df) >= 4:
                    # Calculate autocorrelation at lag 1 and 2
                    intensities = sorted_df['burst_intensity'].values
                    acf_1 = np.corrcoef(intensities[:-1], intensities[1:])[0, 1]
                    element_features['autocorr_lag1'] = acf_1 if not np.isnan(acf_1) else 0
                    
                    if len(sorted_df) >= 5:
                        acf_2 = np.corrcoef(intensities[:-2], intensities[2:])[0, 1]
                        element_features['autocorr_lag2'] = acf_2 if not np.isnan(acf_2) else 0
                    else:
                        element_features['autocorr_lag2'] = 0
            else:
                # Default values for short time series
                element_features['trend_slope'] = 0
                element_features['trend_intercept'] = 0
                element_features['trend_r2'] = 0
                element_features['recent_momentum'] = 0
                element_features['acceleration'] = 0
                element_features['autocorr_lag1'] = 0
                element_features['autocorr_lag2'] = 0
            
            # Create lag features for the last n periods (or fewer if not available)
            for lag in range(1, min(n_periods, len(sorted_df) + 1)):
                lag_idx = -lag if lag <= len(sorted_df) else 0
                element_features[f'intensity_lag{lag}'] = sorted_df['burst_intensity'].iloc[lag_idx] if lag_idx < len(sorted_df) else 0
                element_features[f'count_lag{lag}'] = sorted_df['count'].iloc[lag_idx] if 'count' in sorted_df.columns and lag_idx < len(sorted_df) else 0
            
            # Add target (for training): next period burst intensity
            # This will be NaN and predicted by the model
            element_features['next_intensity'] = np.nan
            
            all_features.append(element_features)
    
    if not all_features:
        return pd.DataFrame()
    
    return pd.DataFrame(all_features)

def train_burst_prediction_model(
    feature_df: pd.DataFrame,
    target_column: str = 'next_intensity',
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: Optional[Dict] = None
) -> Tuple[Any, Dict, Dict]:
    """
    Train a machine learning model for burst prediction.
    
    Args:
        feature_df: DataFrame with features and target
        target_column: Column name for the prediction target
        model_type: Type of machine learning model ('random_forest' or 'gradient_boosting')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        model_params: Dictionary of model parameters
        
    Returns:
        Tuple containing: (trained model, feature scalers, model metrics)
    """
    if feature_df.empty:
        logging.warning("Empty feature DataFrame provided for model training")
        return None, {}, {}
    
    # Prepare training data
    # Keep only rows that have a value for the target
    train_df = feature_df[~feature_df[target_column].isna()].copy()
    
    if len(train_df) < 10:
        logging.warning(f"Insufficient data for training: {len(train_df)} rows")
        return None, {}, {}
    
    # Separate features and target
    X = train_df.drop(columns=[target_column, 'data_type', 'element', 'latest_period', 'max_intensity_period'])
    y = train_df[target_column]
    
    # Handle categorical variables and text columns
    # For this version, we simply drop them and rely on numeric features
    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    # Select model type
    if model_type == 'random_forest':
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': random_state
        }
        if model_params:
            default_params.update(model_params)
        model = RandomForestRegressor(**default_params)
    elif model_type == 'gradient_boosting':
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': random_state
        }
        if model_params:
            default_params.update(model_params)
        model = GradientBoostingRegressor(**default_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    # Sort feature importance
    metrics['feature_importance'] = dict(sorted(metrics['feature_importance'].items(), 
                                              key=lambda x: x[1], reverse=True))
    
    # Store scalers and column information
    preprocessing_info = {
        'scaler': scaler,
        'feature_columns': list(X.columns)
    }
    
    return model, preprocessing_info, metrics

def predict_future_bursts(
    feature_df: pd.DataFrame,
    model,
    preprocessing_info: Dict,
    forecast_periods: int = 1
) -> pd.DataFrame:
    """
    Predict future burst intensities using a trained model.
    
    Args:
        feature_df: DataFrame with current features
        model: Trained prediction model
        preprocessing_info: Dictionary with preprocessing information
        forecast_periods: Number of periods to forecast
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    if feature_df.empty or model is None:
        logging.warning("No features or model provided for prediction")
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    pred_df = feature_df.copy()
    
    # Scale features using the provided scaler
    scaler = preprocessing_info['scaler']
    feature_columns = preprocessing_info['feature_columns']
    
    # For multi-period forecasting, we'll predict iteratively
    all_predictions = []
    
    for period in range(1, forecast_periods + 1):
        # Prepare features for current prediction period
        # Remove rows that don't have enough data
        curr_pred_df = pred_df.copy()
        X = curr_pred_df[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Create prediction results
        for i, idx in enumerate(curr_pred_df.index):
            prediction_result = {
                'data_type': curr_pred_df.loc[idx, 'data_type'],
                'element': curr_pred_df.loc[idx, 'element'],
                'last_known_period': curr_pred_df.loc[idx, 'latest_period'],
                'forecast_period': period,
                'predicted_intensity': predictions[i],
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Add confidence based on model metrics and burst trend
            confidence_score = calculate_prediction_confidence(
                curr_pred_df.loc[idx], 
                predictions[i],
                model, 
                period
            )
            prediction_result['confidence'] = confidence_score
            
            # Apply confidence-based adjustments to prediction
            if confidence_score < 0.3:
                # For low confidence, regress toward the mean
                mean_intensity = curr_pred_df.loc[idx, 'mean_intensity']
                adjusted_prediction = 0.7 * mean_intensity + 0.3 * predictions[i]
                prediction_result['adjusted_prediction'] = adjusted_prediction
            else:
                prediction_result['adjusted_prediction'] = predictions[i]
            
            all_predictions.append(prediction_result)
        
        # If forecasting multiple periods, update features to enable next period prediction
        if period < forecast_periods:
            # Shift lag features
            for lag in range(min(10, len(feature_columns)), 1, -1):
                if f'intensity_lag{lag-1}' in feature_columns:
                    pred_df[f'intensity_lag{lag}'] = pred_df[f'intensity_lag{lag-1}']
                if f'count_lag{lag-1}' in feature_columns:
                    pred_df[f'count_lag{lag}'] = pred_df[f'count_lag{lag-1}']
            
            # Update lag1 with new prediction
            if 'intensity_lag1' in feature_columns:
                for i, idx in enumerate(pred_df.index):
                    # Use adjusted prediction for next period features
                    pred_df.loc[idx, 'intensity_lag1'] = all_predictions[i]['adjusted_prediction']
            
            # Update trend and momentum features
            if 'trend_slope' in feature_columns:
                for idx in pred_df.index:
                    lags = [pred_df.loc[idx, f'intensity_lag{i}'] for i in range(1, 6) if f'intensity_lag{i}' in pred_df.columns]
                    if len(lags) >= 3:
                        # Recalculate trend with new prediction
                        x = np.arange(len(lags))
                        trend_coeffs = np.polyfit(x, lags, 1)
                        pred_df.loc[idx, 'trend_slope'] = trend_coeffs[0]
                        pred_df.loc[idx, 'trend_intercept'] = trend_coeffs[1]
                        
                        # Update momentum
                        if len(lags) >= 3:
                            pred_df.loc[idx, 'recent_momentum'] = lags[0] - lags[2]
                            
                            # Update acceleration
                            if len(lags) >= 4:
                                prev_momentum = lags[1] - lags[3]
                                pred_df.loc[idx, 'acceleration'] = pred_df.loc[idx, 'recent_momentum'] - prev_momentum
    
    # Create DataFrame with predictions
    return pd.DataFrame(all_predictions)

def calculate_prediction_confidence(
    features: pd.Series,
    prediction: float,
    model,
    forecast_period: int
) -> float:
    """
    Calculate confidence score for a burst prediction.
    
    Args:
        features: Features for the predicted element
        prediction: Predicted burst intensity
        model: The prediction model
        forecast_period: How many periods ahead being predicted
        
    Returns:
        float: Confidence score (0-1)
    """
    # Base confidence starts at 0.8 and decreases with forecast period
    base_confidence = max(0.3, 0.8 - (forecast_period - 1) * 0.15)
    
    # Adjust for trend predictability
    trend_confidence = 0.7
    if 'trend_r2' in features:
        # Higher R² means more predictable trend
        trend_r2 = features['trend_r2']
        trend_confidence = min(0.9, 0.5 + trend_r2 * 0.5)
    
    # Adjust for data sufficiency
    data_sufficiency = 0.6
    if 'total_periods' in features:
        periods = features['total_periods']
        # More periods means higher confidence
        data_sufficiency = min(0.9, 0.4 + min(periods / 10, 1) * 0.5)
    
    # Adjust for prediction extremity
    extremity_factor = 1.0
    if prediction > 100:
        # Reduce confidence for extreme predictions
        extremity_factor = max(0.5, 1.0 - (prediction - 100) / 100)
    elif prediction < 0:
        # Reduce confidence for negative predictions
        extremity_factor = max(0.5, 1.0 + prediction / 20)
    
    # Calculate final confidence
    confidence = base_confidence * trend_confidence * data_sufficiency * extremity_factor
    
    # Ensure confidence is within 0-1 range
    return max(0, min(1, confidence))

###########################################
# Ensemble Prediction Methods
###########################################

def ensemble_burst_prediction(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    method: str = 'weighted',
    arima_weight: float = 0.3,
    ml_weight: float = 0.5,
    historical_weight: float = 0.2,
    forecast_periods: int = 3
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create ensemble predictions combining multiple prediction methods.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        method: Ensemble method ('weighted', 'stacking', or 'voting')
        arima_weight: Weight for ARIMA predictions in weighted ensemble
        ml_weight: Weight for ML predictions in weighted ensemble
        historical_weight: Weight for historical pattern predictions in weighted ensemble
        forecast_periods: Number of periods to forecast
        
    Returns:
        Dict: Predicted future burst data structured like the input
    """
    if not burst_data:
        logging.warning("No burst data provided for ensemble prediction")
        return {}
    
    # Extract features for ML-based prediction
    feature_df = extract_burst_features(burst_data)
    
    # Train prediction model
    model, preprocessing_info, metrics = train_burst_prediction_model(feature_df)
    
    # Initialize results structure
    predicted_bursts = {}
    
    for data_type, elements in burst_data.items():
        predicted_bursts[data_type] = {}
        
        for element_name, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns:
                continue
            
            # Get predictions from each method
            arima_predictions = []
            ml_predictions = []
            historical_predictions = []
            
            # 1. ARIMA prediction
            try:
                arima_df = forecast_bursts_arima(
                    df, 
                    column='burst_intensity', 
                    date_column='period',
                    periods_ahead=forecast_periods
                )
                # Extract forecasted values
                arima_forecasts = arima_df[arima_df['is_forecast']]
                for _, row in arima_forecasts.iterrows():
                    arima_predictions.append({
                        'period': row['period'],
                        'burst_intensity': max(0, min(100, row['burst_intensity'])),
                        'lower_bound': max(0, row['forecast_lower']),
                        'upper_bound': min(100, row['forecast_upper'])
                    })
            except Exception as e:
                logging.warning(f"ARIMA prediction failed for {element_name}: {e}")
                # Create dummy ARIMA predictions using simple extrapolation
                last_value = df['burst_intensity'].iloc[-1] if not df.empty else 0
                for i in range(1, forecast_periods + 1):
                    period_label = f"Forecast Period {i}"
                    arima_predictions.append({
                        'period': period_label,
                        'burst_intensity': last_value,
                        'lower_bound': max(0, last_value - 20),
                        'upper_bound': min(100, last_value + 20)
                    })
            
            # 2. ML prediction
            if model is not None:
                try:
                    # Filter feature_df to just this element
                    element_features = feature_df[
                        (feature_df['data_type'] == data_type) & 
                        (feature_df['element'] == element_name)
                    ]
                    
                    if not element_features.empty:
                        ml_pred_df = predict_future_bursts(
                            element_features, 
                            model,
                            preprocessing_info,
                            forecast_periods=forecast_periods
                        )
                        
                        for _, row in ml_pred_df.iterrows():
                            period_label = f"Forecast Period {row['forecast_period']}"
                            ml_predictions.append({
                                'period': period_label,
                                'burst_intensity': max(0, min(100, row['adjusted_prediction'])),
                                'confidence': row['confidence']
                            })
                except Exception as e:
                    logging.warning(f"ML prediction failed for {element_name}: {e}")
            
            # 3. Historical pattern prediction
            try:
                if not df.empty and len(df) >= 3:
                    historical_pattern = analyze_historical_pattern(df)
                    for i in range(1, forecast_periods + 1):
                        period_label = f"Forecast Period {i}"
                        hist_prediction = predict_from_pattern(
                            historical_pattern, 
                            current_values=df['burst_intensity'].values,
                            periods_ahead=i
                        )
                        historical_predictions.append({
                            'period': period_label,
                            'burst_intensity': max(0, min(100, hist_prediction))
                        })
                else:
                    # Not enough data for pattern prediction
                    for i in range(1, forecast_periods + 1):
                        period_label = f"Forecast Period {i}"
                        historical_predictions.append({
                            'period': period_label,
                            'burst_intensity': df['burst_intensity'].mean() if not df.empty else 0
                        })
            except Exception as e:
                logging.warning(f"Historical pattern prediction failed for {element_name}: {e}")
            
            # Combine predictions using the specified ensemble method
            ensemble_results = []
            
            for i in range(forecast_periods):
                period_label = f"Forecast Period {i+1}"
                
                arima_val = arima_predictions[i]['burst_intensity'] if i < len(arima_predictions) else 0
                ml_val = ml_predictions[i]['burst_intensity'] if i < len(ml_predictions) else 0
                hist_val = historical_predictions[i]['burst_intensity'] if i < len(historical_predictions) else 0
                
                if method == 'weighted':
                    # Weighted average of predictions
                    ensemble_val = (
                        arima_weight * arima_val +
                        ml_weight * ml_val +
                        historical_weight * hist_val
                    ) / (arima_weight + ml_weight + historical_weight)
                elif method == 'voting':
                    # Simple average
                    ensemble_val = (arima_val + ml_val + hist_val) / 3
                else:  # Default to weighted
                    ensemble_val = (
                        arima_weight * arima_val +
                        ml_weight * ml_val +
                        historical_weight * hist_val
                    ) / (arima_weight + ml_weight + historical_weight)
                
                # Calculate confidence bounds
                arima_lower = arima_predictions[i]['lower_bound'] if i < len(arima_predictions) else 0
                arima_upper = arima_predictions[i]['upper_bound'] if i < len(arima_predictions) else 100
                ml_confidence = ml_predictions[i]['confidence'] if i < len(ml_predictions) else 0.5
                
                # Wider bounds for lower confidence
                confidence_factor = ml_confidence if ml_confidence > 0 else 0.5
                bound_width = 30 * (1 - confidence_factor)
                lower_bound = max(0, ensemble_val - bound_width)
                upper_bound = min(100, ensemble_val + bound_width)
                
                # Create ensemble result
                ensemble_results.append({
                    'period': period_label,
                    'burst_intensity': ensemble_val,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'arima_prediction': arima_val,
                    'ml_prediction': ml_val,
                    'historical_prediction': hist_val,
                    'confidence': ml_confidence,
                    'is_forecast': True
                })
            
            # Create DataFrame with ensemble predictions
            ensemble_df = pd.DataFrame(ensemble_results)
            
            # Add count column with estimated counts
            if 'count' in df.columns:
                # Estimate counts based on average ratio of count to burst_intensity
                if not df.empty:
                    count_intensity_ratio = df['count'].sum() / df['burst_intensity'].sum() if df['burst_intensity'].sum() > 0 else 1
                    ensemble_df['count'] = ensemble_df['burst_intensity'] * count_intensity_ratio
                else:
                    ensemble_df['count'] = 0
            
            # Store in results
            predicted_bursts[data_type][element_name] = ensemble_df
    
    return predicted_bursts

def analyze_historical_pattern(
    time_series: pd.DataFrame,
    window_size: int = 3
) -> Dict[str, Any]:
    """
    Analyze historical patterns in time series data for prediction.
    
    Args:
        time_series: DataFrame with time series data
        window_size: Window size for pattern analysis
        
    Returns:
        Dict: Pattern analysis results
    """
    if time_series.empty or 'burst_intensity' not in time_series.columns:
        return {'valid': False}
    
    # Sort by period if available
    if 'period' in time_series.columns:
        time_series = time_series.sort_values('period')
    
    # Extract burst intensity values
    intensity_values = time_series['burst_intensity'].values
    
    # Need at least window_size+1 values for meaningful analysis
    if len(intensity_values) < window_size + 1:
        return {'valid': False}
    
    # Calculate trend
    x = np.arange(len(intensity_values))
    trend_coeffs = np.polyfit(x, intensity_values, 1)
    
    # Calculate periodicity via autocorrelation
    autocorr = []
    for lag in range(1, min(len(intensity_values) // 2, 5)):
        ac = np.corrcoef(intensity_values[:-lag], intensity_values[lag:])[0, 1]
        autocorr.append(ac if not np.isnan(ac) else 0)
    
    # Check if there's a periodic pattern
    has_periodicity = False
    dominant_period = 1
    
    if autocorr:
        max_autocorr = max(autocorr)
        dominant_period = autocorr.index(max_autocorr) + 1
        has_periodicity = max_autocorr > 0.4
    
    # Calculate seasonality (if enough data)
    seasonality = {'has_seasonality': False, 'season_values': []}
    if len(intensity_values) >= 8:
        # Detect if every n-th value follows a pattern
        for period in range(2, 5):
            if len(intensity_values) >= 2 * period:
                # Check correlation between values separated by period
                seasonal_corr = np.corrcoef(
                    intensity_values[:-period], 
                    intensity_values[period:]
                )[0, 1]
                
                if not np.isnan(seasonal_corr) and seasonal_corr > 0.3:
                    seasonality['has_seasonality'] = True
                    seasonality['period'] = period
                    
                    # Extract seasonal pattern
                    seasonal_pattern = []
                    for i in range(period):
                        phase_values = intensity_values[i::period]
                        seasonal_pattern.append(np.mean(phase_values))
                    
                    seasonality['seasonal_pattern'] = seasonal_pattern
                    break
    
    return {
        'valid': True,
        'trend_slope': trend_coeffs[0],
        'trend_intercept': trend_coeffs[1],
        'has_periodicity': has_periodicity,
        'dominant_period': dominant_period,
        'autocorrelation': autocorr,
        'seasonality': seasonality,
        'last_values': intensity_values[-window_size:].tolist()
    }

def predict_from_pattern(
    pattern: Dict[str, Any],
    current_values: np.ndarray,
    periods_ahead: int = 1
) -> float:
    """
    Predict future burst intensity based on historical pattern.
    
    Args:
        pattern: Pattern analysis results from analyze_historical_pattern
        current_values: Current burst intensity values
        periods_ahead: Number of periods to predict ahead
        
    Returns:
        float: Predicted burst intensity
    """
    if not pattern['valid'] or len(current_values) < 2:
        return current_values[-1] if len(current_values) > 0 else 0
    
    prediction = 0
    
    # 1. Apply trend component
    trend_prediction = pattern['trend_intercept'] + pattern['trend_slope'] * (len(current_values) + periods_ahead - 1)
    
    # 2. Apply seasonality if detected
    seasonal_component = 0
    if pattern['seasonality']['has_seasonality']:
        season_period = pattern['seasonality']['period']
        seasonal_pattern = pattern['seasonality']['seasonal_pattern']
        season_idx = (len(current_values) + periods_ahead - 1) % season_period
        seasonal_component = seasonal_pattern[season_idx] - np.mean(seasonal_pattern)
    
    # 3. Apply periodicity if detected
    periodic_component = 0
    if pattern['has_periodicity']:
        period = pattern['dominant_period']
        if len(current_values) >= period:
            # Use the value from dominant_period steps back
            historical_idx = -((periods_ahead % period) + period)
            if abs(historical_idx) <= len(current_values):
                base_value = current_values[historical_idx]
                
                # Adjust for trend
                steps_diff = period
                trend_adjustment = pattern['trend_slope'] * steps_diff
                
                periodic_component = base_value - (current_values[-1] + trend_adjustment)
    
    # 4. Last value persistence component
    persistence_weight = max(0, 1 - periods_ahead * 0.2)
    persistence_component = current_values[-1]
    
    # Combine components with weights
    if pattern['seasonality']['has_seasonality'] and pattern['has_periodicity']:
        # Use both seasonality and periodicity
        prediction = (
            0.3 * trend_prediction +
            0.3 * (current_values[-1] + seasonal_component) +
            0.2 * (current_values[-1] + periodic_component) +
            0.2 * persistence_component
        )
    elif pattern['seasonality']['has_seasonality']:
        # Use seasonality but not periodicity
        prediction = (
            0.4 * trend_prediction +
            0.4 * (current_values[-1] + seasonal_component) +
            0.2 * persistence_component
        )
    elif pattern['has_periodicity']:
        # Use periodicity but not seasonality
        prediction = (
            0.4 * trend_prediction +
            0.4 * (current_values[-1] + periodic_component) +
            0.2 * persistence_component
        )
    else:
        # Neither seasonality nor periodicity
        prediction = (
            0.5 * trend_prediction +
            0.5 * persistence_component
        )
    
    # Ensure prediction is within reasonable bounds
    return max(0, min(100, prediction))

###########################################
# Model Management and Persistence
###########################################

def save_prediction_model(
    model,
    preprocessing_info: Dict,
    metrics: Dict,
    model_name: str = 'burst_prediction_model',
    model_dir: str = './models'
) -> str:
    """
    Save a trained prediction model to disk.
    
    Args:
        model: Trained model
        preprocessing_info: Dictionary with preprocessing information
        metrics: Dictionary with model metrics
        model_name: Base name for the model files
        model_dir: Directory to save model files
        
    Returns:
        str: Path to the saved model directory
    """
    if model is None:
        logging.warning("No model provided to save")
        return ""
    
    # Create model directory if it doesn't exist
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_ts = f"{model_name}_{timestamp}"
    model_dir_path = model_path / model_name_ts
    model_dir_path.mkdir(exist_ok=True)
    
    try:
        # Save the model itself
        joblib.dump(model, model_dir_path / "model.joblib")
        
        # Save the scaler
        if 'scaler' in preprocessing_info:
            joblib.dump(preprocessing_info['scaler'], model_dir_path / "scaler.joblib")
        
        # Save preprocessing info and metrics as JSON
        # First remove non-serializable objects
        preproc_json = preprocessing_info.copy()
        if 'scaler' in preproc_json:
            del preproc_json['scaler']
        
        with open(model_dir_path / "preprocessing_info.json", 'w') as f:
            json.dump(preproc_json, f, indent=2)
        
        # Format metrics for JSON (handle numpy types)
        metrics_json = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                metrics_json[k] = {str(k2): float(v2) for k2, v2 in v.items()}
            elif isinstance(v, (np.int64, np.float64, np.float32, np.int32)):
                metrics_json[k] = float(v)
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (np.int64, np.float64, np.float32, np.int32)):
                metrics_json[k] = [float(x) for x in v]
            else:
                metrics_json[k] = v
                
        with open(model_dir_path / "model_metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_type': type(model).__name__,
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(model_dir_path / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
            
        logging.info(f"Model saved to {model_dir_path}")
        return str(model_dir_path)
        
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return ""

def load_prediction_model(model_path: str) -> Tuple[Any, Dict, Dict]:
    """
    Load a prediction model from disk.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple containing: (loaded model, preprocessing info, model metrics)
    """
    try:
        model_dir = Path(model_path)
        
        # Load model
        model = joblib.load(model_dir / "model.joblib")
        
        # Load scaler if exists
        preprocessing_info = {}
        if (model_dir / "scaler.joblib").exists():
            preprocessing_info['scaler'] = joblib.load(model_dir / "scaler.joblib")
        
        # Load preprocessing info
        if (model_dir / "preprocessing_info.json").exists():
            with open(model_dir / "preprocessing_info.json", 'r') as f:
                preproc_json = json.load(f)
                preprocessing_info.update(preproc_json)
        
        # Load metrics
        metrics = {}
        if (model_dir / "model_metrics.json").exists():
            with open(model_dir / "model_metrics.json", 'r') as f:
                metrics = json.load(f)
        
        return model, preprocessing_info, metrics
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, {}, {}

###########################################
# Integration with Dashboard
###########################################

def generate_prediction_summary(
    predicted_data: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Generate a summary of burst predictions for dashboard display.
    
    Args:
        predicted_data: Dictionary of predicted burst data
        
    Returns:
        pd.DataFrame: Summary of burst predictions
    """
    summary_rows = []
    
    for data_type, elements in predicted_data.items():
        for element_name, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
            
            # Calculate average predicted burst across periods
            avg_predicted_burst = df['burst_intensity'].mean()
            
            # Determine if predicted to be bursting in future
            max_predicted_burst = df['burst_intensity'].max()
            
            # Determine confidence bounds
            lower_bound = df['lower_bound'].min() if 'lower_bound' in df.columns else None
            upper_bound = df['upper_bound'].max() if 'upper_bound' in df.columns else None
            
            # Calculate overall prediction confidence
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else None
            
            # Determine trend direction
            if len(df) > 1:
                first_intensity = df['burst_intensity'].iloc[0]
                last_intensity = df['burst_intensity'].iloc[-1]
                if last_intensity > first_intensity * 1.1:
                    trend = "Increasing"
                elif last_intensity < first_intensity * 0.9:
                    trend = "Decreasing"
                else:
                    trend = "Stable"
            else:
                trend = "Unknown"
            
            # Create summary row
            summary_row = {
                'data_type': data_type,
                'element': element_name,
                'avg_predicted_burst': avg_predicted_burst,
                'max_predicted_burst': max_predicted_burst,
                'predicted_to_burst': max_predicted_burst >= 50,
                'trend': trend,
                'confidence': avg_confidence,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            summary_rows.append(summary_row)
    
    if not summary_rows:
        return pd.DataFrame()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by max_predicted_burst
    return summary_df.sort_values('max_predicted_burst', ascending=False)

def predict_burst_co_occurrences(
    predicted_bursts: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 50.0
) -> Dict[Tuple[str, str], Dict]:
    """
    Predict co-occurring bursts in future periods.
    
    Args:
        predicted_bursts: Dictionary of predicted burst data
        min_burst_intensity: Minimum intensity to consider as significant
        
    Returns:
        Dict: Mapping of element pairs to co-occurrence data
    """
    future_co_occurrences = {}
    
    # Extract all periods from predictions
    all_periods = set()
    for data_type, elements in predicted_bursts.items():
        for element, df in elements.items():
            if not df.empty and 'period' in df.columns:
                all_periods.update(df['period'].unique())
    
    # Create mapping of elements to periods with significant predicted bursts
    element_bursts = {}
    for data_type, elements in predicted_bursts.items():
        for element, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
            
            # Create qualified element name
            qualified_name = f"{data_type}:{element}"
            element_bursts[qualified_name] = set()
            
            # Record periods with significant predicted bursts
            for _, row in df.iterrows():
                if row['burst_intensity'] >= min_burst_intensity:
                    element_bursts[qualified_name].add(row['period'])
    
    # Find co-occurring elements
    from itertools import combinations
    
    for elem1, elem2 in combinations(element_bursts.keys(), 2):
        # Find common predicted burst periods
        common_periods = element_bursts[elem1].intersection(element_bursts[elem2])
        
        if common_periods:
            # Calculate co-occurrence strength using Jaccard similarity
            union_periods = element_bursts[elem1].union(element_bursts[elem2])
            strength = len(common_periods) / len(union_periods)
            
            # Store predicted co-occurrence
            pair_key = (elem1, elem2)
            future_co_occurrences[pair_key] = {
                'strength': strength,
                'predicted_common_periods': sorted(list(common_periods)),
                'num_common_periods': len(common_periods)
            }
    
    return future_co_occurrences

def calculate_comparison_with_historical(
    historical_bursts: Dict[str, Dict[str, pd.DataFrame]],
    predicted_bursts: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, Any]:
    """
    Compare predicted bursts with historical patterns.
    
    Args:
        historical_bursts: Dictionary of historical burst data
        predicted_bursts: Dictionary of predicted burst data
        
    Returns:
        Dict: Analysis of changes and emerging patterns
    """
    comparison = {
        'emerging_elements': [],
        'declining_elements': [],
        'persisting_elements': [],
        'overall_trend': None
    }
    
    # Calculate historical averages per element
    historical_avgs = {}
    for data_type, elements in historical_bursts.items():
        for element, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns:
                continue
            
            qualified_name = f"{data_type}:{element}"
            historical_avgs[qualified_name] = df['burst_intensity'].mean()
    
    # Calculate prediction averages per element
    prediction_avgs = {}
    for data_type, elements in predicted_bursts.items():
        for element, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns:
                continue
            
            qualified_name = f"{data_type}:{element}"
            prediction_avgs[qualified_name] = df['burst_intensity'].mean()
    
    # Compare predictions to historical data
    for qualified_name in set(historical_avgs.keys()).union(prediction_avgs.keys()):
        hist_avg = historical_avgs.get(qualified_name, 0)
        pred_avg = prediction_avgs.get(qualified_name, 0)
        
        data_type, element = qualified_name.split(':', 1)
        
        # Calculate relative change
        if hist_avg > 0:
            relative_change = (pred_avg - hist_avg) / hist_avg
        else:
            relative_change = float('inf') if pred_avg > 0 else 0
        
        element_data = {
            'data_type': data_type,
            'element': element,
            'historical_avg': hist_avg,
            'predicted_avg': pred_avg,
            'absolute_change': pred_avg - hist_avg,
            'relative_change': relative_change
        }
        
        # Categorize the element
        if relative_change > 0.3 or (hist_avg == 0 and pred_avg > 30):
            # Significantly increasing / emerging
            comparison['emerging_elements'].append(element_data)
        elif relative_change < -0.3:
            # Significantly decreasing
            comparison['declining_elements'].append(element_data)
        elif max(hist_avg, pred_avg) > 20:
            # Persisting and notable
            comparison['persisting_elements'].append(element_data)
    
    # Sort each category by the magnitude of change
    comparison['emerging_elements'].sort(key=lambda x: x['relative_change'], reverse=True)
    comparison['declining_elements'].sort(key=lambda x: x['relative_change'])
    comparison['persisting_elements'].sort(key=lambda x: x['predicted_avg'], reverse=True)
    
    # Determine overall trend
    if len(comparison['emerging_elements']) > len(comparison['declining_elements']) * 1.5:
        comparison['overall_trend'] = "Increasing activity with new emerging themes"
    elif len(comparison['declining_elements']) > len(comparison['emerging_elements']) * 1.5:
        comparison['overall_trend'] = "Decreasing activity with fading themes"
    elif len(comparison['persisting_elements']) > max(len(comparison['emerging_elements']), len(comparison['declining_elements'])):
        comparison['overall_trend'] = "Stable activity with consistent themes"
    else:
        comparison['overall_trend'] = "Mixed trends with evolving themes"
    
    return comparison