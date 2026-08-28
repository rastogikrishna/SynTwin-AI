import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from src.forecasting.evaluator import score_forecast

def prepare_time_series(df: pd.DataFrame, date_col: str, metric_col: str, 
                        freq: str = 'D', agg_func: str = 'sum') -> pd.Series:
    """
    Parses dates, resamples to regular frequency, aggregates, and fills calendar gaps.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    df = df.dropna(subset=[date_col, metric_col])
    df = df.sort_values(date_col)
    
    # Resample and aggregate
    series = df.set_index(date_col)[metric_col]
    if agg_func == 'mean':
        resampled = series.resample(freq).mean()
    else:
        resampled = series.resample(freq).sum()
        
    # Fill gaps safely via interpolation & ffill/bfill
    resampled = resampled.interpolate(method='linear').ffill().bfill()
    return resampled

def train_and_forecast(series: pd.Series, horizon: int, freq_str: str) -> Dict[str, Any]:
    """
    Splits series chronologically, evaluates baseline models, fits the best on full history,
    and returns forecast values, confidence bounds, and comparison metrics.
    """
    y_values = series.values
    dates = series.index
    n = len(y_values)
    
    if n < 8:
        raise ValueError("Series length too short for training and validation splits.")
        
    # 1. Chronological Split (80% Train, 20% Validation)
    split_idx = int(n * 0.8)
    if n - split_idx < 2:
        split_idx = n - 2
        
    y_train, y_val = y_values[:split_idx], y_values[split_idx:]
    val_len = len(y_val)
    
    # 2. Fit and evaluate Naive baseline
    naive_val_pred = np.full(val_len, y_train[-1])
    naive_metrics = score_forecast(y_val, naive_val_pred)
    
    # 3. Fit and evaluate Moving Average (window=3)
    window = min(3, len(y_train))
    ma_val_pred = np.full(val_len, np.mean(y_train[-window:]))
    ma_metrics = score_forecast(y_val, ma_val_pred)
    
    # 4. Fit and evaluate Exponential Smoothing (ETS)
    ets_val_pred = np.full(val_len, y_train[-1])
    sp = 7 if freq_str == 'D' else 4 if freq_str == 'W' else 12
    ets_success = False
    
    if len(y_train) >= 2 * sp:
        try:
            model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=sp)
            fit_model = model.fit()
            ets_val_pred = fit_model.forecast(val_len)
            ets_success = True
        except Exception:
            pass
            
    if not ets_success:
        try:
            model = SimpleExpSmoothing(y_train)
            fit_model = model.fit()
            ets_val_pred = fit_model.forecast(val_len)
            ets_success = True
        except Exception:
            pass
            
    ets_metrics = score_forecast(y_val, ets_val_pred) if ets_success else {"MAE": float('inf'), "RMSE": float('inf'), "MAPE": float('inf')}
    
    # 5. Model Comparison
    comparison = {
        "Naive": naive_metrics,
        "Moving Average": ma_metrics,
    }
    if ets_success:
        comparison["Exponential Smoothing"] = ets_metrics
        
    # Rank by MAE
    best_model_name = min(comparison, key=lambda k: comparison[k]["MAE"])
    best_metrics = comparison[best_model_name]
    
    # 6. Fit best model on FULL history and forecast out-of-sample
    forecast_dates = pd.date_range(start=dates[-1], periods=horizon + 1, freq=freq_str)[1:]
    
    if best_model_name == "Naive":
        forecast = np.full(horizon, y_values[-1])
        residuals = y_values[1:] - y_values[:-1]
    elif best_model_name == "Moving Average":
        forecast = np.full(horizon, np.mean(y_values[-window:]))
        residuals = []
        for i in range(window, len(y_values)):
            residuals.append(y_values[i] - np.mean(y_values[i-window:i]))
        residuals = np.array(residuals) if residuals else (y_values[1:] - y_values[:-1])
    else: # Exponential Smoothing
        try:
            if len(y_values) >= 2 * sp:
                model = ExponentialSmoothing(y_values, trend='add', seasonal='add', seasonal_periods=sp)
            else:
                model = SimpleExpSmoothing(y_values)
            fit_model = model.fit()
            forecast = fit_model.forecast(horizon)
            residuals = fit_model.resid
        except Exception:
            # Emergency fallback to Naive if fit fails on full data
            best_model_name = "Naive"
            forecast = np.full(horizon, y_values[-1])
            residuals = y_values[1:] - y_values[:-1]
            
    # Calculate Confidence Intervals safely based on residuals
    resid_std = np.std(residuals) if len(residuals) > 0 else (np.std(y_values) * 0.1)
    if resid_std == 0:
        resid_std = 1e-3
        
    steps = np.arange(1, horizon + 1)
    # Multiplier scales with square root of forecast step
    margin = 1.96 * resid_std * np.sqrt(steps)
    
    lower_bound = forecast - margin
    upper_bound = forecast + margin
    
    # Clip lower bound at 0 if historical values are strictly positive
    if np.all(y_values >= 0):
        lower_bound = np.clip(lower_bound, 0, None)
        
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound
    }).set_index("Date")
    
    return {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "comparison": comparison,
        "forecast_df": forecast_df
    }
