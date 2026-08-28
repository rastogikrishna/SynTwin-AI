import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.profiler import profile_dataset
from src.forecasting.detector import detect_forecasting_eligibility
from src.forecasting.evaluator import score_forecast
from src.forecasting.forecaster import prepare_time_series, train_and_forecast

@pytest.fixture
def mock_ts_df():
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "order_date": dates.strftime("%Y-%m-%d"),
        "sales_metric": [10.0 + i*1.5 + np.random.randn()*0.5 for i in range(30)],
        "id_field": list(range(1, 31))
    })

def test_detector_valid(mock_ts_df):
    profile = profile_dataset(mock_ts_df)
    res = detect_forecasting_eligibility(mock_ts_df, profile)
    
    assert res["eligible"] is True
    assert res["recommended_date"] == "order_date"
    assert res["recommended_metric"] == "sales_metric"

def test_detector_no_date():
    df = pd.DataFrame({
        "sales": [float(i) for i in range(20)],
        "id": list(range(1, 21))
    })
    profile = profile_dataset(df)
    res = detect_forecasting_eligibility(df, profile)
    assert res["eligible"] is False
    assert "No suitable datetime column" in res["reason"]

def test_detector_insufficient_obs():
    df = pd.DataFrame({
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "sales": [10.0, 20.0, 15.0]
    })
    profile = profile_dataset(df)
    res = detect_forecasting_eligibility(df, profile)
    assert res["eligible"] is False
    assert "Insufficient temporal observations" in res["reason"]

def test_preparation_aggregation(mock_ts_df):
    df_dup = pd.concat([mock_ts_df, mock_ts_df.head(5)], ignore_index=True)
    resampled = prepare_time_series(df_dup, "order_date", "sales_metric", freq="D", agg_func="sum")
    
    assert len(resampled) == 30
    assert isinstance(resampled.index, pd.DatetimeIndex)

def test_evaluator_mape_zeros():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([5.0, 9.0, 22.0])
    
    metrics = score_forecast(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert abs(metrics["MAPE"] - 0.1) < 1e-4

def test_train_and_forecast(mock_ts_df):
    series = prepare_time_series(mock_ts_df, "order_date", "sales_metric", freq="D")
    res = train_and_forecast(series, horizon=7, freq_str="D")
    
    assert res["best_model"] in ["Naive", "Moving Average", "Exponential Smoothing"]
    assert "forecast_df" in res
    forecast_df = res["forecast_df"]
    assert len(forecast_df) == 7
    assert "Forecast" in forecast_df.columns
    assert "Lower Bound" in forecast_df.columns
    assert "Upper Bound" in forecast_df.columns
    assert np.all(forecast_df["Lower Bound"] <= forecast_df["Forecast"])
    assert np.all(forecast_df["Forecast"] <= forecast_df["Upper Bound"])
