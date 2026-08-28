import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.profiler import profile_dataset
from src.analysis.kpi_engine import discover_kpis
from src.analysis.pattern_engine import analyze_patterns
from src.analysis.anomaly_engine import detect_anomalies

@pytest.fixture
def base_df():
    return pd.DataFrame({
        "id_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "metric_a": [10.0, 12.0, 11.5, 9.5, 10.2, 50.0, 11.0, 10.8, 12.1, 9.9], # 50.0 is an IQR outlier
        "metric_b": [20.0, 24.0, 23.0, 19.0, 20.4, 100.0, 22.0, 21.6, 24.2, 19.8], # strongly correlated to metric_a
        "cat_skew": ["A", "A", "A", "A", "A", "B", "C", "D", "E", "F"], # DOMINATED BY "A" (50%)
        "date_col": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"]
    })

def test_kpi_engine(base_df):
    profile = profile_dataset(base_df)
    kpis = discover_kpis(base_df, profile)
    
    kpi_names = [k["name"] for k in kpis]
    assert "Total Records" in kpi_names
    
    assert any("Metric A" in k["name"] for k in kpis)
    assert any(k["type"] == "Volume" for k in kpis)
    assert any(k["type"] == "Average" for k in kpis)

def test_kpi_engine_no_numerical():
    df = pd.DataFrame({
        "cat_col": ["A", "B", "C"],
        "id_col": ["1", "2", "3"]
    })
    profile = profile_dataset(df)
    kpis = discover_kpis(df, profile)
    
    kpi_names = [k["name"] for k in kpis]
    assert "Total Records" in kpi_names
    numeric_kpis = [k for k in kpis if k["type"] in ["Volume", "Average", "Rate/Percentage"]]
    assert len(numeric_kpis) == 0

def test_pattern_engine_correlations(base_df):
    profile = profile_dataset(base_df)
    patterns = analyze_patterns(base_df, profile)
    
    corrs = patterns["correlations"]
    assert len(corrs) > 0
    assert corrs[0]["col1"] in ["metric_a", "metric_b"]
    assert corrs[0]["col2"] in ["metric_a", "metric_b"]
    assert corrs[0]["type"] == "positive"
    assert abs(corrs[0]["coefficient"] - 1.0) < 0.01

def test_pattern_engine_categorical(base_df):
    profile = profile_dataset(base_df)
    patterns = analyze_patterns(base_df, profile)
    
    cat_pats = patterns["categorical_patterns"]
    assert len(cat_pats) > 0
    assert cat_pats[0]["column"] == "cat_skew"
    assert cat_pats[0]["dominant_value"] == "A"
    assert cat_pats[0]["percentage"] == 50.0

def test_pattern_engine_temporal(base_df):
    profile = profile_dataset(base_df)
    patterns = analyze_patterns(base_df, profile)
    
    temp_pats = patterns["temporal_patterns"]
    assert len(temp_pats) > 0
    assert temp_pats[0]["date_column"] == "date_col"

def test_pattern_engine_no_dates():
    df = pd.DataFrame({
        "metric_a": [1, 2, 3],
        "metric_b": [4, 5, 6]
    })
    profile = profile_dataset(df)
    patterns = analyze_patterns(df, profile)
    
    assert len(patterns["temporal_patterns"]) == 0

def test_anomaly_engine_iqr(base_df):
    profile = profile_dataset(base_df)
    anoms = detect_anomalies(base_df, profile)
    
    assert anoms["columns_with_anomalies"] > 0
    cols = [c["column"] for c in anoms["column_details"]]
    assert "metric_a" in cols
    assert "metric_b" in cols
    
    detail_a = next(d for d in anoms["column_details"] if d["column"] == "metric_a")
    assert detail_a["outlier_count"] == 1
    assert detail_a["outlier_percentage"] == 10.0
    assert detail_a["method"] == "IQR (Tukey's Fences)"

def test_anomaly_engine_no_outliers():
    df = pd.DataFrame({
        "num_col": [10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0]
    })
    profile = profile_dataset(df)
    anoms = detect_anomalies(df, profile)
    
    assert anoms["total_anomalies"] == 0
    assert len(anoms["column_details"]) == 0

def test_anomaly_engine_missing_values():
    df = pd.DataFrame({
        "num_col": [1.0, 1.2, np.nan, 1.1, 0.9, 10.0, np.nan, 1.1, 1.0, 0.9] # 10.0 is outlier
    })
    profile = profile_dataset(df)
    anoms = detect_anomalies(df, profile)
    
    assert anoms["total_anomalies"] == 1
    assert anoms["column_details"][0]["column"] == "num_col"
    assert anoms["column_details"][0]["outlier_count"] == 1
