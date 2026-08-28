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
from src.models.target_detector import detect_targets
from src.models.preprocessor import DataPreprocessor
from src.models.trainer import train_best_model
from src.models.evaluator import evaluate_model
from src.explainability.shap_engine import get_shap_values

def run_standard_pipeline(df, target_col, target_type="binary_classification"):
    """Helper to run a dataset through profiling, diagnosis, target mapping, preprocessor, and training."""
    profile = profile_dataset(df)
    kpis = discover_kpis(df, profile)
    patterns = analyze_patterns(df, profile)
    anoms = detect_anomalies(df, profile)
    targets_info = detect_targets(df, profile)
    
    # Preprocess
    preprocessor = DataPreprocessor(target_col=target_col, target_type=target_type)
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(df)
    
    # Train
    best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, target_type)
    metrics = evaluate_model(best_model, X_test, y_test, target_type)
    
    return {
        "profile": profile,
        "kpis": kpis,
        "patterns": patterns,
        "anoms": anoms,
        "targets_info": targets_info,
        "best_model": best_model,
        "metrics": metrics
    }

def test_edge_case_a_numerical_only():
    # A. Numerical-only dataset
    df = pd.DataFrame({
        "feat1": np.random.uniform(0.0, 10.0, size=50),
        "feat2": np.random.uniform(-5.0, 5.0, size=50),
        "target": np.random.choice([0, 1], size=50, p=[0.5, 0.5])
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None
    assert len(res["profile"]["data_types"]["categorical"]) == 0

def test_edge_case_b_categorical_heavy():
    # B. Categorical-heavy dataset
    df = pd.DataFrame({
        "cat1": np.random.choice(["Red", "Blue", "Green"], size=50),
        "cat2": np.random.choice(["High", "Medium", "Low"], size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None
    assert len(res["profile"]["data_types"]["numerical"]) == 0

def test_edge_case_c_missing_values():
    # C. Dataset with missing values (under 25% missing limit)
    np.random.seed(42)
    f1 = np.random.uniform(0.0, 10.0, size=50)
    f2 = np.random.uniform(0.0, 10.0, size=50)
    # Inject nulls
    f1[::5] = np.nan
    f2[::8] = np.nan
    
    df = pd.DataFrame({
        "feat1": f1,
        "feat2": f2,
        "target": np.random.choice([0, 1], size=50)
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None
    # Check that imputers ran inside pipeline
    assert res["profile"]["column_profiles"]["feat1"]["missing_count"] > 0

def test_edge_case_d_duplicate_rows():
    # D. Dataset with duplicate rows
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0] * 20,
        "feat2": [4.0, 5.0, 6.0] * 20,
        "target": [0, 1, 0] * 20
    })
    res = run_standard_pipeline(df, "target")
    assert res["profile"]["overview"]["duplicate_rows"] > 0
    assert res["best_model"] is not None

def test_edge_case_e_with_datetime():
    # E. Dataset with a datetime column
    dates = pd.date_range(start="2026-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "order_date": dates,
        "feat1": np.random.uniform(0.0, 100.0, size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    profile = profile_dataset(df)
    assert len(profile["data_types"]["datetime"]) == 1
    
    preprocessor = DataPreprocessor(target_col="target", target_type="binary_classification", date_col="order_date")
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(df)
    assert preprocessor.date_col == "order_date"
    assert X_train.shape[0] == 40 # 80% train

def test_edge_case_f_no_datetime():
    # F. Dataset with no datetime column
    df = pd.DataFrame({
        "feat1": np.random.uniform(0.0, 10.0, size=30),
        "target": np.random.choice([0, 1], size=30)
    })
    profile = profile_dataset(df)
    assert len(profile["inferred_columns"]["dates"]) == 0
    
    # Assert target runs without dates
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None

def test_edge_case_g_no_suitable_target():
    # G. Dataset with no suitable target (empty options or only identifiers)
    df = pd.DataFrame({
        "order_id": [f"ID-{i}" for i in range(20)],
        "transaction_code": [f"TX-{i}" for i in range(20)]
    })
    profile = profile_dataset(df)
    targets_info = detect_targets(df, profile)
    # Should fallback gracefully without crashing
    assert targets_info["best_target"] is None
    assert "No suitable supervised prediction target" in targets_info["reason"]

def test_edge_case_h_small_dataset():
    # H. Small dataset (e.g. 8 rows)
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "feat2": [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None

def test_edge_case_i_high_cardinality_identifiers():
    # I. Dataset containing high-cardinality identifiers
    df = pd.DataFrame({
        "user_uuid": [f"user-uuid-{i:04d}" for i in range(50)], # 100% unique strings
        "email_address": [f"email-{i}@company.com" for i in range(50)], # 100% unique strings
        "feat1": np.random.uniform(0.0, 10.0, size=50),
        "feat2": np.random.uniform(0.0, 10.0, size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    profile = profile_dataset(df)
    # Check that high cardinality identifier detection flags user_uuid and email_address
    assert "user_uuid" in profile["inferred_columns"]["ids"]
    assert "email_address" in profile["inferred_columns"]["ids"]
    
    preprocessor = DataPreprocessor(target_col="target", target_type="binary_classification")
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(df)
    # Verify both identifiers were correctly excluded from feature set
    assert "user_uuid" not in feat_names
    assert "email_address" not in feat_names

def test_edge_case_j_mixed_types_categorical():
    # J. Mixed int/string/None categorical column
    df = pd.DataFrame({
        "mixed_col": [1, "1", "unknown", np.nan, "text", 2, "text", 1, np.nan, "unknown"] * 5,
        "feat1": np.random.uniform(0.0, 10.0, size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None
    assert "mixed_col" in res["profile"]["data_types"]["categorical"]

def test_edge_case_k_pandas_nullable_dtypes():
    # K. Pandas nullable integer, float, and boolean dtypes
    df = pd.DataFrame({
        "null_int": pd.Series([1, 2, None, 4, 5] * 10, dtype="Int64"),
        "null_bool": pd.Series([True, False, None, True, False] * 10, dtype="boolean"),
        "null_float": pd.Series([1.5, None, 3.5, 4.5, 5.5] * 10, dtype="Float64"),
        "feat1": np.random.uniform(0.0, 10.0, size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    res = run_standard_pipeline(df, "target")
    assert res["best_model"] is not None

def test_edge_case_l_constant_columns():
    # L. Constant columns should be ignored
    df = pd.DataFrame({
        "const_col": ["constant"] * 50,
        "const_num": [42.0] * 50,
        "feat1": np.random.uniform(0.0, 10.0, size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    preprocessor = DataPreprocessor(target_col="target", target_type="binary_classification")
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(df)
    assert "const_col" not in feat_names
    assert "const_num" not in feat_names
    assert "feat1" in feat_names
