import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.profiler import profile_dataset
from src.models.target_detector import detect_targets
from src.models.preprocessor import DataPreprocessor
from src.models.trainer import train_best_model
from src.models.evaluator import evaluate_model

@pytest.fixture
def mock_classification_df():
    return pd.DataFrame({
        "id_field": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_num": [1.5, 2.5, 3.1, 4.0, 5.2, np.nan, 7.1, 8.0, 9.2, 10.1],
        "feature_cat": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "target_binary": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "date_field": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]
    })

@pytest.fixture
def mock_regression_df():
    return pd.DataFrame({
        "id_field": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_num": [1.5, 2.5, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.2, 10.1],
        "feature_cat": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "target_continuous": [10.5, 20.0, 15.2, 40.8, 50.1, 60.3, 70.2, 80.9, 90.0, 99.5]
    })

def test_target_detector_binary(mock_classification_df):
    profile = profile_dataset(mock_classification_df)
    results = detect_targets(mock_classification_df, profile)
    
    assert results["best_target"] == "target_binary"
    assert results["target_type"] == "binary_classification"
    assert results["confidence"] == 0.90

def test_target_detector_regression(mock_regression_df):
    profile = profile_dataset(mock_regression_df)
    results = detect_targets(mock_regression_df, profile)
    
    assert results["best_target"] == "target_continuous"
    assert results["target_type"] == "regression"
    assert results["confidence"] == 0.85

def test_target_detector_id_exclusion():
    df = pd.DataFrame({
        "id_field": [1, 2, 3],
        "const_field": ["A", "A", "A"]
    })
    profile = profile_dataset(df)
    results = detect_targets(df, profile)
    assert results["best_target"] is None
    assert "No suitable supervised prediction target" in results["reason"]

def test_preprocessor_classification(mock_classification_df):
    preprocessor = DataPreprocessor(
        target_col="target_binary",
        target_type="binary_classification",
        date_col="date_field"
    )
    
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(mock_classification_df)
    
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    
    assert "target_binary" not in feat_names
    assert "date_field" not in feat_names
    assert "date_field_year" in feat_names
    assert "feature_num" in feat_names
    assert any("feature_cat" in name for name in feat_names)

def test_model_training_classification(mock_classification_df):
    preprocessor = DataPreprocessor(
        target_col="target_binary",
        target_type="binary_classification"
    )
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_classification_df)
    
    best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, "binary_classification")
    
    assert best_name in ["Logistic Regression", "Random Forest Classifier"]
    assert "Logistic Regression" in scores
    assert "Random Forest Classifier" in scores

def test_model_training_regression(mock_regression_df):
    preprocessor = DataPreprocessor(
        target_col="target_continuous",
        target_type="regression"
    )
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_regression_df)
    
    best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, "regression")
    
    assert best_name in ["Linear Regression", "Random Forest Regressor"]
    assert "Linear Regression" in scores
    assert "Random Forest Regressor" in scores

def test_model_evaluation_classification(mock_classification_df):
    preprocessor = DataPreprocessor(
        target_col="target_binary",
        target_type="binary_classification"
    )
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_classification_df)
    
    best_model, _, _ = train_best_model(X_train, y_train, X_test, y_test, "binary_classification")
    metrics = evaluate_model(best_model, X_test, y_test, "binary_classification")
    
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1 Score" in metrics

def test_model_evaluation_regression(mock_regression_df):
    preprocessor = DataPreprocessor(
        target_col="target_continuous",
        target_type="regression"
    )
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_regression_df)
    
    best_model, _, _ = train_best_model(X_train, y_train, X_test, y_test, "regression")
    metrics = evaluate_model(best_model, X_test, y_test, "regression")
    
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2 Score" in metrics
