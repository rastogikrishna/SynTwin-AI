import pytest
import numpy as np
import pandas as pd
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.explainability.shap_engine import get_shap_values, explain_globally, explain_locally

@pytest.fixture
def mock_classification_data():
    np.random.seed(42)
    X_train = np.random.randn(50, 4)
    y_train = np.random.randint(0, 2, size=50)
    X_test = np.random.randn(10, 4)
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D"]
    return X_train, y_train, X_test, feature_names

@pytest.fixture
def mock_regression_data():
    np.random.seed(42)
    X_train = np.random.randn(50, 4)
    y_train = np.random.randn(50)
    X_test = np.random.randn(10, 4)
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D"]
    return X_train, y_train, X_test, feature_names

def test_explainability_binary_classification(mock_classification_data):
    X_train, y_train, X_test, feature_names = mock_classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Global explanation checks
    global_drivers = explain_globally(model, X_train, X_test, feature_names, "binary_classification", class_idx=1)
    assert len(global_drivers) == 4
    assert global_drivers[0]["feature"] in feature_names
    assert "importance" in global_drivers[0]
    assert "relative_importance" in global_drivers[0]
    
    # Local explanation checks
    local_explanation = explain_locally(model, X_train, X_test, row_idx=0, feature_names=feature_names, target_type="binary_classification", class_idx=1)
    assert "base_value" in local_explanation
    assert "prediction_value" in local_explanation
    assert len(local_explanation["contributions"]) == 4
    assert len(local_explanation["positives"]) + len(local_explanation["negatives"]) == 4

def test_explainability_regression(mock_regression_data):
    X_train, y_train, X_test, feature_names = mock_regression_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    global_drivers = explain_globally(model, X_train, X_test, feature_names, "regression")
    assert len(global_drivers) == 4
    assert global_drivers[0]["feature"] in feature_names
    
    local_explanation = explain_locally(model, X_train, X_test, row_idx=0, feature_names=feature_names, target_type="regression")
    assert "base_value" in local_explanation
    assert "prediction_value" in local_explanation

def test_linear_regression_explanation(mock_regression_data):
    X_train, y_train, X_test, feature_names = mock_regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    global_drivers = explain_globally(model, X_train, X_test, feature_names, "regression")
    assert len(global_drivers) == 4

def test_logistic_regression_explanation(mock_classification_data):
    X_train, y_train, X_test, feature_names = mock_classification_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    global_drivers = explain_globally(model, X_train, X_test, feature_names, "binary_classification", class_idx=1)
    assert len(global_drivers) == 4

def test_empty_data_handling():
    X_train = np.empty((0, 4))
    X_test = np.empty((0, 4))
    model = LinearRegression()
    with pytest.raises(Exception):
        explain_globally(model, X_train, X_test, ["A", "B", "C", "D"], "regression")

def test_unsupported_model_handling():
    class MockModel:
        pass
    model = MockModel()
    with pytest.raises(Exception):
        explain_globally(model, np.random.randn(5, 4), np.random.randn(2, 4), ["A", "B", "C", "D"], "regression")
