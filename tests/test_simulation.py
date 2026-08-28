import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.data.profiler import profile_dataset
from src.models.preprocessor import DataPreprocessor
from src.simulation.scenario_engine import apply_scenario_changes, rank_simulation_variables
from src.simulation.twin_engine import run_twin_simulation
from src.simulation.comparator import compare_scenarios

@pytest.fixture
def mock_twin_df():
    np.random.seed(42)
    return pd.DataFrame({
        "id_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "feature_cat": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "target_binary": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "target_continuous": [10.5, 20.0, 15.2, 40.8, 50.1, 60.3, 70.2, 80.9, 90.0, 99.5]
    })

def test_scenario_engine_modifications():
    baseline_row = pd.Series({"feature_num": 100.0, "feature_cat": "A", "feature_bool": False})
    
    # 1. Percentage Change
    ch1 = [{"feature": "feature_num", "type": "percentage", "value": 15.0}]
    mod1 = apply_scenario_changes(baseline_row, ch1)
    assert mod1["feature_num"] == pytest.approx(115.0)
    
    # 2. Absolute Change
    ch2 = [{"feature": "feature_num", "type": "absolute", "value": 50.0}]
    mod2 = apply_scenario_changes(baseline_row, ch2)
    assert mod2["feature_num"] == pytest.approx(50.0)
    
    # 3. Categorical change
    ch3 = [{"feature": "feature_cat", "type": "category", "value": "B"}]
    mod3 = apply_scenario_changes(baseline_row, ch3)
    assert mod3["feature_cat"] == "B"
    
    # 4. Multi-variable changes
    ch4 = [
        {"feature": "feature_num", "type": "percentage", "value": -20.0},
        {"feature": "feature_cat", "type": "category", "value": "C"},
        {"feature": "feature_bool", "type": "boolean", "value": True}
    ]
    mod4 = apply_scenario_changes(baseline_row, ch4)
    assert mod4["feature_num"] == pytest.approx(80.0)
    assert mod4["feature_cat"] == "C"
    assert mod4["feature_bool"] is True

def test_rank_simulation_variables(mock_twin_df):
    profile = profile_dataset(mock_twin_df)
    ranked = rank_simulation_variables(mock_twin_df, profile, target_col="target_binary")
    
    features = [r["feature"] for r in ranked]
    assert "id_col" not in features
    assert "target_binary" not in features
    assert "feature_num" in features
    assert "feature_cat" in features

def test_twin_simulation_regression(mock_twin_df):
    profile = profile_dataset(mock_twin_df)
    preprocessor = DataPreprocessor(target_col="target_continuous", target_type="regression")
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_twin_df)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    baseline_row = mock_twin_df.iloc[0]
    changes = [{"feature": "feature_num", "type": "absolute", "value": 50.0}]
    
    sim_res = run_twin_simulation(model, preprocessor, baseline_row, changes, "regression", profile)
    
    assert "baseline_prediction" in sim_res
    assert "scenario_prediction" in sim_res
    assert "abs_difference" in sim_res
    assert "pct_difference" in sim_res

def test_twin_simulation_classification(mock_twin_df):
    profile = profile_dataset(mock_twin_df)
    preprocessor = DataPreprocessor(target_col="target_binary", target_type="binary_classification")
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_twin_df)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    baseline_row = mock_twin_df.iloc[0]
    changes = [{"feature": "feature_num", "type": "percentage", "value": 50.0}]
    
    sim_res = run_twin_simulation(model, preprocessor, baseline_row, changes, "binary_classification", profile)
    
    assert "baseline_prediction" in sim_res
    assert "scenario_prediction" in sim_res
    assert "baseline_probabilities" in sim_res
    assert "scenario_probabilities" in sim_res

def test_comparator():
    scenarios = [
        {
            "name": "Scenario A",
            "scenario_prediction": 120.0,
            "baseline_prediction": 100.0,
            "abs_difference": 20.0,
            "pct_difference": 20.0
        },
        {
            "name": "Scenario B",
            "scenario_prediction": 80.0,
            "baseline_prediction": 100.0,
            "abs_difference": -20.0,
            "pct_difference": -20.0
        }
    ]
    comparison_df = compare_scenarios(scenarios, "regression")
    assert len(comparison_df) == 2
    assert comparison_df.iloc[0]["Scenario Name"] == "Scenario A"
    assert comparison_df.iloc[1]["Scenario Name"] == "Scenario B"
    assert comparison_df.iloc[0]["Percentage Change"] == "+20.00%"
