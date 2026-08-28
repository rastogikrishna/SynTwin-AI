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
from src.decision.action_space import identify_controllable_variables
from src.decision.objective import calculate_fitness
from src.decision.genetic_optimizer import GeneticOptimizer
from src.decision.environment import TwinOptimizationEnv

@pytest.fixture
def mock_decision_df():
    np.random.seed(42)
    return pd.DataFrame({
        "id_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_num1": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        "feature_num2": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0],
        "target_binary": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "target_continuous": [10.5, 20.0, 10.5, 20.0, 10.5, 20.0, 10.5, 20.0, 10.5, 20.0]
    })

def test_identify_controllable_variables(mock_decision_df):
    profile = profile_dataset(mock_decision_df)
    
    # Target target_binary
    controllables = identify_controllable_variables(
        mock_decision_df, profile, target_col="target_binary"
    )
    
    features = [c["feature"] for c in controllables]
    assert "id_col" not in features
    assert "target_binary" not in features
    assert "feature_num1" in features
    assert "feature_num2" in features
    
    # Check bounds are loaded
    f1 = next(c for c in controllables if c["feature"] == "feature_num1")
    assert f1["min"] == 10.0
    assert f1["max"] == 20.0

def test_calculate_fitness():
    # Maximize
    assert calculate_fitness(10.5, "regression", "Maximize") == 10.5
    assert calculate_fitness(1, "binary_classification", "Maximize", prob_val=0.8) == 0.8
    
    # Minimize
    assert calculate_fitness(10.5, "regression", "Minimize") == -10.5
    assert calculate_fitness(0, "binary_classification", "Minimize", prob_val=0.2) == -0.2

def test_genetic_optimizer(mock_decision_df):
    profile = profile_dataset(mock_decision_df)
    preprocessor = DataPreprocessor(target_col="target_continuous", target_type="regression")
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_decision_df)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    controllable_vars = identify_controllable_variables(
        mock_decision_df, profile, target_col="target_continuous"
    )
    
    baseline_row = mock_decision_df.iloc[0]
    
    optimizer = GeneticOptimizer(
        model=model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        controllable_vars=controllable_vars,
        target_type="regression",
        objective_mode="Minimize",
        pop_size=10,
        generations=5
    )
    
    res = optimizer.optimize()
    
    assert "recommended_values" in res
    assert "baseline_prediction" in res
    assert "optimized_prediction" in res
    assert "predicted_improvement" in res
    assert len(res["history"]) == 5

def test_rl_environment(mock_decision_df):
    profile = profile_dataset(mock_decision_df)
    preprocessor = DataPreprocessor(target_col="target_binary", target_type="binary_classification")
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_decision_df)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    controllable_vars = identify_controllable_variables(
        mock_decision_df, profile, target_col="target_binary"
    )
    
    baseline_row = mock_decision_df.iloc[0]
    
    env = TwinOptimizationEnv(
        model=model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        controllable_vars=controllable_vars,
        target_type="binary_classification",
        objective_mode="Maximize"
    )
    
    state, info = env.reset()
    assert state.shape == (3,)
    
    action = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert next_state.shape == (3,)
    assert isinstance(reward, (float, np.floating))
    assert not terminated
    assert not truncated


def test_rl_robustness_nan_inf(mock_decision_df):
    profile = profile_dataset(mock_decision_df)
    preprocessor = DataPreprocessor(target_col="target_binary", target_type="binary_classification")
    X_train, X_test, y_train, y_test, _ = preprocessor.split_and_preprocess(mock_decision_df)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    controllable_vars = identify_controllable_variables(
        mock_decision_df, profile, target_col="target_binary"
    )
    
    # Introduce NaN and inf values into the baseline row and bounds
    baseline_row = mock_decision_df.iloc[0].copy()
    baseline_row["feature_num1"] = np.nan
    baseline_row["feature_num2"] = np.inf
    
    # Modify controllable bounds to contain infs and NaNs
    modified_controllables = []
    for var in controllable_vars:
        v = var.copy()
        if v["feature"] == "feature_num1":
            v["min"] = np.nan
            v["max"] = np.inf
        modified_controllables.append(v)
        
    env = TwinOptimizationEnv(
        model=model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        controllable_vars=modified_controllables,
        target_type="binary_classification",
        objective_mode="Maximize"
    )
    
    state, info = env.reset()
    assert np.isfinite(state).all()
    
    action = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert np.isfinite(next_state).all()
    assert np.isfinite(reward)

