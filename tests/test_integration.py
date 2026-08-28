import pytest
import pandas as pd
import numpy as np
import pathlib
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_data
from src.data.profiler import profile_dataset
from src.analysis.kpi_engine import discover_kpis
from src.analysis.pattern_engine import analyze_patterns
from src.analysis.anomaly_engine import detect_anomalies
from src.models.target_detector import detect_targets
from src.models.preprocessor import DataPreprocessor
from src.models.trainer import train_best_model
from src.models.evaluator import evaluate_model
from src.explainability.shap_engine import get_shap_values, explain_globally, explain_locally
from src.simulation.scenario_engine import rank_simulation_variables, apply_scenario_changes
from src.simulation.twin_engine import run_twin_simulation
from src.decision.action_space import identify_controllable_variables
from src.decision.genetic_optimizer import GeneticOptimizer
from src.decision.environment import TwinOptimizationEnv
from src.genai.context_builder import build_context, format_context_to_text
from src.genai.llm_client import LLMClient
from src.genai.rag_engine import LocalVectorStore
from src.genai.response_generator import generate_grounded_response

@pytest.fixture
def sample_integration_df():
    np.random.seed(42)
    # Generate 15 rows of synthetic retail style data
    return pd.DataFrame({
        "order_id": list(range(1, 16)),
        "sales": [100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0],
        "shipping_cost": [10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0],
        "scheduled_days": [3, 4, 3, 4, 3, 3, 4, 4, 3, 4, 3, 3, 4, 3, 4],
        "real_days": [3, 5, 2, 4, 3, 4, 4, 5, 3, 4, 2, 3, 5, 3, 4],
        "late_risk": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0] # binary target
    })

def test_pipeline_integration_step_1_to_5(sample_integration_df):
    # 1. Profiling
    profile = profile_dataset(sample_integration_df)
    assert profile["overview"]["num_rows"] == 15
    assert profile["overview"]["num_columns"] == 6
    
    # 2. Diagnosis
    kpis = discover_kpis(sample_integration_df, profile)
    patterns = analyze_patterns(sample_integration_df, profile)
    anoms = detect_anomalies(sample_integration_df, profile)
    
    assert len(kpis) > 0
    assert "correlations" in patterns
    assert "total_anomalies" in anoms
    
    # 3. Target Detection & Training
    targets_info = detect_targets(sample_integration_df, profile)
    target_col = "late_risk" # force target to late_risk for prediction
    
    preprocessor = DataPreprocessor(target_col=target_col, target_type="binary_classification")
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(sample_integration_df)
    
    # Train
    best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, "binary_classification")
    assert best_model is not None
    
    # Evaluate
    metrics = evaluate_model(best_model, X_test, y_test, "binary_classification")
    assert "F1 Score" in metrics
    
    # 4. Explanation
    shap_vals, base_value = get_shap_values(best_model, X_test, X_test, "binary_classification", class_idx=1)
    global_drivers = explain_globally(best_model, X_test, X_test, feat_names, "binary_classification", class_idx=1)
    local_driver = explain_locally(best_model, X_test, X_test, 0, feat_names, "binary_classification", class_idx=1)
    
    assert len(global_drivers) > 0
    assert "prediction_value" in local_driver

def test_pipeline_integration_step_6_to_8(sample_integration_df):
    # 1. Preprocess & Fit Model
    preprocessor = DataPreprocessor(target_col="late_risk", target_type="binary_classification")
    X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(sample_integration_df)
    best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, "binary_classification")
    profile = profile_dataset(sample_integration_df)
    
    # 2. Digital Twin What-If Simulation
    baseline_row = preprocessor.X_test_raw.iloc[0]
    changes = [{"feature": "scheduled_days", "type": "absolute", "value": 5.0}]
    
    sim_res = run_twin_simulation(
        model=best_model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        changes=changes,
        target_type="binary_classification",
        profile=profile
    )
    
    assert "baseline_prediction" in sim_res
    assert "scenario_prediction" in sim_res
    
    # 3. Decision Recommendations
    controllable_vars = identify_controllable_variables(
        df=sample_integration_df,
        profile=profile,
        target_col="late_risk"
    )
    
    optimizer = GeneticOptimizer(
        model=best_model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        controllable_vars=controllable_vars,
        target_type="binary_classification",
        objective_mode="Minimize",
        class_idx=1,
        pop_size=10,
        generations=5
    )
    
    ga_res = optimizer.optimize()
    assert "recommended_values" in ga_res
    assert ga_res["optimized_prediction"] is not None
    
    # 4. RL env
    env = TwinOptimizationEnv(
        model=best_model,
        preprocessor=preprocessor,
        baseline_row=baseline_row,
        controllable_vars=controllable_vars,
        target_type="binary_classification",
        objective_mode="Minimize",
        class_idx=1
    )
    
    state, info = env.reset()
    assert state.shape[0] == len([c for c in controllable_vars if c["type"] == "numeric"])

@patch("requests.post")
def test_pipeline_integration_step_9_genai(mock_post, sample_integration_df):
    # Mock LLM API
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "SynTwin analysis report response."}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_res
    
    # 1. Assemble Session state structures
    mock_state = {
        "discovered_kpis": [{"name": "Sales Avg", "value": 150.0, "interpretation": "Overall mean"}],
        "model_late_risk": object(),
        "meta_late_risk": {
            "best_name": "Random Forest",
            "test_samples": 5,
            "feature_names": ["sales", "scheduled_days"],
            "X_test_proc": np.zeros((5, 2)),
            "y_test": np.zeros(5),
            "predictions": np.zeros(5)
        },
        "metrics_late_risk": {"Accuracy": 0.90}
    }
    
    profile = profile_dataset(sample_integration_df)
    
    # 2. Build Context
    context_dict = build_context(sample_integration_df, profile, st_state=mock_state)
    context_text = format_context_to_text(context_dict)
    
    assert "Dataset Profile" in context_text
    assert "Predictive Models" in context_text
    
    # 3. RAG Retrieval
    store = LocalVectorStore(api_key=None)
    store.add_document("policy_notes.txt", b"Customer late delivery thresholds are set at 3 days.")
    store.build_index()
    
    chunks = store.retrieve("late delivery thresholds", top_k=1)
    assert len(chunks) == 1
    
    # 4. GenAI Response
    client = LLMClient(api_key="fake-gemini-key", provider="gemini")
    res = generate_grounded_response(
        llm_client=client,
        user_query="What is the policy for late delivery?",
        analytical_context_str=context_text,
        retrieved_chunks=chunks
    )
    
    assert res["status"] == "success"
    assert res["response"] == "SynTwin analysis report response."
    assert "policy_notes.txt" in res["sources"]
