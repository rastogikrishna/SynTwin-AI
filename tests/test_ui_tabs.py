import pytest
import pandas as pd
import numpy as np
import pathlib
import sys
from unittest.mock import patch

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.profiler import profile_dataset
from streamlit.testing.v1 import AppTest

def test_ui_tabs_flow():
    # 1. Recreate sample df and profile
    df = pd.DataFrame({
        "order_id": list(range(1, 16)),
        "sales": [100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0, 150.0, 100.0],
        "shipping_cost": [10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0],
        "scheduled_days": [3, 4, 3, 4, 3, 3, 4, 4, 3, 4, 3, 3, 4, 3, 4],
        "real_days": [3, 5, 2, 4, 3, 4, 4, 5, 3, 4, 2, 3, 5, 3, 4],
        "late_risk": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    })
    
    profile = profile_dataset(df)
    
    # Initialize AppTest
    at = AppTest.from_file(str(project_root / "app" / "app.py"))
    
    # Inject dataset into session state so it's loaded
    at.session_state["dataset"] = df
    at.session_state["profile"] = profile
    at.session_state["dataset_name"] = "sample_retail.csv"
    at.session_state["active_nav"] = "Overview"
    
    # Run the app first time (Overview)
    at.run(timeout=30)
    assert not at.exception, f"Overview tab crashed: {at.exception}"
    
    # Verify that dataset remains available
    assert at.session_state["dataset"] is not None
    assert at.session_state["dataset_name"] == "sample_retail.csv"
    
    # Test Diagnosis
    at.button(key="nav_btn_diagnosis").click().run(timeout=30)
    assert not at.exception, f"Diagnosis tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test Prediction
    at.button(key="nav_btn_prediction").click().run(timeout=30)
    assert not at.exception, f"Prediction tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test Explainability
    at.button(key="nav_btn_explainability").click().run(timeout=30)
    assert not at.exception, f"Explainability tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test Forecast
    at.button(key="nav_btn_forecast").click().run(timeout=30)
    assert not at.exception, f"Forecast tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test Digital Twin
    at.button(key="nav_btn_digital_twin").click().run(timeout=30)
    assert not at.exception, f"Digital Twin tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test Decision
    at.button(key="nav_btn_decision").click().run(timeout=30)
    assert not at.exception, f"Decision tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None
    
    # Test AI Assistant
    at.button(key="nav_btn_ai_assistant").click().run(timeout=30)
    assert not at.exception, f"AI Assistant tab crashed: {at.exception}"
    assert at.session_state["dataset"] is not None

@patch("src.genai.llm_client.LLMClient.generate")
def test_ai_assistant_with_891_x_12_dataset(mock_generate):
    def side_effect(prompt, system_instruction=None):
        current_question = prompt.split("### User's Current Question:")[-1].lower()
        if "concerned" in current_question:
            return "No critical concerns"
        elif "important factors" in current_question:
            return "No predictive model is trained yet"
        elif "summarize" in current_question:
            return "Grounded response: The dataset has 891 rows and 12 columns."
        else:
            return "No critical concerns"
    mock_generate.side_effect = side_effect

    # 1. Create synthetic dataset of 891 rows x 12 variables
    np.random.seed(42)
    rows = 891
    df = pd.DataFrame({
        "order_id": list(range(1, rows + 1)),
        "sales": np.random.uniform(10, 1000, size=rows),
        "shipping_cost": np.random.uniform(5, 100, size=rows),
        "scheduled_days": np.random.randint(1, 7, size=rows),
        "real_days": np.random.randint(1, 9, size=rows),
        "late_risk": np.random.choice([0, 1], size=rows),
        "customer_segment": np.random.choice(["Consumer", "Corporate", "Home Office"], size=rows),
        "region": np.random.choice(["East", "West", "Central", "South"], size=rows),
        "product_category": np.random.choice(["Technology", "Furniture", "Office Supplies"], size=rows),
        "discount": np.random.uniform(0, 0.3, size=rows),
        "order_priority": np.random.choice(["Low", "Medium", "High", "Critical"], size=rows),
        "ship_mode": np.random.choice(["Standard Class", "Second Class", "First Class", "Same Day"], size=rows),
    })
    
    profile = profile_dataset(df)
    
    # Initialize AppTest
    at = AppTest.from_file(str(project_root / "app" / "app.py"))
    at.session_state["dataset"] = df
    at.session_state["profile"] = profile
    at.session_state["dataset_name"] = "891_x_12_dataset.csv"
    at.session_state["active_nav"] = "AI Assistant"
    at.session_state["chat_history"] = []
    
    # Run the app
    at.run(timeout=30)
    assert not at.exception, f"AI Assistant page failed to run: {at.exception}"
    
    # Question 1: "Summarize the current situation"
    # Find chat_input widget and submit query
    at.chat_input[0].set_value("Summarize the current situation").run(timeout=30)
    assert not at.exception, f"Summarize situation failed: {at.exception}"
    last_msg = at.session_state["chat_history"][-1]
    assert last_msg["role"] == "assistant"
    assert "891" in last_msg["content"]
    assert "12" in last_msg["content"]
    
    # Question 2: "What are the most important factors?"
    at.chat_input[0].set_value("What are the most important factors?").run(timeout=30)
    assert not at.exception, f"Explain drivers failed: {at.exception}"
    last_msg = at.session_state["chat_history"][-1]
    assert last_msg["role"] == "assistant"
    assert "No predictive model is trained yet" in last_msg["content"]
    
    # Question 3: "What should I be concerned about?"
    at.chat_input[0].set_value("What should I be concerned about?").run(timeout=30)
    assert not at.exception, f"Risks failed: {at.exception}"
    last_msg = at.session_state["chat_history"][-1]
    assert last_msg["role"] == "assistant"
    assert any(x in last_msg["content"].lower() for x in ["concern", "warning", "risk", "no profile-level"]), f"Actual content: {last_msg['content']}"
