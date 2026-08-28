import pytest
import pandas as pd
import numpy as np
import pathlib
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.genai.context_builder import build_context, format_context_to_text
from src.genai.llm_client import LLMClient
from src.genai.rag_engine import LocalVectorStore
from src.genai.response_generator import generate_grounded_response

@pytest.fixture
def mock_session_state():
    return {
        "discovered_kpis": [
            {"name": "Total Sales", "value": 150000.0, "interpretation": "Overall revenue"}
        ],
        "model_Late_delivery_risk": object(),
        "meta_Late_delivery_risk": {
            "best_name": "Random Forest",
            "test_samples": 200,
            "feature_names": ["Days", "Shipping Mode"]
        },
        "metrics_Late_delivery_risk": {
            "Accuracy": 0.85,
            "F1 Score": 0.84
        }
    }

def test_context_builder(mock_session_state):
    profile = {
        "overview": {
            "num_rows": 1000,
            "num_columns": 10,
            "memory_formatted": "500 KB"
        },
        "column_profiles": {
            "Days": {"type_group": "numeric"},
            "Shipping Mode": {"type_group": "categorical"}
        },
        "overall_status": "Good",
        "warnings": []
    }
    
    ctx = build_context(profile=profile, st_state=mock_session_state)
    
    assert "dataset" in ctx
    assert ctx["dataset"]["num_rows"] == 1000
    assert "prediction" in ctx
    assert ctx["prediction"]["best_model"] == "Random Forest"
    
    formatted = format_context_to_text(ctx)
    assert "Dataset Profile" in formatted
    assert "Predictive Models" in formatted

def test_rag_engine_extraction():
    store = LocalVectorStore(api_key=None)
    
    # Test text loading
    txt_content = b"SynTwin AI operating procedures. High priority suppliers are evaluated weekly."
    store.add_document("terms.txt", txt_content)
    
    assert store.get_chunks_count() > 0
    
    # Build index offline (uses TF-IDF fallback since api_key is None)
    store.build_index()
    
    # Retrieve
    res = store.retrieve("high priority weekly", top_k=1)
    assert len(res) == 1
    assert "terms.txt" in res[0]["source"]
    assert "weekly" in res[0]["text"]

def test_llm_client_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.genai.llm_client.load_dotenv"):
            client = LLMClient(api_key=None, provider="gemini")
            client.api_key = None
            with pytest.raises(ValueError, match="LLM features require an API key"):
                client.generate("hello")

@patch("requests.post")
def test_llm_client_gemini_success(mock_post):
    # Mock Gemini response
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Mocked Gemini Response Content"}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_res
    
    client = LLMClient(api_key="fake-gemini-key", provider="gemini")
    resp = client.generate("Summarize dataset")
    
    assert resp == "Mocked Gemini Response Content"
    mock_post.assert_called_once()

def test_response_generator_insufficient_context():
    client = LLMClient(api_key="fake-key", provider="gemini")
    
    # Mock generation call inside response generator
    with patch.object(client, "generate", return_value="I don't have enough information in the current SynTwin analysis to answer that reliably."):
        res = generate_grounded_response(
            llm_client=client,
            user_query="What is the stock price of Google?",
            analytical_context_str="## SynTwin Analytical Context\nNo matching data.",
            retrieved_chunks=[]
        )
        
        assert "I don't have enough information" in res["response"]


def test_conversational_routing_and_queries():
    from src.genai.response_generator import is_conversational, generate_grounded_response
    
    # 1. Test is_conversational classification
    assert is_conversational("hello")
    assert is_conversational("hi")
    assert is_conversational("hey")
    assert is_conversational("who are you?")
    assert is_conversational("what's your name?")
    assert is_conversational("My name is Krishna")
    assert is_conversational("What is my name?")
    assert not is_conversational("Summarize the current situation")
    assert not is_conversational("What are the most important factors?")
    assert not is_conversational("What should I be concerned about?")
    
    # 2. Test response generation with mocked client
    client = LLMClient(api_key="fake-key", provider="gemini")
    
    with patch.object(client, "generate") as mock_gen:
        mock_gen.return_value = "Hello! I'm SynTwin AI."
        res = generate_grounded_response(
            llm_client=client,
            user_query="hello",
            analytical_context_str="Analytical Context",
            retrieved_chunks=[]
        )
        assert res["status"] == "success"
        assert res["response"] == "Hello! I'm SynTwin AI."
        mock_gen.assert_called_once()
        args, kwargs = mock_gen.call_args
        assert "User's Current Question:" in args[0]
        assert "friendly, natural conversational response" in kwargs["system_instruction"]


def test_app_local_conversational_fallback():
    # Make sure app folder is on path to import helpers
    app_path = project_root / "app"
    if str(app_path) not in sys.path:
        sys.path.append(str(app_path))
        
    from app import is_conversational as app_is_conversational
    from app import local_conversational_response
    
    assert app_is_conversational("hello")
    assert app_is_conversational("my name is Krishna")
    assert not app_is_conversational("What are the main prediction drivers?")
    
    # Mock streamlit session state for "What is my name?" history check
    with patch("streamlit.session_state", {"chat_history": [{"role": "user", "content": "My name is Krishna"}]}):
        ans_name = local_conversational_response("What is my name?")
        assert "Krishna" in ans_name
        
    ans_greeting = local_conversational_response("hello")
    assert "SynTwin AI" in ans_greeting
    
    ans_identity = local_conversational_response("who are you?")
    assert "decision-intelligence" in ans_identity

