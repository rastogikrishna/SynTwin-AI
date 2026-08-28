import pytest
import pandas as pd
import numpy as np
import tempfile
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_data
from src.data.profiler import profile_dataset

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield pathlib.Path(tmpdirname)

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "id_col": [1, 2, 3, 4, 5],
        "num_col": [10.5, 20.0, np.nan, 40.2, 50.0],
        "cat_col": ["low", "high", "low", "medium", "high"],
        "bool_col": [True, False, True, False, True],
        "date_col": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "const_col": ["constant"] * 5
    })

def test_csv_loading(temp_dir, mock_df):
    csv_path = temp_dir / "test.csv"
    mock_df.to_csv(csv_path, index=False)
    
    df = load_data(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 6)
    assert list(df.columns) == list(mock_df.columns)

def test_excel_loading(temp_dir, mock_df):
    xlsx_path = temp_dir / "test.xlsx"
    mock_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    
    df = load_data(xlsx_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 6)

def test_invalid_file_handling(temp_dir):
    # Test unsupported extension
    txt_path = temp_dir / "test.txt"
    txt_path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(txt_path)
        
    # Test empty dataset
    empty_csv = temp_dir / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_data(empty_csv)

    # Test non-existent path
    with pytest.raises(FileNotFoundError):
        load_data(temp_dir / "non_existent.csv")

def test_basic_profiling(mock_df):
    profile = profile_dataset(mock_df)
    assert profile["overview"]["num_rows"] == 5
    assert profile["overview"]["num_columns"] == 6
    assert profile["overview"]["duplicate_rows"] == 0

def test_missing_value_detection(mock_df):
    profile = profile_dataset(mock_df)
    assert profile["column_profiles"]["num_col"]["missing_count"] == 1
    assert profile["column_profiles"]["num_col"]["missing_percentage"] == 20.0
    assert profile["column_profiles"]["id_col"]["missing_count"] == 0

def test_duplicate_detection(mock_df):
    dup_df = pd.concat([mock_df, mock_df.iloc[[0]]], ignore_index=True)
    profile = profile_dataset(dup_df)
    assert profile["overview"]["duplicate_rows"] == 1
    assert any(w["type"] == "duplicates" for w in profile["warnings"])

def test_column_type_detection(mock_df):
    profile = profile_dataset(mock_df)
    
    assert "num_col" in profile["data_types"]["numerical"]
    assert "cat_col" in profile["data_types"]["categorical"]
    assert "bool_col" in profile["data_types"]["boolean"]
    assert "date_col" in profile["data_types"]["datetime"]
    
    # Inferred rules
    assert "id_col" in profile["inferred_columns"]["ids"]
    assert "date_col" in profile["inferred_columns"]["dates"]
    assert "num_col" in profile["inferred_columns"]["targets"]
    assert any(w["column"] == "const_col" and w["type"] == "constant" for w in profile["warnings"])
