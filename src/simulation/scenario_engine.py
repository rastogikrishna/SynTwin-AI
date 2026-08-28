import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def apply_scenario_changes(baseline_row: pd.Series, changes: List[Dict[str, Any]]) -> pd.Series:
    """
    Applies scenario modifications to a copy of the baseline row Series.
    
    Parameters:
    -----------
    baseline_row : pd.Series
        Original observation variables.
    changes : List[Dict]
        List of change specifications, e.g.:
        [{"feature": "col", "type": "percentage", "value": 10.0}]
        
    Returns:
    --------
    pd.Series
        Modified row Series.
    """
    modified_row = baseline_row.copy().astype(object)
    
    for change in changes:
        col = change["feature"]
        change_type = change["type"]
        val = change["value"]
        
        if col not in modified_row.index:
            continue
            
        if change_type == "percentage":
            orig = float(modified_row[col])
            modified_row[col] = orig * (1.0 + float(val) / 100.0)
        elif change_type == "absolute":
            modified_row[col] = float(val)
        elif change_type == "category":
            modified_row[col] = str(val)
        elif change_type == "boolean":
            # Handle float representation or string toggling
            if isinstance(val, str):
                modified_row[col] = val.lower() == 'true'
            else:
                modified_row[col] = bool(val)
            
    return modified_row

from src.utils.feature_selection import get_controllable_features

def rank_simulation_variables(df: pd.DataFrame, profile: Dict[str, Any], 
                              global_importance: Optional[List[Dict[str, Any]]] = None,
                              target_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Identifies features suitable for simulation in the Digital Twin.
    Ranks them based on SHAP importance, filtering out the target column, IDs, and dates.
    """
    return get_controllable_features(df, profile, target_col, global_importance)
