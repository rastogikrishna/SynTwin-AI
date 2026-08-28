import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils.feature_selection import get_controllable_features

def identify_controllable_variables(df: pd.DataFrame, profile: Dict[str, Any], 
                                     target_col: str, 
                                     global_importance: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Identifies features suitable for control optimization and RL simulation.
    Ranks them based on SHAP/model importance, filtering out dates, IDs, targets, and constants.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Loaded dataset.
    profile : Dict[str, Any]
        Dataset profile statistics.
    target_col : str
        Currently selected prediction target.
    global_importance : Optional[List[Dict]]
        SHAP global importances.
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of candidate controllable variables with statistical boundaries.
    """
    return get_controllable_features(df, profile, target_col, global_importance)
