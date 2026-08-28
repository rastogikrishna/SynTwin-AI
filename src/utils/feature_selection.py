import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def get_controllable_features(df: pd.DataFrame, profile: Dict[str, Any], 
                              target_col: Optional[str] = None, 
                              global_importance: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Identifies features suitable for control optimization and scenario simulation.
    Ranks them based on SHAP/model importance, filtering out dates, IDs, targets, coordinates, and constants.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Loaded dataset.
    profile : Dict[str, Any]
        Dataset profile statistics.
    target_col : Optional[str]
        Currently selected prediction target (if any).
    global_importance : Optional[List[Dict]]
        SHAP global importances.
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of candidate controllable variables with statistical boundaries.
    """
    importance_map = {}
    if global_importance:
        for item in global_importance:
            importance_map[item["feature"]] = item["importance"]
            
    id_cols = profile.get("inferred_columns", {}).get("ids", [])
    date_cols = profile.get("inferred_columns", {}).get("dates", []) + profile.get("data_types", {}).get("datetime", [])
    
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        
        # Exclude IDs, dates, target, and location coordinates
        if col in id_cols or col in date_cols or (target_col and col == target_col):
            continue
        if "id" in col_lower or "key" in col_lower or "code" in col_lower:
            continue
        if "latitude" in col_lower or "longitude" in col_lower or "lat" in col_lower or "lon" in col_lower:
            continue
            
        info = profile["column_profiles"].get(col, {})
        if not info:
            continue
            
        # Exclude constants
        if info["unique_count"] <= 1:
            continue
            
        feat_type = info["type_group"]
        
        # Determine average model importance
        importance_val = 0.0
        if col in importance_map:
            importance_val = importance_map[col]
        else:
            matching_vals = [val for name, val in importance_map.items() if name.startswith(f"{col}_")]
            if matching_vals:
                importance_val = float(np.mean(matching_vals))
                
        candidate = {
            "feature": col,
            "type": feat_type,
            "importance": importance_val,
            "min": None,
            "max": None,
            "categories": []
        }
        
        if feat_type == "numeric":
            summary = info.get("numerical_summary", {})
            candidate["min"] = float(summary.get("min", df[col].min()))
            candidate["max"] = float(summary.get("max", df[col].max()))
        elif feat_type == "categorical":
            cat_summary = info.get("categorical_summary", {})
            candidate["categories"] = list(cat_summary.get("top_frequencies", {}).keys())
            if not candidate["categories"]:
                candidate["categories"] = list(df[col].dropna().unique())
        elif feat_type == "boolean":
            candidate["categories"] = [True, False]
            
        candidates.append(candidate)
        
    candidates.sort(key=lambda x: x["importance"], reverse=True)
    return candidates
