import pandas as pd
import numpy as np
from typing import Dict, Any, List

def detect_targets(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ranks columns as potential prediction targets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame.
    profile : Dict[str, Any]
        The profile dictionary returned by profile_dataset(df).
        
    Returns:
    --------
    Dict[str, Any]
        A dictionary containing:
        - best_target: str or None
        - target_type: str or None ("binary_classification", "multiclass_classification", "regression")
        - confidence: float
        - reason: str
        - alternatives: List[Dict] with alternative targets
    """
    candidates = []
    total_rows = len(df)
    if total_rows == 0:
        return {
            "best_target": None,
            "target_type": None,
            "confidence": 0.0,
            "reason": "No suitable supervised prediction target was automatically identified because the dataset is empty.",
            "alternatives": []
        }

    id_cols = profile.get("inferred_columns", {}).get("ids", [])
    date_cols = profile.get("inferred_columns", {}).get("dates", []) + profile.get("data_types", {}).get("datetime", [])
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Rule: Exclude identifiers by name pattern
        is_id_name = "id" in col_lower or "key" in col_lower or "code" in col_lower or "number" in col_lower or "num" in col_lower
        if col in id_cols or is_id_name or col in date_cols:
            continue
            
        info = profile["column_profiles"].get(col, {})
        if not info:
            continue
            
        missing_pct = info["missing_percentage"]
        unique_count = info["unique_count"]
        
        # Rule: Exclude columns where every row has a unique value (excluding floats)
        is_float = "float" in info.get("dtype", "")
        if unique_count == total_rows and total_rows > 2 and not is_float:
            continue
            
        # Skip if missing rate is too high
        if missing_pct > 25.0:
            continue
            
        # Skip constant columns
        if unique_count <= 1:
            continue
            
        # Skip coordinates and media/link properties
        is_coord = "latitude" in col_lower or "longitude" in col_lower or "lat" in col_lower or "lon" in col_lower
        is_link = "url" in col_lower or "http" in col_lower or "image" in col_lower or "png" in col_lower or "jpg" in col_lower
        if is_coord or is_link:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        # Keywords for target identification
        target_keywords = [
            "sales", "profit", "benefit", "amount", "price", "cost", "loss", "quantity", 
            "delay", "days", "risk", "target", "label", "class", "status", "delivery", 
            "fraud", "churn", "response", "y"
        ]
        keyword_boost = 0.15 if any(kw in col_lower for kw in target_keywords) else 0.0

        # 1. Categorical / Boolean Targets
        if (info["type_group"] in ["categorical", "boolean"] or (info["type_group"] == "numeric" and unique_count <= 15)) and not is_float:
            if 2 <= unique_count <= 15:
                val_counts = series.value_counts(normalize=True)
                top_pct = val_counts.iloc[0]
                
                # Exclude if heavily skewed
                if top_pct > 0.95:
                    continue
                    
                target_type = "binary_classification" if unique_count == 2 else "multiclass_classification"
                
                # Base score: binary is prioritized slightly
                base_score = 0.90 if target_type == "binary_classification" else 0.85
                score = base_score - (missing_pct / 100.0) + keyword_boost
                
                if top_pct > 0.8:
                    score -= 0.10
                    
                confidence = 0.90 if target_type == "binary_classification" else 0.80
                reason = f"Mappable categorical target with {unique_count} distinct classes and low missingness."
                
                candidates.append({
                    "column": col,
                    "target_type": target_type,
                    "score": score,
                    "confidence": confidence,
                    "reason": reason
                })
                
        # 2. Numerical Targets
        elif info["type_group"] == "numeric":
            # Check for sequential index
            sample_sorted = series.head(100).sort_values()
            if len(sample_sorted) > 3:
                diffs = sample_sorted.diff().dropna()
                if diffs.std() == 0:
                    continue
                    
            std = series.std()
            if pd.isna(std) or std == 0:
                continue
                
            base_score = 0.80
            score = base_score - (missing_pct / 100.0) + keyword_boost
            
            confidence = 0.85
            reason = f"Continuous numeric variable with variation (std: {std:.2f}) suitable for regression."
            
            candidates.append({
                "column": col,
                "target_type": "regression",
                "score": score,
                "confidence": confidence,
                "reason": reason
            })
            
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    if not candidates:
        return {
            "best_target": None,
            "target_type": None,
            "confidence": 0.0,
            "reason": "No suitable supervised prediction target was automatically identified.",
            "alternatives": []
        }
        
    best = candidates[0]
    alternatives = [
        {
            "column": c["column"],
            "target_type": c["target_type"],
            "confidence": c["confidence"],
            "reason": c["reason"]
        } for c in candidates[1:6]
    ]
    
    return {
        "best_target": best["column"],
        "target_type": best["target_type"],
        "confidence": best["confidence"],
        "reason": best["reason"],
        "alternatives": alternatives
    }
