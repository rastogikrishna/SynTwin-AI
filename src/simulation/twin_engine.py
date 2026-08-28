import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.simulation.scenario_engine import apply_scenario_changes

def run_twin_simulation(model: Any, preprocessor: Any, baseline_row: pd.Series, 
                        changes: List[Dict[str, Any]], target_type: str, 
                        profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a what-if scenario by processing the modified inputs through the prediction pipeline.
    
    Parameters:
    -----------
    model : Any
        Fitted model.
    preprocessor : Any
        Fitted preprocessor pipeline.
    baseline_row : pd.Series
        Baseline values.
    changes : List[Dict]
        List of changes.
    target_type : str
        Target type ("binary_classification", "multiclass_classification", "regression").
    profile : Dict[str, Any]
        Profile dictionary.
        
    Returns:
    --------
    Dict[str, Any]
        Simulation comparison output.
    """
    # 1. Modify values
    modified_row = apply_scenario_changes(baseline_row, changes)
    
    # 2. Check historical boundary ranges
    warnings = []
    out_of_range = False
    
    for change in changes:
        col = change["feature"]
        change_val = modified_row[col]
        
        info = profile.get("column_profiles", {}).get(col, {})
        if info and info["type_group"] == "numeric":
            summary = info.get("numerical_summary", {})
            min_val = summary.get("min", None)
            max_val = summary.get("max", None)
            
            if min_val is not None and max_val is not None:
                if change_val < min_val or change_val > max_val:
                    out_of_range = True
                    warnings.append(
                        f"Variable '{col.replace('_',' ').title()}' value ({change_val:.2f}) lies outside historical bounds ({min_val:.2f} to {max_val:.2f})."
                    )
                    
    # 3. Map to DataFrames and preprocess
    base_df = pd.DataFrame([baseline_row])
    mod_df = pd.DataFrame([modified_row])
    
    X_base_proc = preprocessor.transform_row(base_df)
    X_mod_proc = preprocessor.transform_row(mod_df)
    
    # 4. Predict
    pred_base = model.predict(X_base_proc)[0]
    pred_mod = model.predict(X_mod_proc)[0]
    
    results = {
        "baseline_prediction": pred_base,
        "scenario_prediction": pred_mod,
        "out_of_range": out_of_range,
        "warnings": warnings,
        "modified_row": modified_row
    }
    
    # Extract results type mapping details
    if target_type == "regression":
        pred_base_val = float(pred_base)
        pred_mod_val = float(pred_mod)
        abs_diff = pred_mod_val - pred_base_val
        pct_diff = (abs_diff / pred_base_val) * 100.0 if pred_base_val != 0.0 else 0.0
        
        results["baseline_prediction"] = pred_base_val
        results["scenario_prediction"] = pred_mod_val
        results["abs_difference"] = abs_diff
        results["pct_difference"] = pct_diff
        
    else: # Classification
        if hasattr(model, "predict_proba"):
            probs_base = model.predict_proba(X_base_proc)[0]
            probs_mod = model.predict_proba(X_mod_proc)[0]
            
            results["baseline_probabilities"] = probs_base.tolist()
            results["scenario_probabilities"] = probs_mod.tolist()
            
            # Infer classes
            classes = getattr(model, "classes_", [0, 1])
            results["class_labels"] = [str(c) for c in classes]
            
            # Map predictions probabilities
            pred_base_idx = int(np.argmax(probs_base))
            results["baseline_prob_val"] = float(probs_base[pred_base_idx])
            results["scenario_prob_val"] = float(probs_mod[pred_base_idx])
            
    return results
