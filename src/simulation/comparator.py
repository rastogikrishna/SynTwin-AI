import pandas as pd
from typing import List, Dict, Any

def compare_scenarios(scenarios: List[Dict[str, Any]], target_type: str) -> pd.DataFrame:
    """
    Formulates a tabular summary comparing multiple saved scenarios against the baseline.
    """
    records = []
    for sc in scenarios:
        rec = {
            "Scenario Name": sc["name"],
            "Prediction": sc["scenario_prediction"]
        }
        
        if target_type == "regression":
            # Float prediction formatting
            rec["Absolute Change"] = sc.get("abs_difference", 0.0)
            rec["Percentage Change"] = f"{sc.get('pct_difference', 0.0):+.2f}%"
        else:
            # Classification probability formats
            baseline_prob = sc.get("baseline_prob_val", None)
            scenario_prob = sc.get("scenario_prob_val", None)
            
            rec["Baseline Probability"] = f"{baseline_prob*100:.1f}%" if baseline_prob is not None else "N/A"
            rec["Scenario Probability"] = f"{scenario_prob*100:.1f}%" if scenario_prob is not None else "N/A"
            
            is_changed = "Outcome Shifted" if sc["scenario_prediction"] != sc["baseline_prediction"] else "Stable Outcome"
            rec["Status"] = is_changed
            
        records.append(rec)
        
    return pd.DataFrame(records)
