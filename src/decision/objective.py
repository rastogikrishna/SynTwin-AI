from typing import Optional, Any

def calculate_fitness(pred_val: Any, target_type: str, objective_mode: str, 
                      prob_val: Optional[float] = None) -> float:
    """
    Translates predictive model outputs to a numeric fitness score.
    
    Parameters:
    -----------
    pred_val : Any
        Predicted target class or continuous regression output.
    target_type : str
        Type of predictive target ("binary_classification", "multiclass_classification", "regression").
    objective_mode : str
        Objective direction ("Maximize" or "Minimize").
    prob_val : Optional[float]
        Probability of the class of interest (for classification tasks).
        
    Returns:
    --------
    float
        Numeric score to maximize/minimize.
    """
    # 1. Classification Metric Ingestion
    if target_type in ["binary_classification", "multiclass_classification"]:
        score = float(prob_val) if prob_val is not None else (1.0 if pred_val == 1 else 0.0)
    else:
        # Regression
        score = float(pred_val)
        
    # 2. Objective Direction Mapping
    if objective_mode == "Minimize":
        return -score
    else:
        return score
