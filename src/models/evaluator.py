from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
from typing import Dict, Any

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, target_type: str) -> Dict[str, float]:
    """
    Computes validation metrics for a fitted model.
    
    Parameters:
    -----------
    model : Any
        Fitted scikit-learn model.
    X_test : np.ndarray
        Preprocessed test features.
    y_test : np.ndarray
        Test target labels.
    target_type : str
        Target type ("binary_classification", "multiclass_classification", "regression").
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics.
    """
    metrics = {}
    preds = model.predict(X_test)
    
    if target_type in ["binary_classification", "multiclass_classification"]:
        metrics["Accuracy"] = float(accuracy_score(y_test, preds))
        metrics["Precision"] = float(precision_score(y_test, preds, average='macro', zero_division=0))
        metrics["Recall"] = float(recall_score(y_test, preds, average='macro', zero_division=0))
        metrics["F1 Score"] = float(f1_score(y_test, preds, average='macro', zero_division=0))
        
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1:
                    if target_type == "binary_classification" and len(unique_classes) == 2:
                        metrics["ROC-AUC"] = float(roc_auc_score(y_test, probs[:, 1]))
                    elif target_type == "multiclass_classification":
                        metrics["ROC-AUC"] = float(roc_auc_score(y_test, probs, multi_class='ovr', average='macro'))
            except Exception:
                pass
                
    elif target_type == "regression":
        metrics["MAE"] = float(mean_absolute_error(y_test, preds))
        mse = mean_squared_error(y_test, preds)
        metrics["RMSE"] = float(np.sqrt(mse))
        metrics["R2 Score"] = float(r2_score(y_test, preds))
        
    else:
        raise ValueError(f"Unknown target type: {target_type}")
        
    return metrics
