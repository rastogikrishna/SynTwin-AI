import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict

def score_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression/forecasting validation metrics (MAE, RMSE, MAPE).
    Protects against division by zero in MAPE calculations.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Calculate MAPE safely by masking zeros
    mask = np.abs(y_true) > 1e-5
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = 0.0
        
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
