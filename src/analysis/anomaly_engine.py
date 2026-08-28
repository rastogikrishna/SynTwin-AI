import pandas as pd
import numpy as np
from typing import Dict, Any, List

def detect_anomalies(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identifies statistical outliers (potential anomalies) in numerical columns.
    
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
        - columns_with_anomalies: int
        - total_anomalies: int
        - anomaly_percentage: float
        - column_details: List[Dict] with keys:
          - column: str
          - outlier_count: int
          - outlier_percentage: float
          - method: str
          - min_outlier: float or None
          - max_outlier: float or None
    """
    results = {
        "columns_with_anomalies": 0,
        "total_anomalies": 0,
        "anomaly_percentage": 0.0,
        "column_details": []
    }
    
    total_rows = len(df)
    if total_rows == 0:
        return results

    numeric_cols = profile.get("data_types", {}).get("numerical", [])
    id_cols = profile.get("inferred_columns", {}).get("ids", [])
    
    all_anomalous_indices = set()
    
    for col in numeric_cols:
        # Ignore ID variables which may be represented as integers
        if col in id_cols:
            continue
            
        series = df[col].dropna()
        if len(series) < 5:
            continue
            
        # Tukey's fences calculation
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        
        if iqr > 0:
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            
            outliers_mask = (series < lower_fence) | (series > upper_fence)
            outlier_indices = series[outliers_mask].index
            outlier_vals = series[outliers_mask]
            
            outlier_cnt = len(outlier_vals)
            if outlier_cnt > 0:
                all_anomalous_indices.update(outlier_indices)
                
                results["column_details"].append({
                    "column": col,
                    "outlier_count": outlier_cnt,
                    "outlier_percentage": float((outlier_cnt / total_rows) * 100),
                    "method": "IQR (Tukey's Fences)",
                    "min_outlier": float(outlier_vals.min()) if pd.notna(outlier_vals.min()) else None,
                    "max_outlier": float(outlier_vals.max()) if pd.notna(outlier_vals.max()) else None
                })
                
    results["columns_with_anomalies"] = len(results["column_details"])
    results["total_anomalies"] = len(all_anomalous_indices)
    results["anomaly_percentage"] = float((len(all_anomalous_indices) / total_rows) * 100) if total_rows > 0 else 0.0
    results["column_details"].sort(key=lambda x: x["outlier_count"], reverse=True)
    
    return results
