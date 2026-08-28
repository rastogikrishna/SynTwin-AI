import pandas as pd
import numpy as np
from typing import Dict, Any, List

def detect_forecasting_eligibility(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scans the dataset to determine if it is suitable for time-series forecasting.
    Ranks combinations of date columns and numeric columns.
    
    Returns:
    --------
    Dict[str, Any]
        - eligible: bool
        - date_columns: List[str]
        - metric_columns: List[str]
        - recommended_date: str or None
        - recommended_metric: str or None
        - reason: str
        - frequency: str or None
    """
    total_rows = len(df)
    if total_rows < 15:
        return {
            "eligible": False,
            "date_columns": [],
            "metric_columns": [],
            "recommended_date": None,
            "recommended_metric": None,
            "reason": "Insufficient temporal observations (dataset has less than 15 rows) for reliable forecasting.",
            "frequency": None
        }
        
    # 1. Parse Date candidates
    date_candidates = profile.get("inferred_columns", {}).get("dates", []) + profile.get("data_types", {}).get("datetime", [])
    date_candidates = list(set(date_candidates))
    
    # 2. Parse Numeric candidates
    all_numeric = profile.get("data_types", {}).get("numerical", [])
    id_cols = profile.get("inferred_columns", {}).get("ids", [])
    
    metric_candidates = []
    for col in all_numeric:
        col_lower = col.lower()
        if col in id_cols or "id" in col_lower or "key" in col_lower or "code" in col_lower:
            continue
        if "latitude" in col_lower or "longitude" in col_lower or "lat" in col_lower or "lon" in col_lower:
            continue
            
        info = profile["column_profiles"].get(col, {})
        if info:
            if info["unique_count"] <= 1 or info["missing_percentage"] > 20.0:
                continue
            std = info.get("numerical_summary", {}).get("std", 0.0)
            if pd.isna(std) or std == 0:
                continue
            metric_candidates.append(col)
            
    if not date_candidates:
        return {
            "eligible": False,
            "date_columns": [],
            "metric_columns": metric_candidates,
            "recommended_date": None,
            "recommended_metric": None,
            "reason": "No suitable datetime column was detected in the dataset.",
            "frequency": None
        }
        
    if not metric_candidates:
        return {
            "eligible": False,
            "date_columns": date_candidates,
            "metric_columns": [],
            "recommended_date": None,
            "recommended_metric": None,
            "reason": "No suitable numerical metrics were found to forecast over time.",
            "frequency": None
        }
        
    # 3. Score combinations to find best pair
    best_date = None
    best_metric = None
    best_score = -1.0
    
    date_keywords = ["date", "time", "day", "month", "year", "period", "timestamp", "order", "shipping"]
    metric_keywords = ["sales", "profit", "revenue", "demand", "amount", "cost", "price", "quantity", "total", "count"]
    
    for date_col in date_candidates:
        date_score = 0.5
        date_col_lower = date_col.lower()
        if any(kw in date_col_lower for kw in date_keywords):
            date_score += 0.4
            
        date_info = profile["column_profiles"].get(date_col, {})
        if date_info:
            date_score -= (date_info["missing_percentage"] / 100.0)
            
        for metric_col in metric_candidates:
            metric_score = 0.5
            metric_col_lower = metric_col.lower()
            if any(kw in metric_col_lower for kw in metric_keywords):
                metric_score += 0.4
                
            metric_info = profile["column_profiles"].get(metric_col, {})
            if metric_info:
                metric_score -= (metric_info["missing_percentage"] / 100.0)
                
            total_score = date_score + metric_score
            if total_score > best_score:
                best_score = total_score
                best_date = date_col
                best_metric = metric_col
                
    # 4. Final verification of distinct periods
    try:
        unique_dates_for_check = df[best_date].dropna().unique()
        parsed_dates = pd.to_datetime(unique_dates_for_check, errors='coerce', format='mixed')
        distinct_dates = pd.Series(parsed_dates).dropna().nunique()
        if distinct_dates < 10:
            return {
                "eligible": False,
                "date_columns": date_candidates,
                "metric_columns": metric_candidates,
                "recommended_date": best_date,
                "recommended_metric": best_metric,
                "reason": f"Insufficient temporal coverage (only {distinct_dates} distinct date periods) for forecasting.",
                "frequency": None
            }
    except Exception:
        pass
        
    # Infer frequency
    inferred_freq = None
    try:
        unique_dates = df[best_date].dropna().unique()
        date_series = pd.Series(pd.to_datetime(unique_dates, errors='coerce', format='mixed')).sort_values()
        inferred_freq = pd.infer_freq(date_series.head(50))
    except Exception:
        pass
        
    return {
        "eligible": True,
        "date_columns": date_candidates,
        "metric_columns": metric_candidates,
        "recommended_date": best_date,
        "recommended_metric": best_metric,
        "reason": "Suitable datetime and numerical metrics exist for automatic time-series forecasting.",
        "frequency": inferred_freq
    }
