import pandas as pd
from typing import Dict, Any, List

def discover_kpis(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Auto-discovers and calculates suitable business KPIs from a DataFrame
    using the dataset profile dictionary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame.
    profile : Dict[str, Any]
        The profile dictionary returned by profile_dataset(df).
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of KPI dicts with:
        - name: str
        - value: float or int
        - type: str ("Volume", "Average", "Rate/Percentage", "Count")
        - source_column: str
        - interpretation: str
    """
    kpis = []
    total_rows = len(df)
    if total_rows == 0:
        return kpis
        
    # 1. Base Count KPIs (Total Transactions/Records)
    kpis.append({
        "name": "Total Records",
        "value": total_rows,
        "type": "Count",
        "source_column": "Dataset Level",
        "interpretation": "Total volume of observations/records in the dataset."
    })
    
    # Unique counts for potential ID columns
    id_cols = profile.get("inferred_columns", {}).get("ids", [])
    for col in id_cols:
        unique_cnt = int(df[col].nunique())
        clean_name = col.replace("_", " ").title()
        if "Id" in clean_name:
            clean_name = clean_name.replace("Id", "Count").strip()
        else:
            clean_name = f"Unique {clean_name}"
            
        kpis.append({
            "name": clean_name,
            "value": unique_cnt,
            "type": "Count",
            "source_column": col,
            "interpretation": f"Total count of unique {col} entities observed."
        })

    # 2. Extract targets/metrics from profile
    targets = profile.get("inferred_columns", {}).get("targets", [])
    numerical_cols = profile.get("data_types", {}).get("numerical", [])
    
    # Filter numerical columns that are targets and not IDs
    valid_numeric_kpis = [col for col in numerical_cols if col in targets and col not in id_cols]
    
    for col in valid_numeric_kpis:
        series = df[col].dropna()
        if series.empty:
            continue
            
        col_lower = col.lower()
        if "latitude" in col_lower or "longitude" in col_lower or "lat" in col_lower or "lon" in col_lower:
            continue
            
        col_min = float(series.min())
        col_max = float(series.max())
        col_mean = float(series.mean())
        col_sum = float(series.sum())
        
        # Clean title for display
        clean_col = col.replace("_", " ").title()
        
        is_rate = (col_min >= 0.0 and col_max <= 1.0) or \
                  ("rate" in col_lower or "ratio" in col_lower or "percent" in col_lower or "pct" in col_lower)
                  
        if is_rate:
            display_val = col_mean * 100 if col_max <= 1.0 else col_mean
            name_suffix = " (Avg)" if "avg" not in col_lower and "mean" not in col_lower else ""
            kpis.append({
                "name": f"{clean_col}{name_suffix}",
                "value": round(display_val, 2),
                "type": "Rate/Percentage",
                "source_column": col,
                "interpretation": f"Average rate observed for {col}."
            })
        else:
            is_volume = col_min >= 0 and col_mean > 2 and "day" not in col_lower and "year" not in col_lower and "month" not in col_lower
            
            if is_volume:
                kpis.append({
                    "name": f"Total {clean_col}",
                    "value": round(col_sum, 2) if col_sum % 1 != 0 else int(col_sum),
                    "type": "Volume",
                    "source_column": col,
                    "interpretation": f"Cumulative sum of {col} across all records."
                })
                kpis.append({
                    "name": f"Average {clean_col}",
                    "value": round(col_mean, 2),
                    "type": "Average",
                    "source_column": col,
                    "interpretation": f"Mean value of {col} per record."
                })
            else:
                kpis.append({
                    "name": f"Average {clean_col}",
                    "value": round(col_mean, 2),
                    "type": "Average",
                    "source_column": col,
                    "interpretation": f"Mean value of {col}."
                })
                
    type_priority = {"Count": 1, "Volume": 2, "Rate/Percentage": 3, "Average": 4}
    kpis.sort(key=lambda k: type_priority.get(k["type"], 5))
    
    return kpis
