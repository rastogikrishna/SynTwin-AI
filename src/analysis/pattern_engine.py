import pandas as pd
import numpy as np
from typing import Dict, Any, List

def analyze_patterns(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds numerical correlations, categorical distributions, and temporal trends in df.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame.
    profile : Dict[str, Any]
        The profile dictionary returned by profile_dataset(df).
        
    Returns:
    --------
    Dict[str, Any]
        Structured dictionary of correlations, categorical skews, and temporal trends.
    """
    patterns = {
        "correlations": [],
        "categorical_patterns": [],
        "temporal_patterns": []
    }
    
    total_rows = len(df)
    if total_rows == 0:
        return patterns

    # 1. Numerical Correlations
    numeric_cols = profile.get("data_types", {}).get("numerical", [])
    # Limit number of columns to analyze to prevent slowing down
    if len(numeric_cols) > 1:
        selected_numeric = numeric_cols[:15]
        # Calculate Pearson correlations
        corr_matrix = df[selected_numeric].corr(method='pearson')
        
        seen_pairs = set()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.index:
                if col1 != col2 and (col2, col1) not in seen_pairs:
                    val = corr_matrix.loc[col2, col1]
                    if pd.notna(val) and abs(val) >= 0.4:
                        seen_pairs.add((col1, col2))
                        patterns["correlations"].append({
                            "col1": col1,
                            "col2": col2,
                            "coefficient": float(val),
                            "type": "positive" if val > 0 else "inverse"
                        })
        patterns["correlations"].sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    # 2. Categorical Patterns
    categorical_cols = profile.get("data_types", {}).get("categorical", [])
    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        val_counts = series.value_counts()
        if not val_counts.empty:
            top_val = val_counts.index[0]
            top_pct = (val_counts.iloc[0] / total_rows) * 100
            
            if top_pct >= 40.0:
                patterns["categorical_patterns"].append({
                    "column": col,
                    "dominant_value": str(top_val),
                    "percentage": float(top_pct),
                    "message": f"Column '{col}' is dominated by '{top_val}' ({top_pct:.1f}% of observations)."
                })
                
    # 3. Temporal Patterns
    date_cols = profile.get("inferred_columns", {}).get("dates", []) + profile.get("data_types", {}).get("datetime", [])
    date_cols = list(set(date_cols))
    
    if date_cols and numeric_cols:
        primary_date_col = date_cols[0]
        targets = profile.get("inferred_columns", {}).get("targets", [])
        numeric_targets = [c for c in targets if c in numeric_cols]
        primary_numeric = numeric_targets[0] if numeric_targets else numeric_cols[0]
        
        try:
            temp_df = pd.DataFrame({
                "date": pd.to_datetime(df[primary_date_col], errors='coerce', format='mixed'),
                "value": pd.to_numeric(df[primary_numeric], errors='coerce')
            }).dropna()
            
            if not temp_df.empty:
                temp_df["period"] = temp_df["date"].dt.to_period("M")
                monthly_agg = temp_df.groupby("period")["value"].mean().reset_index()
                
                if len(monthly_agg) >= 3:
                    monthly_agg["time_idx"] = np.arange(len(monthly_agg))
                    corr_val = monthly_agg["time_idx"].corr(monthly_agg["value"])
                    
                    trend_type = "stable"
                    if corr_val > 0.4:
                        trend_type = "increasing"
                    elif corr_val < -0.4:
                        trend_type = "decreasing"
                        
                    patterns["temporal_patterns"].append({
                        "date_column": primary_date_col,
                        "value_column": primary_numeric,
                        "trend_type": trend_type,
                        "correlation": float(corr_val) if pd.notna(corr_val) else 0.0,
                        "periods_count": len(monthly_agg),
                        "message": f"Average '{primary_numeric}' shows an {trend_type} temporal trend over {len(monthly_agg)} monthly intervals based on '{primary_date_col}'."
                    })
        except Exception:
            pass

    return patterns
