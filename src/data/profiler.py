import pandas as pd
import numpy as np
from typing import Dict, Any, List

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profiles a Pandas DataFrame and returns a comprehensive dictionary 
    of statistics, column info, data quality alerts, and detected columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to profile.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary of dataset metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Dataset Overview
    memory_bytes = int(df.memory_usage(deep=True).sum())
    duplicate_rows = int(df.duplicated().sum())
    
    overview = {
        "num_rows": total_rows,
        "num_columns": total_cols,
        "memory_bytes": memory_bytes,
        "memory_formatted": _format_size(memory_bytes),
        "duplicate_rows": duplicate_rows
    }
    
    # 2. Type Detection & Column Roles Inferences
    numeric_cols = []
    categorical_cols = []
    boolean_cols = []
    datetime_cols = []
    
    potential_ids = []
    potential_dates = []
    potential_targets = []
    
    column_profiles = {}
    
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        unique_count = int(series.nunique())
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / total_rows) * 100) if total_rows > 0 else 0.0
        
        # General Column Info
        col_info = {
            "name": col,
            "dtype": dtype,
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
        }
        
        # Determine base types and date/ID potentials
        is_bool = pd.api.types.is_bool_dtype(series) or (unique_count == 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0, '0', '1', 'True', 'False', 'true', 'false', 'T', 'F', 'Y', 'N', 'y', 'n'}))
        is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool
        is_datetime = pd.api.types.is_datetime64_any_dtype(series) or "datetime" in dtype
        
        # Check if object/string column is date-like
        is_potential_date = False
        if not is_datetime and not is_numeric and not is_bool:
            non_null_sample = series.dropna().head(100)
            if not non_null_sample.empty:
                try:
                    sample_strs = non_null_sample.astype(str)
                    if all(s.isdigit() and len(s) < 4 for s in sample_strs):
                        is_potential_date = False
                    else:
                        parsed = pd.to_datetime(sample_strs, errors='coerce', format='mixed')
                        parsed_ratio = parsed.notna().sum() / len(non_null_sample)
                        if parsed_ratio > 0.8:
                            is_potential_date = True
                except Exception:
                    pass
        
        # Column categorization
        if is_datetime or is_potential_date:
            datetime_cols.append(col)
            if is_potential_date:
                potential_dates.append(col)
            type_group = "datetime"
        elif is_bool:
            boolean_cols.append(col)
            type_group = "boolean"
        elif is_numeric:
            numeric_cols.append(col)
            type_group = "numeric"
        else:
            categorical_cols.append(col)
            type_group = "categorical"
            
        col_info["type_group"] = type_group
        
        # Infer potential IDs
        # Characteristics: unique values ratio is very high (>= 99%) and is not a float or boolean
        non_null_count = total_rows - missing_count
        if non_null_count > 0:
            unique_ratio = unique_count / non_null_count
            is_float = pd.api.types.is_float_dtype(series)
            if unique_ratio >= 0.99 and unique_count >= 5 and not is_float and not is_bool:
                potential_ids.append(col)

        # Categorical Summary
        if type_group in ["categorical", "boolean"] or (type_group == "numeric" and unique_count < 10):
            if not series.dropna().empty:
                val_counts = series.value_counts()
                top_val = val_counts.index[0]
                top_freq = int(val_counts.iloc[0])
                col_info["categorical_summary"] = {
                    "num_categories": unique_count,
                    "top_value": str(top_val),
                    "top_frequency": top_freq,
                    "top_percentage": float((top_freq / total_rows) * 100) if total_rows > 0 else 0.0,
                    "top_frequencies": {str(k): int(v) for k, v in val_counts.head(5).items()}
                }
            else:
                col_info["categorical_summary"] = {
                    "num_categories": 0,
                    "top_value": None,
                    "top_frequency": 0,
                    "top_percentage": 0.0,
                    "top_frequencies": {}
                }
                
        # Numerical Summary
        if type_group == "numeric":
            if not series.dropna().empty:
                clean_s = series.dropna()
                col_info["numerical_summary"] = {
                    "mean": float(clean_s.mean()) if not isinstance(clean_s.mean(), complex) else None,
                    "median": float(clean_s.median()) if not isinstance(clean_s.median(), complex) else None,
                    "std": float(clean_s.std()) if len(clean_s) > 1 and not isinstance(clean_s.std(), complex) else 0.0,
                    "min": float(clean_s.min()) if not isinstance(clean_s.min(), complex) else None,
                    "max": float(clean_s.max()) if not isinstance(clean_s.max(), complex) else None,
                    "q25": float(clean_s.quantile(0.25)) if not isinstance(clean_s.quantile(0.25), complex) else None,
                    "q50": float(clean_s.quantile(0.50)) if not isinstance(clean_s.quantile(0.50), complex) else None,
                    "q75": float(clean_s.quantile(0.75)) if not isinstance(clean_s.quantile(0.75), complex) else None,
                }
            else:
                col_info["numerical_summary"] = {}

        column_profiles[col] = col_info

    # 3. Target / Business Metric Inference
    for col in df.columns:
        if col in potential_ids or col in datetime_cols or col in potential_dates:
            continue
        series = df[col]
        unique_count = series.nunique()
        
        # Categorical targets
        if col in categorical_cols or col in boolean_cols:
            if 2 <= unique_count <= 15:
                is_like_zip = "zip" in col.lower() or "post" in col.lower()
                if not is_like_zip:
                    potential_targets.append(col)
        # Numerical targets
        elif col in numeric_cols:
            is_lat_lon = "latitude" in col.lower() or "longitude" in col.lower() or "lat" in col.lower() or "lon" in col.lower()
            if not is_lat_lon and unique_count > 1:
                std = series.std()
                if pd.notna(std) and std > 0:
                    potential_targets.append(col)

    # 4. Data Quality Warnings / Alerts
    warnings = []
    
    if duplicate_rows > 0:
        warnings.append({
            "column": "Dataset Level",
            "type": "duplicates",
            "severity": "medium",
            "message": f"Dataset contains {duplicate_rows} duplicate rows."
        })

    for col, info in column_profiles.items():
        # High missing values warning
        if info["missing_percentage"] > 50:
            warnings.append({
                "column": col,
                "type": "high_missing",
                "severity": "high" if info["missing_percentage"] > 90 else "medium",
                "message": f"Column '{col}' has {info['missing_percentage']:.2f}% missing values."
            })
            
        # Constant column warning
        if info["unique_count"] == 1:
            warnings.append({
                "column": col,
                "type": "constant",
                "severity": "medium",
                "message": f"Column '{col}' is constant (has only 1 unique value)."
            })

        # Suspicious ID warning
        is_id_name = "id" in col.lower() or "key" in col.lower() or "code" in col.lower()
        non_null_count = total_rows - info["missing_count"]
        if is_id_name and non_null_count > 0:
            unique_ratio = info["unique_count"] / non_null_count
            if unique_ratio < 0.10:
                warnings.append({
                    "column": col,
                    "type": "suspicious_id",
                    "severity": "low",
                    "message": f"Column '{col}' contains ID/Key terminology in its name, but has very low uniqueness ({unique_ratio*100:.2f}% unique values). It might represent a categorical code rather than a unique identifier."
                })
                
        # High cardinality warning
        if info["type_group"] == "categorical":
            if non_null_count > 0:
                unique_ratio = info["unique_count"] / non_null_count
                if unique_ratio > 0.90 and info["unique_count"] > 10:
                    warnings.append({
                        "column": col,
                        "type": "high_cardinality",
                        "severity": "low",
                        "message": f"Categorical column '{col}' has very high cardinality: {info['unique_count']} unique values ({unique_ratio*100:.1f}% ratio)."
                    })
                    
        # Date parsing issues
        is_date_name = "date" in col.lower() or "time" in col.lower()
        if is_date_name and col not in datetime_cols:
            warnings.append({
                "column": col,
                "type": "date_parsing_issue",
                "severity": "medium",
                "message": f"Column '{col}' contains date/time terminology in its name, but was not automatically classified as a date. Inconsistent formats may prevent date parsing."
            })

    # Overall dataset quality status: Good, Needs Attention, Critical
    overall_status = "Good"
    high_count = sum(1 for w in warnings if w["severity"] == "high")
    med_count = sum(1 for w in warnings if w["severity"] == "medium")
    
    if high_count > 0:
        overall_status = "Critical"
    elif med_count > 0 or len(warnings) > 3:
        overall_status = "Needs Attention"

    return {
        "overview": overview,
        "column_profiles": column_profiles,
        "data_types": {
            "numerical": numeric_cols,
            "categorical": categorical_cols,
            "boolean": boolean_cols,
            "datetime": datetime_cols
        },
        "inferred_columns": {
            "ids": potential_ids,
            "dates": potential_dates,
            "targets": potential_targets
        },
        "warnings": warnings,
        "overall_status": overall_status
    }

def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

class DatasetProfiler:
    """Compatibility class wrapping profile_dataset."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def get_basic_stats(self) -> Dict[str, Any]:
        profile = profile_dataset(self.df)
        return {
            "num_rows": profile["overview"]["num_rows"],
            "num_columns": profile["overview"]["num_columns"],
            "memory_usage_bytes": profile["overview"]["memory_bytes"],
            "memory_usage_formatted": profile["overview"]["memory_formatted"],
            "duplicate_rows": profile["overview"]["duplicate_rows"]
        }
    def profile_columns(self) -> Dict[str, Dict[str, Any]]:
        profile = profile_dataset(self.df)
        return profile["column_profiles"]
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        profile = profile_dataset(self.df)
        anomalies = []
        for w in profile["warnings"]:
            anomalies.append({
                "column": w["column"],
                "type": w["type"],
                "severity": w["severity"],
                "message": w["message"]
            })
        return anomalies
