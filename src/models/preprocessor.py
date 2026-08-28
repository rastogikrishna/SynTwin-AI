from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class DataPreprocessor:
    """
    A reusable preprocessing pipeline that cleans numeric/categorical features,
    extracts date parts, splits data chronologically or randomly, and prevents data leakage.
    """
    def __init__(self, target_col: str, target_type: str, date_col: Optional[str] = None):
        self.target_col = target_col
        self.target_type = target_type
        self.date_col = date_col
        self.preprocessor = None
        self.feature_names = []
        self.feature_cols = []
        
    def split_and_preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        1. Extracts date features if date_col is present.
        2. Drops targets, IDs, and highly missing columns from features X.
        3. Splits X and y into train and test sets (chronological or random).
        4. Fits preprocessor on X_train and transforms X_train and X_test.
        5. Returns X_train_proc, X_test_proc, y_train, y_test, and feature_names.
        
        Safeguards:
        -----------
        - Leakage: The target column is popped from features immediately.
        - Leakage: Preprocessing fits on training partition only. Test set is transformed.
        - Leakage: If date_col is present, chronological split is used so future data is not leaked.
        - Leakage: Excludes identifiers/sequence indices from feature columns.
        """
        df = df.copy()
        
        # Normalize pandas nullable, boolean, and mixed object/categorical/string dtypes to standard numpy/python types
        for col in df.columns:
            dt_name = df[col].dtype.name
            if dt_name in ['Int64', 'Int32', 'Int16', 'Int8', 'UInt64', 'UInt32', 'UInt16', 'UInt8', 'Float64', 'Float32', 'boolean', 'bool']:
                df[col] = df[col].astype(float)
            elif dt_name.lower() in ["object", "category", "string"]:
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

        # 1. Derive datetime features if date_col is present and exists in df
        derived_date_cols = []
        if self.date_col and self.date_col in df.columns:
            try:
                # Convert to datetime
                df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce', format='mixed')
                
                # Simple imputation for date gaps
                if df[self.date_col].isna().sum() > 0:
                    mode_val = df[self.date_col].mode()
                    fill_date = mode_val[0] if not mode_val.empty else pd.Timestamp("2026-01-01")
                    df[self.date_col] = df[self.date_col].fillna(fill_date)
                
                # Extract temporal details
                df[f"{self.date_col}_year"] = df[self.date_col].dt.year
                df[f"{self.date_col}_month"] = df[self.date_col].dt.month
                df[f"{self.date_col}_day"] = df[self.date_col].dt.day
                df[f"{self.date_col}_dayofweek"] = df[self.date_col].dt.dayofweek
                
                derived_date_cols = [
                    f"{self.date_col}_year",
                    f"{self.date_col}_month",
                    f"{self.date_col}_day",
                    f"{self.date_col}_dayofweek"
                ]
            except Exception:
                pass
                
        # 2. Drop rows with a missing target value (a target can have up to ~25%
        # missingness and still be selected by the target detector, but a
        # supervised model cannot be fit against an unknown label).
        if df[self.target_col].isna().any():
            df = df[df[self.target_col].notna()].reset_index(drop=True)

        # Separate target and features
        y = df[self.target_col].values
        
        # Base drop list
        drop_cols = [self.target_col]
        if self.date_col and self.date_col in df.columns:
            drop_cols.append(self.date_col)
            
        # Detect identifiers, high missing rates (>80%), or constant columns to exclude
        for col in df.columns:
            if col in drop_cols or col in derived_date_cols:
                continue
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            
            # Simple ID/unique signature detection (exclude float features from unique check)
            is_float = pd.api.types.is_float_dtype(df[col])
            is_id = "id" in col.lower() or "key" in col.lower() or "code" in col.lower() or (df[col].nunique() == len(df) and not is_float)
            is_const = df[col].nunique(dropna=True) <= 1
            is_high_card = (not is_float) and (not pd.api.types.is_numeric_dtype(df[col])) and (df[col].nunique() > 100)
            if missing_pct > 80.0 or is_id or is_const or is_high_card:
                drop_cols.append(col)
                
        X = df.drop(columns=drop_cols)
        
        # 3. Train/Test Splitting
        if self.date_col and self.date_col in df.columns:
            # Chronological splitting (prevents temporal leak)
            sort_idx = df[self.date_col].argsort()
            X_sorted = X.iloc[sort_idx]
            y_sorted = y[sort_idx]
            
            split_idx = int(len(X_sorted) * 0.8)
            X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
            y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]
        else:
            # Random splitting
            stratify = y if self.target_type in ["binary_classification", "multiclass_classification"] else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify
                )
            except ValueError:
                # Fallback if classes are too small/imbalanced for stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
        # 4. Set numerical & categorical lists
        self.X_test_raw = X_test
        num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        
        # 5. Define pipelines
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_feats),
            ('cat', cat_transformer, cat_feats)
        ])
        
        # 6. Fit only on training data (Leakage Safeguard)
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        
        # 7. Extract Feature Names
        self.feature_cols = X.columns.tolist()
        self.feature_names = list(num_feats)
        if cat_feats:
            try:
                onehot_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_names = list(onehot_encoder.get_feature_names_out(cat_feats))
                self.feature_names.extend(cat_names)
            except Exception:
                self.feature_names.extend([f"{c}_encoded" for c in cat_feats])
                
        return X_train_proc, X_test_proc, y_train, y_test, self.feature_names

    def get_raw_feature_map(self) -> dict:
        """Maps every processed/encoded feature name (what SHAP and the
        trained model actually operate on — e.g. a one-hot column like
        'region_West' or a scaled numeric column) back to the original raw
        column name in the source dataset (e.g. 'region'). Used to show
        business-meaningful values ($4,200, "West") in the Explainability
        page instead of the scaled/encoded numbers the model sees
        internally (-0.42, 1.0), which are not meaningful on their own.
        """
        feature_map = {}
        for name in self.num_feats:
            feature_map[name] = name
        for name in self.feature_names:
            if name in feature_map:
                continue
            # One-hot encoded names look like f"{original_col}_{category}"
            # (sklearn's OneHotEncoder.get_feature_names_out format). Match
            # the longest cat_feats prefix so columns with underscores in
            # their own names are still matched correctly.
            best_match = None
            for col in self.cat_feats:
                if name == col or name.startswith(col + "_"):
                    if best_match is None or len(col) > len(best_match):
                        best_match = col
            feature_map[name] = best_match if best_match else name
        return feature_map

    def transform_row(self, df_row: pd.DataFrame) -> np.ndarray:
        """
        Transforms a single row DataFrame using the fitted components.
        """
        df_row = df_row.copy()
        
        # Normalize pandas nullable, boolean, and mixed object/categorical/string dtypes to standard numpy/python types
        for col in df_row.columns:
            dt_name = df_row[col].dtype.name
            if dt_name in ['Int64', 'Int32', 'Int16', 'Int8', 'UInt64', 'UInt32', 'UInt16', 'UInt8', 'Float64', 'Float32', 'boolean', 'bool']:
                df_row[col] = df_row[col].astype(float)
            elif dt_name.lower() in ["object", "category", "string"]:
                df_row[col] = df_row[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

        # Derive date features
        if self.date_col and self.date_col in df_row.columns:
            try:
                df_row[self.date_col] = pd.to_datetime(df_row[self.date_col], errors='coerce', format='mixed')
                if df_row[self.date_col].isna().sum() > 0:
                    df_row[self.date_col] = df_row[self.date_col].fillna(pd.Timestamp("2026-01-01"))
                
                df_row[f"{self.date_col}_year"] = df_row[self.date_col].dt.year
                df_row[f"{self.date_col}_month"] = df_row[self.date_col].dt.month
                df_row[f"{self.date_col}_day"] = df_row[self.date_col].dt.day
                df_row[f"{self.date_col}_dayofweek"] = df_row[self.date_col].dt.dayofweek
            except Exception:
                pass
                
        # Handle columns alignment
        for col in self.feature_cols:
            if col not in df_row.columns:
                df_row[col] = np.nan
                
        X_row = df_row[self.feature_cols]
        return self.preprocessor.transform(X_row)
