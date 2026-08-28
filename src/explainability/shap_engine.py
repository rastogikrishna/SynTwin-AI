import numpy as np
import shap
from typing import Dict, Any, List, Tuple


def _make_background(X_train: np.ndarray, max_rows: int = 100) -> np.ndarray:
    """Downsample the background/reference dataset used by Linear/Kernel
    explainers, purely for speed. Never touches the evaluation rows, so it
    cannot cause a row-index mismatch downstream."""
    if X_train.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        indices = rng.choice(X_train.shape[0], size=max_rows, replace=False)
        return X_train[indices]
    return X_train


def _compute_shap(model: Any, X_background: np.ndarray, X_eval: np.ndarray) -> Tuple[Any, Any]:
    """Runs the appropriate SHAP explainer for X_eval exactly as given —
    callers are responsible for deciding how many/which rows go into X_eval,
    so the returned shap_values always line up 1:1 with X_eval's rows."""
    model_name = type(model).__name__.lower()

    if "randomforest" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)
        base_value = explainer.expected_value
    elif "logisticregression" in model_name or "linearregression" in model_name:
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X_eval)
        base_value = explainer.expected_value
    else:
        # KernelExplainer is slow — cap the number of evaluation rows it has
        # to run on. Only used for the "global importance" path (many rows);
        # explain_locally always passes exactly 1 row, so this cap never
        # discards the specific record the user asked to explain.
        X_eval_capped = X_eval[:30] if X_eval.shape[0] > 30 else X_eval
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(X_eval_capped)
        base_value = explainer.expected_value

    return shap_values, base_value


def get_shap_values(model: Any, X_train: np.ndarray, X_test: np.ndarray, target_type: str, class_idx: int = 0) -> Tuple[np.ndarray, float]:
    """
    Computes SHAP values for a *batch* of evaluation rows (used for global
    feature importance, where averaging over many rows is the point).

    Performance Safeguards:
    ----------------------
    - Background: Limits background dataset to 100 rows to ensure fast execution.
    - Test: Limits evaluation to maximum 200 rows (this function is for
      aggregate/global importance only — see explain_locally() for
      single-record explanations, which never truncates the requested row).
    """
    X_background = _make_background(X_train)

    X_eval = X_test
    if X_test.shape[0] > 200:
        X_eval = X_test[:200]

    try:
        shap_values, base_value = _compute_shap(model, X_background, X_eval)
        extracted_shap = _extract_class_shap(shap_values, target_type, class_idx)
        extracted_base = _extract_class_base_value(base_value, class_idx)
        return extracted_shap, extracted_base
    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {str(e)}")

def _extract_class_shap(shap_values: Any, target_type: str, class_idx: int) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) > class_idx:
            return np.array(shap_values[class_idx])
        return np.array(shap_values[-1])
        
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:  # shape: (rows, features, classes)
            if shap_values.shape[2] > class_idx:
                return shap_values[:, :, class_idx]
            return shap_values[:, :, -1]
        return shap_values
        
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if len(vals.shape) == 3:
            if vals.shape[2] > class_idx:
                return vals[:, :, class_idx]
            return vals[:, :, -1]
        return vals
        
    return np.array(shap_values)

def _extract_class_base_value(base_value: Any, class_idx: int) -> float:
    if isinstance(base_value, (list, np.ndarray)):
        if len(base_value) > class_idx:
            return float(base_value[class_idx])
        return float(base_value[-1])
    return float(base_value)

def explain_globally(model: Any, X_train: np.ndarray, X_test: np.ndarray, 
                     feature_names: List[str], target_type: str, class_idx: int = 0) -> List[Dict[str, Any]]:
    """
    Calculates global feature importance based on mean absolute SHAP values.
    """
    shap_vals, _ = get_shap_values(model, X_train, X_test, target_type, class_idx)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    
    total_imp = sum(mean_abs)
    importance_list = []
    
    for name, val in zip(feature_names, mean_abs):
        rel_imp = float(val / total_imp) if total_imp > 0 else 0.0
        importance_list.append({
            "feature": name,
            "importance": float(val),
            "relative_importance": rel_imp
        })
        
    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list

def explain_locally(model: Any, X_train: np.ndarray, X_test: np.ndarray,
                    row_idx: int, feature_names: List[str], target_type: str, class_idx: int = 0,
                    raw_row=None, raw_feature_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Explains an individual prediction (local explanation) for a specific row index.

    Computes SHAP for exactly the one requested row (X_test[row_idx:row_idx+1]),
    never a truncated batch — a previous version reused the batch-truncated
    global-importance path (capped at 200, or 30 for KernelExplainer models)
    and silently clamped row_idx down to the last available row when it
    exceeded that cap, so selecting e.g. record #800 on a large test set
    would silently show the SHAP breakdown for record #199 while the
    Actual/Predicted metrics above it still correctly showed record #800 —
    a confusing mismatch. That can no longer happen: this function only ever
    asks SHAP to explain the single row the caller asked for.

    Parameters
    ----------
    raw_row : pd.Series, optional
        The original (pre-preprocessing) row, e.g. preprocessor.X_test_raw.iloc[row_idx].
        When provided along with raw_feature_map, feature_value in each
        contribution is the real business value (e.g. "$4,200" / "West
        Region") instead of the scaled/one-hot-encoded number SHAP actually
        operates on (e.g. "-0.42" / "1.0"), which is what a preprocessed
        feature matrix looks like and is not meaningful to a business user.
    raw_feature_map : dict, optional
        Maps each processed feature name (e.g. "region_West") back to the
        original raw column name (e.g. "region"). See
        DataPreprocessor.get_raw_feature_map().
    """
    if row_idx < 0 or row_idx >= X_test.shape[0]:
        raise IndexError(f"row_idx {row_idx} is out of range for a test set of size {X_test.shape[0]}.")

    X_background = _make_background(X_train)
    X_single = X_test[row_idx:row_idx + 1]

    try:
        shap_values, base_value = _compute_shap(model, X_background, X_single)
        row_shap = _extract_class_shap(shap_values, target_type, class_idx)[0]
        base_value = _extract_class_base_value(base_value, class_idx)
    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {str(e)}")

    row_feat = X_test[row_idx]

    contribs = []
    for i, (name, s_val, f_val) in enumerate(zip(feature_names, row_shap, row_feat)):
        direction = "increase" if s_val > 0 else "decrease"

        display_val = float(f_val) if isinstance(f_val, (int, float, np.floating, np.integer)) else str(f_val)
        if raw_row is not None and raw_feature_map is not None:
            raw_col = raw_feature_map.get(name)
            if raw_col is not None and raw_col in raw_row.index:
                rv = raw_row[raw_col]
                display_val = float(rv) if isinstance(rv, (int, float, np.floating, np.integer)) else str(rv)

        contribs.append({
            "feature": name,
            "feature_value": display_val,
            "shap_value": float(s_val),
            "direction": direction,
            "message": f"'{name}' contributed to pushing the prediction {'higher' if s_val > 0 else 'lower'} (SHAP impact: {s_val:+.4f})."
        })
        
    contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    positives = [c for c in contribs if c["shap_value"] > 0]
    negatives = [c for c in contribs if c["shap_value"] < 0]
    
    return {
        "base_value": base_value,
        "prediction_value": float(base_value + np.sum(row_shap)),
        "contributions": contribs,
        "positives": positives,
        "negatives": negatives
    }
