import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Graceful Gymnasium Ingestion
try:
    import gymnasium as gym
    from gymnasium import spaces
    gym_available = True
    base_env = gym.Env
except ImportError:
    gym_available = False
    base_env = object
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=None):
                self.low = np.array(low)
                self.high = np.array(high)
                self.shape = shape
                self.dtype = dtype

class TwinOptimizationEnv(base_env):
    """
    Gymnasium-compatible simulated environment for RL policy learning.
    State is the controllable variable values. Actions represent step adjustments.
    """
    def __init__(self, 
                 model: Any, 
                 preprocessor: Any, 
                 baseline_row: pd.Series, 
                 controllable_vars: List[Dict[str, Any]], 
                 target_type: str, 
                 objective_mode: str, 
                 class_idx: int = 0,
                 bounds_dict: Optional[Dict[str, Tuple[float, float]]] = None):
        if gym_available:
            super(TwinOptimizationEnv, self).__init__()
            
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_row = baseline_row.copy()
        self.controllable_vars = controllable_vars
        self.target_type = target_type
        self.objective_mode = objective_mode
        self.class_idx = class_idx
        
        self.low_bounds = []
        self.high_bounds = []
        self.var_names = []
        
        for var in controllable_vars:
            if var["type"] != "numeric":
                continue
            name = var["feature"]
            if name not in self.baseline_row.index:
                continue
                
            try:
                # 1. Parse low/high bounds
                low = var.get("min", None)
                high = var.get("max", None)
                if bounds_dict and name in bounds_dict:
                    low, high = bounds_dict[name]
                    
                # Replace infs with NaN in bounds
                if low is not None and (np.isinf(low) or pd.isna(low)):
                    low = np.nan
                if high is not None and (np.isinf(high) or pd.isna(high)):
                    high = np.nan
                    
                # Parse baseline value
                base_val = self.baseline_row[name]
                if isinstance(base_val, str):
                    try:
                        base_val = float(base_val)
                    except ValueError:
                        base_val = np.nan
                else:
                    base_val = float(base_val)
                    
                if np.isinf(base_val) or pd.isna(base_val):
                    base_val = np.nan
                    
                # Impute baseline value if NaN
                if np.isnan(base_val):
                    median_val = None
                    try:
                        if hasattr(preprocessor, "num_feats") and name in preprocessor.num_feats:
                            idx = preprocessor.num_feats.index(name)
                            imputer = preprocessor.preprocessor.named_transformers_['num'].named_steps['imputer']
                            median_val = float(imputer.statistics_[idx])
                    except Exception:
                        pass
                        
                    if median_val is not None and not np.isnan(median_val) and not np.isinf(median_val):
                        base_val = median_val
                    else:
                        if low is not None and not np.isnan(low) and not np.isinf(low):
                            base_val = low
                        elif high is not None and not np.isnan(high) and not np.isinf(high):
                            base_val = high
                        else:
                            base_val = 0.0
                            
                # Update baseline_row with cleaned/imputed value
                self.baseline_row[name] = float(base_val)
                
                # Impute low/high bounds if NaN
                if low is None or np.isnan(low):
                    low = base_val - 1.0
                if high is None or np.isnan(high):
                    high = base_val + 1.0
                    
                if low >= high:
                    high = low + 1.0
                    
                self.var_names.append(name)
                self.low_bounds.append(float(low))
                self.high_bounds.append(float(high))
            except Exception:
                continue
                
        self.num_vars = len(self.var_names)
        if self.num_vars == 0:
            raise ValueError("No controllable numeric variables found.")
            
        self.low_bounds = np.array(self.low_bounds, dtype=np.float32)
        self.high_bounds = np.array(self.high_bounds, dtype=np.float32)
        
        # Double check bounds are finite
        if not np.isfinite(self.low_bounds).all():
            self.low_bounds = np.nan_to_num(self.low_bounds, nan=-10.0, posinf=10.0, neginf=-10.0)
        if not np.isfinite(self.high_bounds).all():
            self.high_bounds = np.nan_to_num(self.high_bounds, nan=10.0, posinf=10.0, neginf=-10.0)
            
        # Action space: values in [-1, 1] mapped to step shifts
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_vars,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_bounds, high=self.high_bounds, dtype=np.float32)
            
        self.state = None
        self.steps_taken = 0
        self.max_steps = 15
        
    def reset(self, seed=None, options=None):
        if gym_available:
            super().reset(seed=seed)
        self.steps_taken = 0
        self.state = np.array([float(self.baseline_row[name]) for name in self.var_names], dtype=np.float32)
        
        # Replace non-finite state values just in case
        if not np.isfinite(self.state).all():
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=1.0, neginf=-1.0)
            
        self.state = np.clip(self.state, self.low_bounds, self.high_bounds)
        return self.state, {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps_taken += 1
        
        # Ensure action contains only finite values
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Action scaling: scale actions in [-1, 1] to max 10% of the variable range per step
        scaled_action = action * (self.high_bounds - self.low_bounds) * 0.1
        self.state = self.state + scaled_action
        
        # Check and clip to bounds
        if not np.isfinite(self.state).all():
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=1.0, neginf=-1.0)
            
        clipped_state = np.clip(self.state, self.low_bounds, self.high_bounds)
        penalty = 0.0
        if not np.allclose(self.state, clipped_state, equal_nan=False):
            penalty -= 10.0  # Violation penalty
            self.state = clipped_state
            
        # Digital Twin execution
        modified_row = self.baseline_row.copy().astype(object)
        for i, name in enumerate(self.var_names):
            modified_row[name] = float(self.state[i])
            
        try:
            row_df = pd.DataFrame([modified_row])
            X_proc = self.preprocessor.transform_row(row_df)
            pred = self.model.predict(X_proc)[0]
            
            prob = None
            if self.target_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_proc)[0]
                    prob = float(probs[self.class_idx])
                else:
                    prob = 1.0 if pred == self.class_idx else 0.0
                    
            from src.decision.objective import calculate_fitness
            reward = calculate_fitness(pred, self.target_type, self.objective_mode, prob)
            
        except Exception:
            reward = -100.0
            
        reward += penalty
        
        # Penalize excessive action sizes to keep recommendations realistic
        reward -= 0.05 * np.sum(np.abs(action))
        
        # Ensure reward is finite
        if not np.isfinite(reward):
            reward = -100.0
            
        terminated = False
        truncated = self.steps_taken >= self.max_steps
        
        return self.state, float(reward), terminated, truncated, {}
