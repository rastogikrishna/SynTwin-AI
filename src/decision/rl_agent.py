import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

try:
    from stable_baselines3 import PPO
    rl_available = True
except ImportError:
    rl_available = False

def train_rl_agent(env: Any, total_timesteps: int = 1000) -> Dict[str, Any]:
    """
    Trains a stable-baselines3 PPO agent inside the Gymnasium optimization environment.
    
    Parameters:
    -----------
    env : Any
        Instantiated Gymnasium-compatible environment.
    total_timesteps : int
        Number of steps to train.
        
    Returns:
    --------
    Dict[str, Any]
        Training results dictionary.
    """
    if not rl_available:
        return {
            "status": "unavailable",
            "reason": "Stable-Baselines3 or PyTorch packages are not installed in the environment."
        }
        
    try:
        # Create a lightweight PPO agent with small batch sizes and network structures
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=16,
            n_epochs=4,
            gamma=0.99,
            verbose=0,
            seed=42
        )
        
        # Train
        model.learn(total_timesteps=total_timesteps)
        
        # Roll out policy from initial state
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            
        # Compile recommended actions mapping
        recommended_values = {}
        for i, name in enumerate(env.var_names):
            recommended_values[name] = float(obs[i])
            
        # Predict using preprocessor + model
        base_df = pd.DataFrame([env.baseline_row])
        X_base = env.preprocessor.transform_row(base_df)
        pred_base = env.model.predict(X_base)[0]
        
        opt_row = env.baseline_row.copy().astype(object)
        for name, val in recommended_values.items():
            opt_row[name] = val
            
        opt_df = pd.DataFrame([opt_row])
        X_opt = env.preprocessor.transform_row(opt_df)
        pred_opt = env.model.predict(X_opt)[0]
        
        # Calculate improvement
        prob_base_val = None
        prob_opt_val = None
        
        if env.target_type == "regression":
            pred_base = float(pred_base)
            pred_opt = float(pred_opt)
            improvement = pred_opt - pred_base
            if env.objective_mode == "Minimize":
                improvement = -improvement
        else:
            if hasattr(env.model, "predict_proba"):
                prob_base_val = float(env.model.predict_proba(X_base)[0][env.class_idx])
                prob_opt_val = float(env.model.predict_proba(X_opt)[0][env.class_idx])
                improvement = prob_opt_val - prob_base_val
                if env.objective_mode == "Minimize":
                    improvement = -improvement
            else:
                improvement = 1.0 if pred_opt != pred_base else 0.0
                
        return {
            "status": "success",
            "recommended_values": recommended_values,
            "baseline_prediction": float(pred_base) if env.target_type == "regression" else pred_base,
            "optimized_prediction": float(pred_opt) if env.target_type == "regression" else pred_opt,
            "baseline_prob_val": prob_base_val,
            "scenario_prob_val": prob_opt_val,
            "predicted_improvement": improvement,
            "timesteps": total_timesteps
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "reason": f"RL Training Exception: {str(e)}"
        }
