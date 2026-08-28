import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from src.decision.objective import calculate_fitness

class GeneticOptimizer:
    """
    Optimizes controllable variables to maximize/minimize a business target using a Genetic Algorithm.
    """
    def __init__(self, 
                 model: Any, 
                 preprocessor: Any, 
                 baseline_row: pd.Series, 
                 controllable_vars: List[Dict[str, Any]], 
                 target_type: str, 
                 objective_mode: str, 
                 class_idx: int = 0,
                 pop_size: int = 50, 
                 generations: int = 20, 
                 mutation_rate: float = 0.15, 
                 crossover_rate: float = 0.8,
                 bounds_dict: Optional[Dict[str, Tuple[float, float]]] = None,
                 allow_out_of_bounds: bool = False):
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_row = baseline_row
        self.controllable_vars = controllable_vars
        self.target_type = target_type
        self.objective_mode = objective_mode
        self.class_idx = class_idx
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.allow_out_of_bounds = allow_out_of_bounds
        
        # Setup bounds
        self.var_names = []
        self.bounds = []
        self.hist_bounds = []
        
        for var in controllable_vars:
            if var["type"] != "numeric":
                continue
            name = var["feature"]
            self.var_names.append(name)
            
            # Default bounds
            low, high = var["min"], var["max"]
            self.hist_bounds.append((low, high))
            
            # User defined bounds
            if bounds_dict and name in bounds_dict:
                low, high = bounds_dict[name]
            self.bounds.append((low, high))
            
    def _evaluate_individual(self, individual: np.ndarray) -> float:
        # Build modified row
        modified = self.baseline_row.copy().astype(object)
        
        # Overwrite controllable variables
        out_of_range_penalty = 0.0
        for i, name in enumerate(self.var_names):
            val = float(individual[i])
            modified[name] = val
            
            # Enforce range constraints if allow_out_of_bounds is False
            if not self.allow_out_of_bounds:
                hist_low, hist_high = self.hist_bounds[i]
                if val < hist_low or val > hist_high:
                    out_of_range_penalty += 1000.0
                    
        # Predict using preprocessor + model
        try:
            row_df = pd.DataFrame([modified])
            X_proc = self.preprocessor.transform_row(row_df)
            pred = self.model.predict(X_proc)[0]
            
            prob = None
            if self.target_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_proc)[0]
                    prob = float(probs[self.class_idx])
                else:
                    prob = 1.0 if pred == self.class_idx else 0.0
                    
            fit = calculate_fitness(pred, self.target_type, self.objective_mode, prob)
            fit -= out_of_range_penalty
            return fit
            
        except Exception:
            return float('-inf')
            
    def optimize(self) -> Dict[str, Any]:
        num_vars = len(self.bounds)
        if num_vars == 0:
            return {
                "recommended_values": {},
                "baseline_prediction": None,
                "optimized_prediction": None,
                "predicted_improvement": 0.0,
                "history": [],
                "out_of_range": False,
                "warnings": []
            }
            
        # 1. Initialize Population
        pop = np.zeros((self.pop_size, num_vars))
        for i, (low, high) in enumerate(self.bounds):
            pop[:, i] = np.random.uniform(low, high, size=self.pop_size)
            
        best_ind = pop[0].copy()
        best_fit = float('-inf')
        history = []
        
        # 2. Main Generation Loop
        for gen in range(self.generations):
            fitnesses = np.array([self._evaluate_individual(ind) for ind in pop])
            
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fit:
                best_fit = fitnesses[max_idx]
                best_ind = pop[max_idx].copy()
                
            history.append(float(best_fit))
            
            # Selection: Tournament
            next_pop = []
            for _ in range(self.pop_size):
                cand_idx = np.random.choice(self.pop_size, size=3, replace=False)
                best_cand_idx = cand_idx[np.argmax(fitnesses[cand_idx])]
                next_pop.append(pop[best_cand_idx].copy())
            pop = np.array(next_pop)
            
            # Crossover: Arithmetic
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.rand()
                    p1, p2 = pop[i].copy(), pop[i+1].copy()
                    pop[i] = alpha * p1 + (1 - alpha) * p2
                    pop[i+1] = (1 - alpha) * p1 + alpha * p2
                    
            # Mutation: Gaussian
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    for j, (low, high) in enumerate(self.bounds):
                        scale = (high - low) * 0.1
                        pop[i, j] += np.random.normal(0, scale)
                        pop[i, j] = np.clip(pop[i, j], low, high)
                        
        # 3. Assemble Output Recommendations
        recommended_values = {}
        out_of_range = False
        warnings = []
        
        for i, name in enumerate(self.var_names):
            val = float(best_ind[i])
            recommended_values[name] = val
            
            hist_low, hist_high = self.hist_bounds[i]
            if val < hist_low or val > hist_high:
                out_of_range = True
                warnings.append(
                    f"Recommended value {val:.2f} for '{name}' is outside historical bounds ({hist_low:.2f} to {hist_high:.2f})."
                )
                
        # Generate baseline vs optimized predictions
        base_df = pd.DataFrame([self.baseline_row])
        X_base = self.preprocessor.transform_row(base_df)
        pred_base = self.model.predict(X_base)[0]
        
        opt_row = self.baseline_row.copy().astype(object)
        for name, val in recommended_values.items():
            opt_row[name] = val
            
        opt_df = pd.DataFrame([opt_row])
        X_opt = self.preprocessor.transform_row(opt_df)
        pred_opt = self.model.predict(X_opt)[0]
        
        # Calculate impact improvement
        prob_base_val = None
        prob_opt_val = None
        if self.target_type == "regression":
            pred_base = float(pred_base)
            pred_opt = float(pred_opt)
            improvement = pred_opt - pred_base
            if self.objective_mode == "Minimize":
                improvement = -improvement
        else:
            if hasattr(self.model, "predict_proba"):
                prob_base_val = float(self.model.predict_proba(X_base)[0][self.class_idx])
                prob_opt_val = float(self.model.predict_proba(X_opt)[0][self.class_idx])
                improvement = prob_opt_val - prob_base_val
                if self.objective_mode == "Minimize":
                    improvement = -improvement
            else:
                improvement = 1.0 if pred_opt != pred_base else 0.0
                
        return {
            "recommended_values": recommended_values,
            "baseline_prediction": float(pred_base) if self.target_type == "regression" else pred_base,
            "optimized_prediction": float(pred_opt) if self.target_type == "regression" else pred_opt,
            "baseline_prob_val": prob_base_val,
            "scenario_prob_val": prob_opt_val,
            "predicted_improvement": improvement,
            "history": history,
            "out_of_range": out_of_range,
            "warnings": warnings,
            "objective_mode": self.objective_mode,
            "generations": self.generations,
            "population_size": self.pop_size
        }
