from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
import numpy as np
from typing import Tuple, Dict, Any

def train_best_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     target_type: str) -> Tuple[Any, str, Dict[str, float]]:
    """
    Trains multiple scikit-learn baseline models and selects the best one.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Training target labels.
    X_test : np.ndarray
        Preprocessed testing features.
    y_test : np.ndarray
        Testing target labels.
    target_type : str
        Target type ("binary_classification", "multiclass_classification", "regression").

    Returns:
    --------
    Tuple[Any, str, Dict[str, float]]
        - Best fitted model instance.
        - Best model name (str).
        - Dict of all trained models and their comparison scores.
    """
    scores = {}
    models = {}
    
    if target_type in ["binary_classification", "multiclass_classification"]:
        # 1. Logistic Regression
        clf1_name = "Logistic Regression"
        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        clf1.fit(X_train, y_train)
        pred1 = clf1.predict(X_test)
        score1 = float(f1_score(y_test, pred1, average='macro'))
        scores[clf1_name] = score1
        models[clf1_name] = clf1
        
        # 2. Random Forest Classifier
        clf2_name = "Random Forest Classifier"
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf2.fit(X_train, y_train)
        pred2 = clf2.predict(X_test)
        score2 = float(f1_score(y_test, pred2, average='macro'))
        scores[clf2_name] = score2
        models[clf2_name] = clf2
        
        best_name = max(scores, key=scores.get)
        best_model = models[best_name]
        
    elif target_type == "regression":
        # 1. Linear Regression
        reg1_name = "Linear Regression"
        reg1 = LinearRegression()
        reg1.fit(X_train, y_train)
        pred1 = reg1.predict(X_test)
        score1 = float(r2_score(y_test, pred1))
        scores[reg1_name] = score1
        models[reg1_name] = reg1
        
        # 2. Random Forest Regressor
        reg2_name = "Random Forest Regressor"
        reg2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        reg2.fit(X_train, y_train)
        pred2 = reg2.predict(X_test)
        score2 = float(r2_score(y_test, pred2))
        scores[reg2_name] = score2
        models[reg2_name] = reg2
        
        best_name = max(scores, key=scores.get)
        best_model = models[best_name]
        
    else:
        raise ValueError(f"Unknown target type: {target_type}")
        
    return best_model, best_name, scores
