import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import statsmodels.api as sm
from modules.config import PROJECT_ROOT

def calibrate_mfrm_ground_truth(ratings_records: list) -> pd.DataFrame:
    df = pd.DataFrame(ratings_records)
    ols_fit = sm.OLS.from_formula("raw_score ~ C(utterance_id) + C(prompt_id)", data=df).fit()
    
    latent_thetas = {}
    for k, v in ols_fit.params.items():
        if "C(utterance_id)[T." in k:
            uid = k.replace("C(utterance_id)[T.", "").replace("]", "")
            latent_thetas[uid] = v

    base_uid = df["utterance_id"].iloc[0]
    if base_uid not in latent_thetas:
        latent_thetas[base_uid] = 0.0

    vals = np.array(list(latent_thetas.values()))
    mu, sigma = np.mean(vals), np.std(vals)

    records = []
    for uid, th in latent_thetas.items():
        z = (th - mu) / (sigma + 1e-8)
        cefr_val = np.clip(3.20 + (z * 0.95), 1.00, 6.00)
        records.append({"utterance_id": uid, "cefr_target": round(float(cefr_val), 2)})
    return pd.DataFrame(records)

class CEFRStackingEnsembleHead:
    def __init__(self, model_save_path: str = None):
        self.model_save_path = model_save_path if model_save_path else str(PROJECT_ROOT / "cefr_xgboost_head.json")
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            random_state=42
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )
        self.meta_weights = [0.50, 0.30, 0.20]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.gb_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)

        p1 = self.xgb_model.predict(X_val)
        p2 = self.gb_model.predict(X_val)
        p3 = self.rf_model.predict(X_val)
        
        ensemble_preds = (self.meta_weights[0] * p1) + (self.meta_weights[1] * p2) + (self.meta_weights[2] * p3)
        ensemble_preds = np.clip(ensemble_preds, 1.00, 6.00)
        
        rmse = root_mean_squared_error(y_val, ensemble_preds)
        r2 = r2_score(y_val, ensemble_preds)
        
        b_true = np.round(y_val).astype(int)
        b_pred = np.round(ensemble_preds).astype(int)
        exact_acc = np.mean(b_true == b_pred) * 100.0
        adj_acc = np.mean(np.abs(b_true - b_pred) <= 1) * 100.0

        print(f"[Stacking Ensemble Head Fit Complete]:")
        print(f"  - Validation RMSE           : {rmse:.4f}")
        print(f"  - Validation R² Score       : {r2:.4f}")
        print(f"  - Exact CEFR Band Accuracy  : {exact_acc:.2f}% (Target: >90%)")
        print(f"  - Adjacent Band Accuracy    : {adj_acc:.2f}%")

        self.xgb_model.save_model(self.model_save_path)
        print(f"  - Stacking Primary Model saved to -> {self.model_save_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.xgb_model.predict(X)
        p2 = self.gb_model.predict(X)
        p3 = self.rf_model.predict(X)
        preds = (self.meta_weights[0] * p1) + (self.meta_weights[1] * p2) + (self.meta_weights[2] * p3)
        return np.clip(preds, 1.00, 6.00)

def train_and_save_xgboost_head(X: np.ndarray, y: np.ndarray, model_save_path: str = None):
    ensemble = CEFRStackingEnsembleHead(model_save_path)
    ensemble.fit(X, y)
    return ensemble.xgb_model

if __name__ == "__main__":
    np.random.seed(42)
    N = 600
    X_sim = np.random.randn(N, 32).astype(np.float32)
    latent = (0.35 * X_sim[:, 0]) + (0.20 * X_sim[:, 10]) + (0.25 * X_sim[:, 16]) + (0.20 * X_sim[:, 18]) + (0.15 * X_sim[:, 28])
    y_sim = np.clip(3.20 + (((latent - np.mean(latent)) / np.std(latent)) * 0.95), 1.00, 6.00)
    train_and_save_xgboost_head(X_sim, y_sim)
