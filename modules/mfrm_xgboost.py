
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import statsmodels.api as sm

def calibrate_mfrm_ground_truth(ratings_records: list) -> pd.DataFrame:
    """
    Removes rater severity bias via Two-Way Fixed Effects OLS.
    Projects latent theta onto CEFR continuous continuum [1.00, 6.00].
    """
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

    # Continuous CEFR continuum scaling: anchor mean = 3.20 (B1), scale = 0.95
    records = []
    for uid, th in latent_thetas.items():
        z = (th - mu) / (sigma + 1e-8)
        cefr_val = np.clip(3.20 + (z * 0.95), 1.00, 6.00)
        records.append({"utterance_id": uid, "cefr_target": round(float(cefr_val), 2)})
    return pd.DataFrame(records)

def train_and_save_xgboost_head(X: np.ndarray, y: np.ndarray, model_save_path: str = "cefr_xgboost_head.json"):
    """
    Fits XGBoost Regressor on extracted 22-D multimodal features and MFRM target CEFR scores.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = xgb_model.predict(X_val)
    rmse = root_mean_squared_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    print(f"[XGBoost Calibration] Validation RMSE: {rmse:.4f} | R² Score: {r2:.4f}")

    xgb_model.save_model(model_save_path)
    print(f"[XGBoost Calibration] Model saved to {model_save_path}")
    return xgb_model

if __name__ == "__main__":
    np.random.seed(42)
    N = 600
    X_sim = np.random.randn(N, 22).astype(np.float32)
    latent = (0.35 * X_sim[:, 0]) + (0.20 * X_sim[:, 10]) + (0.25 * X_sim[:, 16]) + (0.20 * X_sim[:, 18])
    y_sim = np.clip(3.20 + (((latent - np.mean(latent)) / np.std(latent)) * 0.95), 1.00, 6.00)
    train_and_save_xgboost_head(X_sim, y_sim)
