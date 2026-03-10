import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

def evaluate_models(X, y, preprocessor):
    models = {
        "ElasticNet": Pipeline([
            ("preprocessor", preprocessor),
            ("model", ElasticNet(max_iter=10000))
        ]),

        "Ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge())
        ]),

        "PCA_ElasticNet": Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("model", ElasticNet(max_iter=10000))
        ]),

        "Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor())
        ])
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    results = {}

    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
        rmse_mean = np.sqrt(-scores.mean())
        rmse_std = np.sqrt(scores.std())
        results[name] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std
        }
        print(f"{name}: RMSE = {rmse_mean:.3f} ± {rmse_std:.3f}")

    return results

if __name__ == "__main__":
    from preprocessing import build_preprocessor, load_data

    X, y, X_new = load_data()
    

    preprocessor = build_preprocessor(X)
    results = evaluate_models(X, y, preprocessor)
