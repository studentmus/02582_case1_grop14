import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

from preprocessing import load_data, build_preprocessor

# 1. Loading raw data
X, y, X_new = load_data()

# dropping C_02 here instead of inside processing as build preprocessor does not return it
X = X.drop(columns=["C_02"])
X_new = X_new.drop(columns=["C_02"])

# 2. Building preprocessor
preprocessor = build_preprocessor(X)

# 3. Defining model and pipeline
model = ElasticNet(max_iter=50000, tol=1e-3, random_state=0) # added more iterations and a higher tolerance to handle the convergence warning
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# 4. Hyperparameter grid
param_grid = {
    "model__alpha": np.logspace(-3, 2, 10), 
    "model__l1_ratio": [0.1, 0.5, 0.9]
}

# 5. Nested CV
## inner CV: tunes hyperparameters
## outer CV: estimates generalization performance 

inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=1) # different seed to ensure different splits

# inner loop for hyperparameter tuning
inner_grid = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    scoring='neg_root_mean_squared_error', 
    cv=inner_cv
)

# outer loop for performance estimation
outer_scores = cross_val_score(
    estimator=inner_grid, 
    X=X, 
    y=y, 
    cv=outer_cv, 
    scoring='neg_root_mean_squared_error'
)

# nested cv RMSE
nested_rmse_per_fold = -outer_scores
nested_rmse_mean = nested_rmse_per_fold.mean()
nested_rmse_std = nested_rmse_per_fold.std()

print("Nested CV RMSE per fold:", nested_rmse_per_fold)
print(f"Nested CV RMSE: {nested_rmse_mean:.3f} ± {nested_rmse_std:.3f}")

# 6. Final model training on the entire dataset
final_grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=inner_cv
)

final_grid.fit(X, y)

inner_rmse = -final_grid.best_score_
best_params = final_grid.best_params_

print("Best hyperparameters:", best_params)
print("Inner CV RMSE:", inner_rmse)
print("Nested CV RMSE:", nested_rmse_mean)

# 7. Getting the best parameters and refitting the model on the data
best_pipeline = final_grid.best_estimator_
best_pipeline.fit(X, y)

# 8. Predicting on the new data
y_new_pred = best_pipeline.predict(X_new)

# 9. Saving predictions to CSV
np.savetxt("predictions/predictions_s225210_s223007.csv", y_new_pred, delimiter=",")

# 10. Saving the estimated RMSE
np.savetxt("predictions/estimatedRMSE_s225210_s223007.csv", [nested_rmse_mean], fmt="%.4f", delimiter=",")


