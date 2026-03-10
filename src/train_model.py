import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV

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

# 4. Hyperparameter tuning with GridSearchCV
param_grid = {
    "model__alpha": np.logspace(-3, 2, 10), # will reduce if it takes too long
    "model__l1_ratio": [0.1, 0.5, 0.9]
}

cv = KFold(n_splits=10, shuffle=True, random_state=0)
grid = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    scoring='neg_root_mean_squared_error', 
    cv=cv
)

# 5. Fitting the model on training data
grid.fit(X, y)

best_rmse = -grid.best_score_
best_params = grid.best_params_

print("Best hyperparameters:", best_params)
print("Estimated RMSE from CV:", best_rmse)

# 6. Getting the best parameters and refitting the model on the data
best_pipeline = grid.best_estimator_
best_pipeline.fit(X, y)

# 7. Predicting on the new data
y_new_pred = best_pipeline.predict(X_new)

# 8. Saving predictions to CSV
np.savetxt("predictions/predictions_s225210_s223007.csv", y_new_pred, delimiter=",")

# 9. Saving the estimated RMSE
np.savetxt("predictions/estimatedRMSE_s225210_s223007.csv", [best_rmse], fmt="%.4f", delimiter=",")


