import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data():
    train = pd.read_csv("data/case1Data.csv")
    X = train.drop(columns=["y"])
    y = train["y"]
    X_new = pd.read_csv("data/case1Data_Xnew.csv")
    return X, y, X_new

def build_preprocessor(X):
    num_cols = [c for c in X.columns if c.startswith("x_")]
    cat_cols = [c for c in X.columns if c.startswith("C_")]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    return preprocessor


def get_preprocessed_data():
    X, y, X_new = load_data()

    
    X = X.drop(columns=["C_02"])
    X_new = X_new.drop(columns=["C_02"])

    preprocessor = build_preprocessor(X) 
    
    X_processed = preprocessor.fit_transform(X)
    
    X_new_processed = preprocessor.transform(X_new)

    return X_processed, X_new_processed, y, preprocessor

if __name__ == "__main__":
    X_processed, X_new_processed, y, preprocessor = get_preprocessed_data()
    print("X_processed shape:", X_processed.shape)
    print("X_new_processed shape:", X_new_processed.shape)
    print("y shape:", y.shape)
    print("No NaN in X:", pd.DataFrame(X_processed).isna().sum().sum() == 0)
