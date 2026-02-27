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
    import matplotlib.pyplot as plt
    import seaborn as sns

    X_processed, X_new_processed, y, preprocessor = get_preprocessed_data()
    X, y_raw, X_new = load_data()

    print("X_processed shape:", X_processed.shape)
    print("X_new_processed shape:", X_new_processed.shape)
    print("y shape:", y_raw.shape)
    print("No NaN in X:", pd.DataFrame(X_processed).isna().sum().sum() == 0)
    print("No NaN in X_new:", pd.DataFrame(X_new_processed).isna().sum().sum() == 0)

    # 1. Распределение y
    plt.figure(figsize=(7, 4))
    plt.hist(y_raw, bins=20, edgecolor="black")
    plt.title("Distribution of y (response variable)")
    plt.xlabel("y")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/y_distribution.png")
    plt.close()

    # 2. Heatmap пропусков
    num_cols = [c for c in X.columns if c.startswith("x_")]
    missing = X[num_cols].isna()
    plt.figure(figsize=(18, 4))
    sns.heatmap(missing.T, cbar=False, yticklabels=False, xticklabels=False)
    plt.title("Missing values in numeric features (yellow = missing)")
    plt.tight_layout()
    plt.savefig("plots/missing_heatmap.png")
    plt.close()

    # 3. Распределения категорий
    cat_cols = [c for c in X.columns if c.startswith("C_")]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(14, 3))
    for i, col in enumerate(cat_cols):
        X[col].value_counts().sort_index().plot(kind="bar", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.savefig("plots/category_distributions.png")
    plt.close()

    print("Plots saved to plots/")
