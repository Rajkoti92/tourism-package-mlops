"""Model Training: train, tune, evaluate, log with MLflow, and save the best model."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)
import joblib
import mlflow

# --- Load the train and test splits ---
train_df = pd.read_csv("tourism_project/data/train.csv")
test_df = pd.read_csv("tourism_project/data/test.csv")

TARGET = "ProdTaken"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Identify column types
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print(f"Categorical: {cat_cols}")
print(f"Numerical:   {num_cols}")

# --- Preprocessing pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), cat_cols),
    ]
)


def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    """Train and evaluate a pipeline, returning a metrics dict."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_te, y_proba)
    return metrics


# --- Baseline model comparison ---
models = {
    "Decision Tree": Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))]),
    "Random Forest": Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("pre", preprocessor), ("clf", GradientBoostingClassifier(random_state=42))]),
    "XGBoost": Pipeline([("pre", preprocessor), ("clf", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"))]),
}

mlflow.set_experiment("tourism-package-prediction")

results = {}
for name, pipe in models.items():
    with mlflow.start_run(run_name=f"baseline-{name}"):
        metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test)
        results[name] = metrics
        mlflow.log_params({"model": name, "stage": "baseline"})
        mlflow.log_metrics(metrics)
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(classification_report(y_test, pipe.predict(X_test)))

# Pick the best baseline model by F1 score
results_df = pd.DataFrame(results).T
print("\n=== Baseline Model Comparison ===")
print(results_df.round(4))
best_baseline = results_df["f1"].idxmax()
print(f"\nBest baseline model by F1: {best_baseline}")


# --- Hyperparameter Tuning (top 2 models: Random Forest and XGBoost) ---
print("\n=== Hyperparameter Tuning ===")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Tune Random Forest
rf_param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_leaf": [2, 5],
}

rf_grid = GridSearchCV(
    Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)
rf_grid.fit(X_train, y_train)
print(f"\nBest RF params: {rf_grid.best_params_}")
print(f"Best RF CV F1: {rf_grid.best_score_:.4f}")

with mlflow.start_run(run_name="tuned-random-forest"):
    rf_metrics = evaluate_model(rf_grid.best_estimator_, X_train, y_train, X_test, y_test)
    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_params({"model": "Random Forest", "stage": "tuned"})
    mlflow.log_metrics(rf_metrics)
    print("Tuned Random Forest:")
    for k, v in rf_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, rf_grid.best_estimator_.predict(X_test)))


# Tune XGBoost
xgb_param_grid = {
    "clf__n_estimators": [150, 300],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
}

xgb_grid = GridSearchCV(
    Pipeline([("pre", preprocessor), ("clf", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"))]),
    param_grid=xgb_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)
xgb_grid.fit(X_train, y_train)
print(f"\nBest XGBoost params: {xgb_grid.best_params_}")
print(f"Best XGBoost CV F1: {xgb_grid.best_score_:.4f}")

with mlflow.start_run(run_name="tuned-xgboost"):
    xgb_metrics = evaluate_model(xgb_grid.best_estimator_, X_train, y_train, X_test, y_test)
    mlflow.log_params(xgb_grid.best_params_)
    mlflow.log_params({"model": "XGBoost", "stage": "tuned"})
    mlflow.log_metrics(xgb_metrics)
    print("Tuned XGBoost:")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, xgb_grid.best_estimator_.predict(X_test)))


# --- Select and save the best overall model ---
all_tuned = {
    "Tuned Random Forest": (rf_grid.best_estimator_, rf_metrics),
    "Tuned XGBoost": (xgb_grid.best_estimator_, xgb_metrics),
}

best_name = max(all_tuned, key=lambda k: all_tuned[k][1]["f1"])
best_model = all_tuned[best_name][0]
best_metrics = all_tuned[best_name][1]

print(f"\n=== Final Model: {best_name} ===")
for k, v in best_metrics.items():
    print(f"  {k}: {v:.4f}")

# Save the model for the deployment step
MODEL_PATH = "tourism_project/deployment/tourism_model.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
