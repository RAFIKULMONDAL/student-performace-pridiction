from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student-mat.csv"
OUT = ROOT / "outputs"
ART = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)

# These 11 columns match the questions shown in app.py.
FEATURES = ["age", "studytime", "failures", "absences", "famsup",
            "schoolsup", "internet", "activities", "goout", "Dalc", "health"]
NUMERIC = ["age", "studytime", "failures", "absences", "goout", "Dalc", "health"]
CATEGORICAL = ["famsup", "schoolsup", "internet", "activities"]
TARGET = "G3"

def make_pipeline(estimator):
    prep = ColumnTransformer([
        ("numbers", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUMERIC),
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    return Pipeline([("prep", prep), ("model", estimator)])

def metrics(model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
        "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 4),
        "R2": round(float(r2_score(y_test, pred)), 4)
    }, pred

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\\n"
            "Download the UCI Student Performance dataset and copy student-mat.csv into data."
        )

    df = pd.read_csv(DATA, sep=";")
    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[needed].dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Rows: {len(df)} | Inputs: {len(FEATURES)} | Missing input cells: {int(X.isna().sum().sum())}")
    print(f"Training records: {len(x_train)} | Test records: {len(x_test)}")
    print("Target: G3, final mathematics grade (0-20)")

    models = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Decision Tree": make_pipeline(DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=4, random_state=42)),
        "Random Forest": make_pipeline(RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1))
    }
    cv_mae = {}
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=5,
                                 scoring="neg_mean_absolute_error")
        cv_mae[name] = round(float(-scores.mean()), 4)

    tuned = GridSearchCV(
        make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1)),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 8, None],
            "model__min_samples_leaf": [2, 4]
        },
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    tuned.fit(x_train, y_train)
    models["Tuned Random Forest"] = tuned.best_estimator_
    cv_mae["Tuned Random Forest"] = round(float(-tuned.best_score_), 4)

    results, predictions = {}, {}
    for name, model in models.items():
        if name != "Tuned Random Forest":
            model.fit(x_train, y_train)
        results[name], predictions[name] = metrics(model, x_test, y_test)
        print(f"{name}: {results[name]} | training CV MAE: {cv_mae[name]}")

    # Select the final model using cross-validation, not test-set performance.
    best_name = min(cv_mae, key=cv_mae.get)
    best_model = models[best_name]
    best_model.fit(x_train, y_train)
    joblib.dump({
        "model": best_model,
        "features": FEATURES,
        "best_model_name": best_name
    }, ART / "student_performance_model.joblib")

    (OUT / "evaluation_results.json").write_text(json.dumps({
        "dataset_rows": len(df),
        "training_records": len(x_train),
        "test_records": len(x_test),
        "target": "G3 (final mathematics grade, 0-20)",
        "selected_model": best_name,
        "training_cross_validation_mae": cv_mae,
        "test_metrics": results,
        "best_random_forest_parameters": tuned.best_params_
    }, indent=2), encoding="utf-8")

    names = list(results)
    plt.figure(figsize=(9, 5))
    plt.bar(names, [results[n]["MAE"] for n in names])
    plt.ylabel("MAE (lower is better)")
    plt.title("Model Comparison on Test Data")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close()

    final_pred = best_model.predict(x_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, final_pred, alpha=0.75)
    plt.plot([0, 20], [0, 20], "--")
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted: {best_name}")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
    plt.close()
    pd.DataFrame({"actual_G3": y_test, "predicted_G3": final_pred.round(2)}).to_csv(
        OUT / "test_predictions.csv", index=False
    )
    print(f"Selected model: {best_name}")
    print("Saved model, metrics, plots, and test predictions.")

if __name__ == "__main__":
    main()
