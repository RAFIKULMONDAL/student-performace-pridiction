# Student Performance Prediction System

A third-year BE IT AIML mini-project that estimates a student's final mathematics grade using selected fields from the UCI Student Performance dataset.

## Included
- Streamlit form with 11 inputs that match the trained model
- Linear Regression, Decision Tree, and Random Forest comparison
- Random Forest hyperparameter tuning
- Evaluation using MAE, RMSE, and R²
- Test-set predictions and charts
- Final grade out of 20, converted percentage, and project-defined category

## Dataset setup
Download the official UCI Student Performance dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

Extract the archive and copy `student-mat.csv` to `data/student-mat.csv`.
The dataset's target is `G3` (final mathematics grade, 0–20). Earlier grades `G1` and `G2` are excluded.

## Run on Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If activation is blocked in PowerShell, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Outputs
- `artifacts/student_performance_model.joblib`
- `outputs/evaluation_results.json`
- `outputs/model_comparison.png`
- `outputs/actual_vs_predicted.png`
- `outputs/test_predictions.csv`

## Limitations
The dataset is relatively small and comes from two Portuguese schools. Predictions are estimates, not guarantees. The performance categories are illustrative bands for this project, not official grade boundaries.
