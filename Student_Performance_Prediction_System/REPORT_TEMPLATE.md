# Student Performance Prediction System Using Machine Learning

## 1. Aim
To build a machine-learning system that estimates final mathematics grades from selected student and study-habit information.

## 2. Problem Statement
The project explores whether selected study habits and student circumstances can be used to estimate academic performance. Regression algorithms are trained on the UCI Student Performance dataset.

## 3. Dataset
The project uses `student-mat.csv` from the UCI Student Performance dataset. `G3`, the final mathematics grade on a 0–20 scale, is the target. Input fields include age, weekly study time, previous failures, absences, family support, school support, internet access, activities, time spent going out, weekday alcohol-consumption scale, and health.

## 4. Preprocessing
Numerical missing values are imputed using the median and categorical missing values using the most frequent category. Categorical values are one-hot encoded. Data is split into 80% training and 20% testing with a fixed random seed. `G1` and `G2` are excluded because the application is designed to use the questions shown in its form.

## 5. Algorithms
- Linear Regression: baseline regression model.
- Decision Tree Regressor: learns non-linear decision rules.
- Random Forest Regressor: combines multiple decision trees.
- Tuned Random Forest: tests parameter combinations using cross-validation on training data.

## 6. Results
Run `python train_model.py` and copy the actual values from `outputs/evaluation_results.json`. Do not invent results.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | [actual result] | [actual result] | [actual result] |
| Decision Tree | [actual result] | [actual result] | [actual result] |
| Random Forest | [actual result] | [actual result] | [actual result] |
| Tuned Random Forest | [actual result] | [actual result] | [actual result] |

## 7. Application
The Streamlit interface collects the same 11 features used during training and displays a predicted grade, approximate percentage, and project-defined performance category.

## 8. Limitations
The dataset is small and represents students from two Portuguese schools. Its patterns may not apply to all colleges or curricula. A prediction is not a guarantee, and the displayed category bands are illustrative rather than official grading rules.

## 9. Conclusion
This project demonstrates data preparation, preprocessing, model training, comparison, tuning, evaluation, and deployment through a simple web interface. The actual test metrics should be used to discuss how well the model performs.
