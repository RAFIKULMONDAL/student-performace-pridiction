from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "student_performance_model.joblib"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction System")
st.write("Enter student study and lifestyle details to estimate a final mathematics grade.")
st.info("This is an educational estimate, not a guarantee of actual exam results.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
st.caption(f"Model used: {bundle.get('best_model_name', 'Trained model')}")

with st.form("student_details"):
    st.subheader("Academic information")
    age = st.number_input("Student age", 15, 22, 17)
    studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
        format_func=lambda n: {1:"Less than 2 hours", 2:"2–5 hours",
                               3:"5–10 hours", 4:"More than 10 hours"}[n])
    failures = st.selectbox("Previous class failures", [0, 1, 2, 3],
        format_func=lambda n: "None" if n == 0 else str(n))
    absences = st.number_input("Number of school absences", 0, 100, 4)

    st.subheader("Support and activities")
    famsup = st.selectbox("Family educational support?", ["yes", "no"])
    schoolsup = st.selectbox("Extra educational support at school?", ["yes", "no"])
    internet = st.selectbox("Internet access at home?", ["yes", "no"])
    activities = st.selectbox("Extracurricular activities?", ["yes", "no"])

    st.subheader("Lifestyle information")
    goout = st.select_slider("How often do you go out with friends?", [1, 2, 3, 4, 5],
        value=3, format_func=lambda n: {
            1:"Very rarely", 2:"Rarely", 3:"Sometimes", 4:"Often", 5:"Very often"}[n])
    dalc = st.select_slider("Weekday alcohol-consumption scale used in dataset",
        [1, 2, 3, 4, 5], value=1,
        format_func=lambda n: {1:"1 — Very low/none", 2:"2", 3:"3", 4:"4", 5:"5 — Very high"}[n])
    health = st.select_slider("Self-reported health (1 = very poor, 5 = very good)",
                              [1, 2, 3, 4, 5], value=4)
    submit = st.form_submit_button("Predict final grade", type="primary")

if submit:
    row = pd.DataFrame([{
        "age": age, "studytime": studytime, "failures": failures,
        "absences": absences, "famsup": famsup, "schoolsup": schoolsup,
        "internet": internet, "activities": activities, "goout": goout,
        "Dalc": dalc, "health": health
    }])
    grade = max(0.0, min(20.0, float(model.predict(row)[0])))
    percentage = grade * 5
    if grade < 10:
        category, advice = "Needs improvement", "Try a consistent study schedule, practise questions, and ask for help when needed."
    elif grade < 13:
        category, advice = "Average", "Revise regularly and practise questions from previous tests."
    elif grade < 16:
        category, advice = "Good", "Keep your routine consistent and focus on difficult topics."
    else:
        category, advice = "Excellent", "Maintain your routine and continue practising challenging questions."

    st.divider()
    st.subheader("Prediction result")
    c1, c2 = st.columns(2)
    c1.metric("Estimated grade", f"{grade:.1f} / 20")
    c2.metric("Approx. percentage", f"{percentage:.1f}%")
    st.success(f"Estimated performance category: **{category}**")
    st.write(advice)
    st.caption("Percentage is grade × 5. The category bands are illustrative, not official grading rules.")

with st.expander("About the model and its limitations"):
    st.write("The model uses 11 selected fields from the UCI Student Performance dataset. G1 and G2 are excluded.")
    st.write("The dataset is relatively small and represents students from two Portuguese schools. Results may not generalize to every college or curriculum.")
