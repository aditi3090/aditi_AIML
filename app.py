import streamlit as st
import numpy as np
import joblib

# ===== LOAD MODEL =====
model = joblib.load('placement__model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Student Placement Risk & Readiness Analysis System")

# ===== INPUT FIELDS =====
cgpa = st.number_input("CGPA")
tenth = st.number_input("10th Percentage")
twelfth = st.number_input("12th Percentage")
backlogs = st.number_input("Backlogs")

study_hours = st.number_input("Study Hours per Day")
attendance = st.number_input("Attendance Percentage")

projects = st.number_input("Projects Completed")
internships = st.number_input("Internships Completed")

coding = st.number_input("Coding Skill Rating (1-5)")
communication = st.number_input("Communication Skill Rating (1-5)")
aptitude = st.number_input("Aptitude Skill Rating (1-5)")

hackathons = st.number_input("Hackathons Participated")
certifications = st.number_input("Certifications Count")

stress = st.number_input("Stress Level (1-10)")

# ===== CATEGORICAL INPUTS =====
gender = st.selectbox("Gender", ["Male", "Female"])
branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
part_time = st.selectbox("Part Time Job", ["Yes", "No"])
income = st.selectbox("Family Income", ["Low", "Medium", "High"])
city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
extra = st.selectbox("Extracurricular Involvement", ["Yes", "No"])

# ===== ENCODING =====
gender = 1 if gender == "Male" else 0
part_time = 1 if part_time == "Yes" else 0
extra = 1 if extra == "Yes" else 0

income_map = {"Low": 0, "Medium": 1, "High": 2}
income = income_map[income]

city_map = {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}
city = city_map[city]

branch_map = {"CSE": 0, "IT": 1, "ECE": 2, "Mechanical": 3, "Civil": 4}
branch = branch_map[branch]

# ===== ANALYSIS FUNCTION (FIXED) =====
def student_analysis(prob):
    
    if prob >= 0.7:
        placement = "Placed (High Probability)"
        readiness = "High"
        risk = "Low Risk"
        msg = "Great job! You are highly placement ready."

    elif prob >= 0.4:
        placement = "Borderline Case"
        readiness = "Medium"
        risk = "Medium Risk"
        msg = "Good, but improvement is needed."

    else:
        placement = "Not Placed (Low Probability)"
        readiness = "Low"
        risk = "High Risk"
        msg = "Focus on improving your skills."

    return placement, readiness, risk, msg

# ===== PREDICTION =====
if st.button("Predict"):

    data = np.array([[
        gender, branch, cgpa, tenth, twelfth, backlogs,
        study_hours, attendance, projects, internships,
        coding, communication, aptitude,
        hackathons, certifications,
        stress, part_time, income, city,
        extra
    ]])

    # scale input
    data = scaler.transform(data)

    # ===== FIX: PROBABILITY BASED SYSTEM =====
    prob = model.predict_proba(data)[0][1]

    placement, readiness, risk, msg = student_analysis(prob)

    # ===== DISPLAY =====
    st.success(f"Prediction: {placement}")
    st.info(f"Risk Level: {risk}")
    st.warning(f"Readiness Level: {readiness}")

    st.markdown("### Analysis")
    st.write(msg)

    st.markdown("### Confidence Score")
    st.progress(float(prob))
    st.write(f"Placement Probability: {prob:.2f}")
