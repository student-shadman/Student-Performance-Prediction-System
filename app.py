import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Haldia Institute of Technology - Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

pipeline = joblib.load("student_pipeline.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.image("hit_logo.png", use_container_width=True)
st.sidebar.title("Haldia Institute of Technology")
st.sidebar.markdown("### 🎓 Student Performance Prediction")

menu = st.sidebar.radio(
    "Navigation",
    ["🎓 Performance Prediction", "👨‍💻 Creators"]
)

theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

# ---------------- THEME STYLING ----------------
if theme:
    st.markdown("""
        <style>
        body {background-color: #0E1117;}
        .card {
            background: #1c1f26;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .card {
            background: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------- MAIN PAGE ----------------
if menu == "🎓 Performance Prediction":

    # Header with logo + title
    col_logo, col_title = st.columns([1, 3])

    with col_logo:
        st.image("hit_logo.png", width=120)

    with col_title:
        st.markdown("<h1>Haldia Institute of Technology</h1>", unsafe_allow_html=True)
        st.markdown("<h3>🎓 Student Performance Prediction System</h3>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([2, 1])

    # ---------------- LEFT SIDE FORM ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(" Student Details")

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 15, 22, 17)
            studytime = st.number_input("Study Time (1-4)", 1, 4, 2)
            failures = st.number_input("Past Failures", 0, 4, 0)
            absences = st.number_input("Absences", 0, 100, 5)
            G1 = st.number_input("G1 Marks", 0, 20, 10)

        with c2:
            G2 = st.number_input("G2 Marks", 0, 20, 10)
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Gender", ["M", "F"])
            address = st.selectbox("Address", ["U", "R"])
            famsize = st.selectbox("Family Size", ["GT3", "LE3"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(" Predict Performance", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT SIDE RESULT PANEL ----------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(" Prediction Result")

        if predict_btn:

            default_values = {
                "Pstatus": "T",
                "Medu": 2,
                "Fedu": 2,
                "Mjob": "other",
                "Fjob": "other",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 2,
                "famsup": "yes",
                "schoolsup": "no",
                "paid": "no",
                "activities": "no",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes",
                "romantic": "no",
                "famrel": 3,
                "freetime": 3,
                "goout": 3,
                "Dalc": 1,
                "Walc": 1,
                "health": 3,
            }

            user_input = {
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "G1": G1,
                "G2": G2,
                "school": school,
                "sex": sex,
                "address": address,
                "famsize": famsize,
            }

            final_input = {**default_values, **user_input}
            input_df = pd.DataFrame([final_input])

            prediction = pipeline.predict(input_df)
            probabilities = pipeline.predict_proba(input_df)[0]

            fail_prob = probabilities[0]
            pass_prob = probabilities[1]

            if prediction[0] == 1:
                st.success(f"✅ PASS\nConfidence: {pass_prob*100:.2f}%")
            else:
                st.error(f"❌ FAIL\nConfidence: {fail_prob*100:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(["Fail", "Pass"], [fail_prob, pass_prob])
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        else:
            st.info("Submit details to view prediction")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CREATORS PAGE ----------------
if menu == "👨‍💻 Creators":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👨‍💻 Project Team")
    st.write("Haldia Institute of Technology – ML Project")
    st.write("• Sarim Sajjad")
    st.write("• Manika Prasad")
    st.write("• Md Shadman")
    #st.write("• Member 4")
    st.markdown("</div>", unsafe_allow_html=True)
