import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page ki settings aur Title
st.set_page_config(page_title="Exam Pass Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Future Exam Pass/Fail Predictor")
st.write("Apne parhai ke ghante aur attendance enter karein aur check karein ke aap exam mein pass honge ya nahi!")

# 2. Saved Model ko Load karna (.pkl file aap ke folder mein honi chahiye)
@st.cache_resource  # Is se model baar baar load nahi hoga aur app fast chalegi
def load_model():
    with open('exam_predictor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    
    # 3. User se Input lene ke liye Sliders aur Boxes
    st.header("📝 Apni Details Enter Karein:")
    
    study_hours = st.slider("Rozana Parhai ke Ghante (Study Hours):", min_value=0, max_value=16, value=6, step=1)
    attendance = st.slider("School/College ki Attendance (%):", min_value=0, max_value=100, value=80, step=1)
    prev_score = st.number_input("Pichle Exam ke Marks (Out of 100):", min_value=0, max_value=100, value=65)

    # 4. Predict Button aur Result
    if st.button("Predict My Result! 🎉"):
        # Input ko model ke format mein convert karna
        user_data = np.array([[study_hours, attendance, prev_score]])
        
        # Prediction aur Probability nikalna
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1] * 100
        
        st.write("---")
        st.subheader("🔮 Model Ka Faisla:")
        
        if prediction == 1 or prediction == 'Pass':
            st.success(f"**Mubarak Ho! Aap PASS ho jayenge. 🎉**")
            st.info(f"Pass hone ke chances: **{probability:.2f}%**")
        else:
            st.error(f"**⚠️ Khatra! Aap ko mazeed mehnat ki zaroorat hy (FAIL hone ka chance hy).**")
            st.info(f"Pass hone ke chances sirf: **{probability:.2f}%**")

except FileNotFoundError:
    st.error("❌ Error: 'exam_predictor_model.pkl' file nahi mili! Pehle apni model file ko isi folder mein copy-paste karein jahan app.py maujood hy.")
