import streamlit as st
import numpy as np
import pickle

# Load model dan scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Prediksi Penyakit Diabetes")

preg = st.number_input("Jumlah Kehamilan", 0)
glu = st.number_input("Kadar Glukosa", 0)
bp = st.number_input("Tekanan Darah", 0)
skin = st.number_input("Ketebalan Kulit", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Umur", 0)

if st.button("Prediksi"):
    data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ Pasien Berpotensi Diabetes")
    else:
        st.success("✅ Pasien Tidak Diabetes")
