import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

st.title("Customer Churn Prediction App")
st.write(
    "Bu uygulama, demografik ve finansal özelliklere dayanarak bir müşterinin bankayı terk edip etmeyeceğini (churn) TensorFlow ve Keras tabanlı Yapay Sinir Ağı (ANN) modeli ile tahmin eder."
)

@st.cache_resource
def load_nn_model():
    return load_model("churn_ann_model.h5")

model = load_nn_model()

st.subheader("Musteri Bilgilerini Giriniz:")
credit_score = st.number_input("Kredi Skoru (CreditScore)", min_value=300, max_value=850, value=600)
geography = st.selectbox("Ulke (Geography)", ["France", "Spain", "Germany"])
gender = st.selectbox("Cinsiyet (Gender)", ["Female", "Male"])
age = st.number_input("Yas (Age)", min_value=18, max_value=100, value=35)
tenure = st.number_input("Banka ile Calisma Suresi (Tenure)", min_value=0, max_value=10, value=3)
balance = st.number_input("Bakiye (Balance)", min_value=0.0, max_value=250000.0, value=50000.0)
num_of_products = st.number_input("Urun Sayisi (NumOfProducts)", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Kredi Karti Var mi? (HasCrCard)", [0, 1])
is_active_member = st.selectbox("Aktif Uye mi? (IsActiveMember)", [0, 1])
estimated_salary = st.number_input("Tahmini Maas (EstimatedSalary)", min_value=0.0, max_value=200000.0, value=50000.0)

if st.button("Musteri Durumunu Tahmin Et", type="primary"):
    try:
        geo_map = {"France": 0, "Germany": 1, "Spain": 2}
        gen_map = {"Female": 0, "Male": 1}
        
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography": [geo_map[geography]],
            "Gender": [gen_map[gender]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })
        
        prediction = model.predict(input_data)
        churn_prob = prediction[0][0]
        
        if churn_prob > 0.5:
            st.error(f"Tahmin Sonucu: Bu müşterinin bankayı terk etme olasılığı yüksektir (Olasılık: {churn_prob:.2f}).")
        else:
            st.success(f"Tahmin Sonucu: Bu müşteri bankada kalmaya devam edecektir (Olasılık: {churn_prob:.2f}).")
    except Exception as e:
        st.error(f"Tahmin sirasinda bir hata olustu: {e}")