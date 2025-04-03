import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# Load model and encoders
model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title and UI
st.title("📉 Customer Churn Prediction")
st.markdown("Predict the probability of a customer churning based on their profile.")

# Inputs
st.header("🧍 Customer Info")
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
with col2:
    credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, value=650)
    balance = st.number_input('💰 Balance', value=0.0)
    estimated_salary = st.number_input('💼 Estimated Salary', value=50000.0)
    has_cr_card = st.selectbox('🏦 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
churn_proba = prediction[0][0]

# Display result
st.header("🔍 Prediction Result")
st.write(f"**Churn Probability:** `{churn_proba:.2f}`")

# Visualization: Gauge Chart
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=churn_proba,
    delta={'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 0.5], 'color': 'lightgreen'},
            {'range': [0.5, 1], 'color': 'lightcoral'}
        ],
    },
    title={'text': "Churn Probability"}
))
st.plotly_chart(fig_gauge)

# Result text
if churn_proba > 0.5:
    st.error("🚨 The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")

# Visualization: Bar Chart of inputs
st.header("📊 Customer Feature Overview")
input_features = {
    'Credit Score': credit_score,
    'Age': age,
    'Balance': balance,
    'Estimated Salary': estimated_salary,
    'Tenure': tenure,
    'Num of Products': num_of_products
}
fig, ax = plt.subplots()
ax.barh(list(input_features.keys()), list(input_features.values()), color='skyblue')
ax.set_xlabel("Value")
ax.set_title("Selected Customer Features")
st.pyplot(fig)

# Visualization: Radar Chart
st.header("🕸️ Normalized Feature Radar")
selected = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
normalized_values = input_scaled[0]
normalized_dict = {feature: normalized_values[input_data.columns.get_loc(feature)] for feature in selected}

fig_radar = go.Figure(data=go.Scatterpolar(
    r=list(normalized_dict.values()),
    theta=list(normalized_dict.keys()),
    fill='toself',
    name='Customer Profile',
    line_color='royalblue'
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    title="Customer Profile (Normalized)"
)
st.plotly_chart(fig_radar)
