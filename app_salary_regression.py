import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('utils/regression_model.h5')

## load the encoder and scaler
with open('utils/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('utils/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('utils/scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)
    
import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Customer Salary Prediction",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a title with a subtitle
st.title("💼 Customer Salary Prediction")
st.subheader("Estimate the potential salary of a customer based on their attributes.")

# Add a styled description
st.markdown("""
<style>
    .description {
        font-size: 16px;
        color: #555;
    }
</style>
<p class="description">
    Provide the customer's details below to calculate their estimated salary. Hover over the <b>?</b> icon for additional information about each field.
</p>
""", unsafe_allow_html=True)

# Create a two-column layout for better organization
col1, col2 = st.columns(2)

with col1:
    # Geography
    geography = st.selectbox(
        '🌍 Geography',
        onehot_encoder_geo.categories_[0],
        help="The country or region where the customer resides."
    )

    # Gender
    gender = st.selectbox(
        '👤 Gender',
        label_encoder_gender.classes_,
        help="The gender of the customer."
    )

    # Age
    age = st.slider(
        '🎂 Age',
        18, 92,
        help="The customer's age in years."
    )

    # Balance
    balance = st.number_input(
        '💰 Balance',
        help="The current balance in the customer's bank account. Currency: USD ($)."
    )

with col2:
    # Credit Score
    credit_score = st.number_input(
        '📊 Credit Score',
        help="A numerical representation of the customer's creditworthiness (300–850)."
    )

    # Exited
    exited = st.selectbox(
        '❌ Exited',
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
        help="Whether the customer has exited (closed their account) with the bank (1: Yes, 0: No)."
    )

    # Tenure
    tenure = st.slider(
        '📅 Tenure',
        0, 10,
        help="The number of years the customer has been with the bank."
    )

# Add additional inputs in a horizontal layout
st.markdown("### Additional Details")
num_of_products = st.slider(
    '🛍️ Number of Products',
    1, 4,
    help="The number of products the customer holds with the bank."
)

has_cr_card = st.selectbox(
    '💳 Has Credit Card',
    [0, 1],
    format_func=lambda x: "Yes" if x else "No",
    help="Whether the customer has a credit card (1: Yes, 0: No)."
)

is_active_member = st.selectbox(
    '🔗 Is Active Member',
    [0, 1],
    format_func=lambda x: "Yes" if x else "No",
    help="Whether the customer is an active member (1: Yes, 0: No)."
)

# Prepare the input data when the Predict button is clicked
if st.button("💡 Predict Salary"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict salary
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    # Display the prediction results
    st.write(f"### 🧮 Estimated Salary: ${predicted_salary:,.2f}")

    # Provide styled feedback
    if predicted_salary > 0:
        st.success("💼 Prediction successfully calculated!")
    else:
        st.error("⚠️ Unable to calculate a valid salary. Please check the inputs.")

# Add a footer with style
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 12px;'>
    Developed with ❤️ using Streamlit | © 2024 Your Name
</p>
""", unsafe_allow_html=True)


