import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


model = load_model('ann_model.h5')
with open('onehotencoder_geograph.pkl', 'rb') as f:
    onehotencoder_geograph = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

st.title("Customer Churn Prediction")

st.header("Enter Customer Details:")

# Define input widgets for each feature
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", value=100000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])

if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography]
        })

    # Apply label encoding to Gender
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

    # Apply one-hot encoding to Geography
    onehot_encoded_geo = onehotencoder_geograph.transform(input_data[['Geography']])
    onehot_encoded_geo_df = pd.DataFrame(onehot_encoded_geo.toarray(), columns=onehotencoder_geograph.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.drop('Geography', axis=1), onehot_encoded_geo_df], axis=1)

    # Ensure all columns are in the correct order and format as during training
    # This step is crucial to match the columns of the training data
    # You might need to manually specify the column order based on your X_train
    # For example:
    X_train_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
    input_data = input_data[X_train_columns]


    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)
    churn_probability = prediction[0][0]

    # Set a threshold and display the result
    threshold = 0.5
    if churn_probability > threshold:
        st.error(f"Prediction: The customer is likely to churn with a probability of {churn_probability:.2f}")
    else:
        st.success(f"Prediction: The customer is not likely to churn with a probability of {churn_probability:.2f}")