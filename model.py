import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
import requests

# Initialize the model variable
model = None

# Construct the raw GitHub URL
github_raw_url = "https://github.com/JacksonDivakar/Gender-Classification/raw/main/trainedmodel%20(7).pkl"

try:
    # Download the pickle file content using requests
    response = requests.get(github_raw_url)

    if response.status_code == 200:
        # Load the data from the pickle file
        model = pk.loads(response.content)
    else:
        st.error(f"Failed to retrieve pickle file. Status code: {response.status_code}")
except Exception as e:
    st.error(f"Error reading pickle file: {e}")

# Function to classify face
def classify_face(forehead_width, forehead_height, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long):
    if model is not None:
        # Create a DataFrame from the input values
        df = pd.DataFrame({
            'forehead_width_cm': [forehead_width],
            'forehead_height_cm': [forehead_height],
            'nose_wide': [nose_wide],
            'nose_long': [nose_long],
            'lips_thin': [lips_thin],
            'distance_nose_to_lip_long': [distance_nose_to_lip_long]
        })

        # Make prediction
        prediction = model.predict(df)[0]

        # Return the predicted gender
        return 'Male' if prediction == 0 else 'Female'
    else:
        st.warning("Model not loaded. Please check if the model was successfully loaded.")

# Streamlit app
st.title("Gender Classification App")

# Get user input using choice buttons
forehead_width = st.slider("Select forehead width (cm):", 11.4, 15.5, 13.2, step=0.1)
forehead_height = st.slider("Select forehead height (cm):", 5.1, 7.1, 5.9, step=0.1)
user_input1 = st.radio("Is the nose wide?", options=["Yes", "No"])
nose_wide =  1 if user_input1 == "Yes" else 0
user_input2 = st.radio("Is the nose long?", options=["Yes", "No"])
nose_long =  1 if user_input2 == "Yes" else 0
user_input3 = st.radio("Are the lips thin?", options=["Yes", "No"])
lips_thin =  1 if user_input3 == "Yes" else 0
user_input4 = st.radio("Is the distance from nose to lip long?", options=["Yes", "No"])
distance_nose_to_lip_long =  1 if user_input4 == "Yes" else 0

# Classify face and display result
if st.button("Classify"):
    result = classify_face(forehead_width, forehead_height, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long)
    st.write(f"The predicted gender is: {result}")
