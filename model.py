import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np

# Load the trained model
model = pk.load(open("C:\\Users\\jacks\\Downloads\\trainedmodel.pkl", 'rb'))

# Function to classify face
def classify_face(forehead_width, forehead_height, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long):
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
    return 'Male' if prediction == 1 else 'Female'

# Streamlit app
st.title("Face Classification App")

# Get user input using choice buttons
forehead_width = st.slider("Select forehead width (cm):", 11.4, 15.5, 13.2, step=0.1)
forehead_height = st.slider("Select forehead height (cm):", 5.1, 7.1, 5.9, step=0.1)
nose_wide = st.radio("Is the nose wide?", [0, 1], index=1)
nose_long = st.radio("Is the nose long?", [0, 1], index=1)
lips_thin = st.radio("Are the lips thin?", [0, 1], index=1)
distance_nose_to_lip_long = st.radio("Is the distance from nose to lip long?", [0, 1], index=1)

# Classify face and display result
if st.button("Classify"):
    result = classify_face(forehead_width, forehead_height, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long)
    st.write(f"The predicted gender is: {result}")