import streamlit as st
import pandas as pd
import pickle

# Load model
with open("diamond_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💎 Diamond Price Prediction")

carat = st.number_input("Carat", 0.1, 5.0, 1.0)
depth = st.number_input("Depth", value=60.0)
table = st.number_input("Table", value=55.0)
x = st.number_input("Length (x)", value=5.0)
y = st.number_input("Width (y)", value=5.0)
z = st.number_input("Depth (z)", value=3.0)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

if st.button("Predict"):

    input_df = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity]
    })


    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Price: ${prediction:,.2f}")