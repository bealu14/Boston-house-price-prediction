import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


model_path = "boston_house_price_predictor.pkl"
model = pickle.load(open(model_path, "rb"))

def predict_house_price(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]


def main():
    
    st.title("Boston House Price Predictor")

    st.write("Enter the features to predict the house price:")
    st.title("Keys")
    
    st.write("INDUS (Proportion of Non-Retail Business Acres):")
    st.write("RM (Average number of Rooms)")
    st.write("AGE (Proportion of owner-occupied units built before 1940)")
    st.write("DIS (Weighted Distances to Boston Employment Centres)")
    st.write("TAX (Property Tax Rate per $10,000)")

 
    
    feature_names = ['INDUS','RM','AGE','DIS','TAX']
    features = []
    for feature in feature_names:
        value = st.number_input(feature, step=0.1)
        features.append(value)

    if st.button("Predict"):
        prediction = predict_house_price(features)
        final_prediction = prediction * 10000
        st.success(f"The predicted house price is: ${final_prediction :.2f}")

if __name__ == "__main__":
    main()
