import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import os
from flask_cors import CORS


# Add the following line to allow CORS
from flask import Flask
app = Flask(__name__)
CORS(app)

# Load the model
model_filename = 'workfile'
with open(model_filename, 'rb') as f:
    copy_of_model = pickle.load(f)

# Create a Streamlit web app
def main():
    st.title('Sales Prediction to Reduce Waste')

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Load the data from the uploaded file
        data = pd.read_csv(uploaded_file)

        # Select relevant columns
        X = data[["Inflation", "Disposable Income", "Month of the Year"]]
        y = data["Sold"]

        # Normalize the data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_normalized, y)
        st.write('Model trained successfully!')

        # Ask the user for input variables
        inflation = st.number_input('Inflation in Percents:', min_value=0.0)
        disposable_income = st.number_input('Disposable Income:')
        month_of_year = st.number_input('Month of the Year (1-12):', min_value=1, max_value=12)

        if st.button('Predict Sales'):
            # Make predictions based on user inputs
            x_input = {"Inflation": [inflation / 100], "Disposable Income": [disposable_income], "Month of the Year": [month_of_year]}
            predict = pd.DataFrame.from_dict(x_input)
            predicted_sales = copy_of_model.predict(predict)

            # Calculate other values as in your original code
            pre_dict = {"Raw Material (KGs)": [], "Packaging Material (KGs)": [], "Energy Consumption (kWh)": [],
                        "Waste in Transportation (Number of Bags)": [], "Predicted Sale (Number of Bags)": []}

            pre_dict["Raw Material (KGs)"].append(int(int(predicted_sales) * 0.142))
            pre_dict["Packaging Material (KGs)"].append(int(int(predicted_sales) * 0.028))
            pre_dict["Energy Consumption (kWh)"].append(int(predicted_sales) * 0.3)
            pre_dict["Waste in Transportation (Number of Bags)"].append(int(int(predicted_sales) * 0.025))
            pre_dict["Predicted Sale (Number of Bags)"].append(int(predicted_sales))

            pre_data = pd.DataFrame.from_dict(pre_dict)

            # Display the results in a DataFrame-like format
            st.subheader('Predicted Results')
            st.dataframe(pre_data.T, width=600)  # Transpose for a row-like display

if __name__ == "__main__":
    main()
