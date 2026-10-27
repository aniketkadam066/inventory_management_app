import streamlit as st
import pandas as pd
import numpy as np
import pickle

# page title
st.title("Inventory Management - Stock Risk Prediction")
st.write("This app predicts whether an inventory item is at risk of Stockout or Overstock.")

# loading the saved model and scaler
model = pickle.load(open('inventory_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.subheader("Enter Item Details:")

# taking input from the user 
current_stock = st.number_input("Current Stock (Units)", min_value=0, value=500)
avg_daily_demand = st.number_input("Average Daily Demand (Units/Day)", min_value=0.0, value=50.0)
lead_time_days = st.number_input("Lead Time (Days)", min_value=0.0, value=7.0)
reorder_point = st.number_input("Reorder Point (Units)", min_value=0.0, value=350.0)
sales_last_30_days = st.number_input("Sales Last 30 Days", min_value=0.0, value=1500.0)
stock_turnover_ratio = st.number_input("Stock Turnover Ratio", min_value=0.0, value=4.0)
forecast_error = st.number_input("Forecast Error", value=0.0)

# dropdown options
season = st.selectbox("Season", ["peak", "off-peak"])
item_category = st.selectbox("Item Category", ["essential", "non-essential"])
supplier_reliability = st.selectbox("Supplier Reliability", ["high", "low"])

# encoding (Converting verbal information into numerical form)
season_val = 1 if season == "peak" else 0
category_val = 1 if item_category == "essential" else 0
reliability_val = 1 if supplier_reliability == "high" else 0

# button for making a prediction
if st.button("Predict Stock Status"):
    
   # creating a dataframe from the input data
    input_data = pd.DataFrame([[
        current_stock, avg_daily_demand, lead_time_days, reorder_point,
        sales_last_30_days, stock_turnover_ratio, forecast_error,
        season_val, category_val, reliability_val
    ]], columns=[
        'current_stock', 'avg_daily_demand', 'lead_time_days', 'reorder_point',
        'sales_last_30_days', 'stock_turnover_ratio', 'forecast_error',
        'season', 'item_category', 'supplier_reliability'
    ])
    
    # scaling data
    input_scaled = scaler.transform(input_data)
    
    # making a prediction
    prediction = model.predict(input_scaled)
    
   # displaying the result on the screen
    st.subheader("Result:")
    
    if prediction[0] == 1:
        st.warning("The item is at risk of **OVERSTOCK**! (Excess stock available)")
    else:
        st.error("The item is at risk of **STOCKOUT**! (Low stock, immediate reorder needed)")