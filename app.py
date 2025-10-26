import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

# Load the pre-trained model
with open("models/lightgbm.pkl", 'rb') as f:
    model = pickle.load(f)

# load the fitted encoder
encoder = joblib.load('models/onehot_encoder.pkl')

st.set_page_config(page_title="Doordash Delivery Time Prediction", page_icon='🚚', layout="wide")

st.title("🚚 Doordash Delivery Time Predictor")
st.markdown("Predict delivery times based on order details using a pre-trained LightGBM model.")

user_end, back_end = st.tabs(["User Input", "Backend Info"])

# -----------------------
# Input from users
# -----------------------
create_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with user_end:

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enter market id:")
        market_id = st.selectbox("Market ID", options=[1, 2, 3, 4, 5, 6])
        order_protocol = st.selectbox("Order Protocol", options=[1, 2, 3, 4, 5, 6, 7])

    with col2:
        st.subheader("Enter store details:")
        store_id = st.number_input("Store ID", min_value=1, max_value=6743, value=1, step=1)

        categories = ['american', 'mexican', 'indian', 'italian', 'sandwich',
       'thai', 'cafe', 'salad', 'pizza', 'chinese', 'singaporean',
       'burger', 'breakfast', 'mediterranean', 'japanese', 'greek',
       'catering', 'filipino', 'convenience-store', 'other', 'korean',
       'vegan', 'asian', 'barbecue', 'fast', 'dessert', 'smoothie',
       'seafood', 'vietnamese', 'cajun', 'steak', 'middle-eastern',
       'soup', 'vegetarian', 'persian', 'nepalese', 'sushi',
       'latin-american', 'hawaiian', 'chocolate', 'burmese', 'british',
       'pasta', 'alcohol', 'dim-sum', 'peruvian', 'turkish', 'malaysian',
       'ethiopian', 'afghan', 'bubble-tea', 'german', 'french',
       'caribbean', 'gluten-free', 'comfort-food', 'gastropub',
       'pakistani', 'moroccan', 'spanish', 'southern', 'tapas', 'russian',
       'brazilian', 'european', 'cheese', 'african', 'argentine',
       'kosher', 'irish', 'lebanese', 'belgian', 'indonesian',
       'alcohol-plus-food']
        store_primary_category = st.selectbox("Store Primary Category", options=categories)

    col3 = st.columns(1)
    with col3[0]:
        st.subheader("Enter order features: ")
        total_items = st.number_input("Total Items", min_value=1, max_value=20, value=1, step=1)
        subtotal = st.number_input("Subtotal ($)", min_value=1.0, max_value=500.0, value=10.0, step=0.5)
        num_distinct_items = st.number_input("Number of Distinct Items", min_value=1, max_value=20, value=1, step=1)
        min_item_price = st.number_input("Minimum Item Price ($)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
        max_item_price = st.number_input("Maximum Item Price ($)", min_value=0.5, max_value=200.0, value=20.0, step=0.5)

# -----------------------
# Backend Info
# -----------------------
with back_end:
    col4 = st.columns(1)
    with col4[0]:
        st.subheader("State of marketplace:")
        total_onshift_dashers = st.number_input("Total Onshift Dashers", min_value=1, max_value=500, value=50, step=1)
        total_busy_dashers = st.number_input("Total Busy Dashers", min_value=0, max_value=500, value=20, step=1)
        total_outstanding_orders = st.number_input("Total Outstanding Orders", min_value=0, max_value=1000, value=100, step=1)
        estimated_order_place_duration = st.number_input("Estimated Order Place Duration (secs)", min_value=30, max_value=3600, value=300, step=10)
        estimated_store_to_consumer_duration = st.number_input("Estimated Store to Consumer Duration (secs)", min_value=300, max_value=7200, value=1800, step=10)

predict = st.button("Predict Delivery Time", type='primary')

# feature engineering on user inputs
if predict:
    input_data = pd.DataFrame({
        'market_id': [market_id],
        'store_id': [store_id],
        'store_primary_category': [store_primary_category],
        'order_protocol': [order_protocol],
        'total_items': [total_items],
        'subtotal': [subtotal],
        'num_distinct_items': [num_distinct_items],
        'min_item_price': [min_item_price],
        'max_item_price': [max_item_price],
        'total_onshift_dashers': [total_onshift_dashers],
        'total_busy_dashers': [total_busy_dashers],
        'total_outstanding_orders': [total_outstanding_orders],
        'estimated_order_place_duration': [estimated_order_place_duration],
        'estimated_store_to_consumer_driving_duration': [estimated_store_to_consumer_duration]
    })

    # Perform feature engineering (similar to training)
    input_data['dasher_availability_ratio'] = input_data['total_busy_dashers'] / (input_data['total_onshift_dashers'] + 1)
    input_data['non_prep_duration'] = input_data['estimated_order_place_duration'] + input_data['estimated_store_to_consumer_driving_duration']
    input_data['price_range'] = input_data['max_item_price'] - input_data['min_item_price']
    input_data['avg_item_price'] = input_data['subtotal'] / input_data['total_items']
    input_data['distinct_items_ratio'] = input_data['num_distinct_items'] / input_data['total_items']

    collinear_to_drop = [
        'total_onshift_dashers',
        'total_busy_dashers',
        'subtotal',
        'num_distinct_items',
        'min_item_price',
        'max_item_price',
        'non_prep_duration'
    ]
    input_data.drop(columns=collinear_to_drop, inplace=True)

    # Log transform numeric features
    to_logtransform = input_data.drop(columns=[
                                'market_id',
                                'store_id',
                                'store_primary_category',
                                'order_protocol',
                                'nan_store_primary_category',
                                'delivery_duration'
                                ])
    for col in to_logtransform:
        input_data[col] = np.log1p(input_data[col])

    # One-hot encode categorical features
    to_encode = ['market_id', 'store_primary_category', 'order_protocol']
    encoded = encoder.transform(input_data[to_encode])
    encoded_cols = encoder.get_feature_names_out(to_encode)
    
    # rename columns for custom prefixes
    encoded_cols = [
        col.replace('nan_store_primary_category_', 'category_')
           .replace('order_protocol_', 'protocol_')
        for col in encoded_cols
    ]
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_data.index)
    df = pd.concat([input_data.drop(columns=to_encode), encoded_df], axis=1)

    df.drop(columns=['store_id', 'store_primary_category'], inplace=True)

    # Predict using the pre-trained model
    prediction = model.predict(df)
    predicted_seconds = int(np.expm1(prediction)[0])  # inverse of log
    st.success(f"Predicted Delivery Time: {predicted_seconds} seconds 🚚")