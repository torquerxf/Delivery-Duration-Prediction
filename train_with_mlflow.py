from src.data_loader import data_loader
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering_step
from src.train_model import train_model
from sklearn.model_selection import train_test_split
import pandas as pd

# load data
print("loading data...")
df = data_loader()
print('Data Loaded -> Preprocessing...')

# preprocess
preprocessed_df = preprocess_data(df)
print('Preprocessed -> engineering...')
# feature engineering
engineered_df = feature_engineering_step(preprocessed_df)
print('Engineered -> splitting...')

# split into training and testing data
X = engineered_df.drop(columns='delivery_duration')
y = engineered_df['delivery_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Everthing done -> training model')
# train with mlflow
train_model(X_train, y_train, X_test, y_test)