import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle

def train_model(X_train, y_train, X_test, y_test):
    mlflow.set_experiment("doordash_delivery_time_prediction")
    # define models and their params
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(ln_estimators=500, learning_rate=0.05, max_depth=8, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
    }

    # to store results
    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f'\n🔹 Training {name}...')

            # log model type
            mlflow.log_param("model_name", name)

            # log hyperparameters (for tree-based models
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
            
            # fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            # log model itself
            mlflow.sklearn.log_model(model, name.lower())

            # save local pickle file
            if name != 'RandomForest':  # RandomForest model is large
                with open(f'models/{name.lower()}.pkl', 'wb') as f:
                    pickle.dump(model, f)

            # Append for local tracking
            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(f"✅ {name} logged | MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

    # compare results
    results_df = pd.DataFrame(results).sort_values(by='MAE')
    print("\n📊 Model Comparision:\n")
    print(results_df)

    # save results to reports