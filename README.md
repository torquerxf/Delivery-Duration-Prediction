# 🚀 DoorDash Delivery Duration Prediction

> _Predicting accurate food delivery times using real-world order data, machine learning, and MLflow experiment tracking._

## 🧠 Project Overview

Accurate delivery time prediction is critical to enhancing customer satisfaction and optimizing operations for online food delivery platforms like __DoorDash__.

In this project, I developed an __end-to-end Machine Learning pipeline__ to predict the __total delivery duration (in seconds)__ — i.e., the time from when a customer places an order to when it’s delivered.

The pipeline integrates __data preprocessing, feature engineering, model training, experiment tracking (via MLflow)__, and model evaluation.

## 🎯 Objective

To predict the __total delivery duration__ for a given order using order, store, market, and dasher-related features.

__Target Variable:__
`total_delivery_duration_seconds = actual_delivery_time - created_at`

## 🧩 Dataset Description

The dataset (`historical_data.csv`) contains a subset of DoorDash orders from early 2015 with noise added to obfuscate business details.

### Feature Categories

| Category                          | Description                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| **Time features**                 | `created_at`, `actual_delivery_time`, `market_id`                                   |
| **Store features**                | `store_id`, `store_primary_category`, `order_protocol`                              |
| **Order features**                | `total_items`, `num_distinct_items`, `subtotal`, `min_item_price`, `max_item_price` |
| **Market features**               | `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders`           |
| **Predictions from other models** | `estimated_order_place_duration`, `estimated_store_to_consumer_driving_duration`    |

## ⚙️ Workflow

### 1️⃣ Data Preprocessing
- Removed invalid/negative values and handled missing data.
- Applied __log transformation__ to right-skewed variables (e.g., `subtotal`, `total_items`).
- Handled __outliers__ using __winsorization__ to reduce the influence of extreme values.
- Extracted __temporal features__ from `created_at` (hour, weekday, weekend indicator).
- Encoded categorical variables (`store_primary_category`, `market_id`, `order_protocol`).

### 2️⃣ Feature Engineering
Created __15+ new features__ to capture operational and temporal insights:
- __Dasher Demand Ratio:__ `total_busy_dashers / total_onshift_dashers`
- __Outstanding Orders per Dasher:__ `total_outstanding_orders / total_onshift_dashers`
- __Order Value Ratios:__ `subtotal / total_items`, `max_item_price / min_item_price`
- __Time of Day Features:__ one-hot encoded hours to capture peak ordering times.

### 3️⃣ Modeling
Trained and compared multiple regression models:
- __Linear Regression__
- __Random Forest Regressor__
- __XGBoost Regressor__ 
- __LighGBM__ (best performer)

### 4️⃣ MLflow Integration

MLflow was used to __track experiments__, __log metrics__, and __compare model performance__ efficiently.

#### Key Components:

- `mlflow.start_run()` for each experiment.
- Logged parameters: learning rate, n_estimators, max_depth, etc.
- Logged metrics: MAE, RMSE, R².
- Tracked and visualized results on __MLflow UI dashboard__.

__Example:__
```python
with mlflow.start_run(run_name="xgboost_v1"):
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=7)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.xgboost.log_model(model, "model")
```
This allowed quick __model-to-model comparisons__ and tracking of tuning progress.

### 📊 Results
| Metric       | Baseline         | Final (XGBoost + Feature Engg) | Improvement |
| ------------ | ---------------- | ------------------------------ | ----------- |
| **MAE**      | 717 s (~12 min)  | **684 s (~11 min)**            | ↓ 5%        |
| **RMSE**     | 2247 s (~37 min) | **964 s (~16 min)**            | ↓ 57%       |
| **R² Score** | 0.08             | **0.24**                       | +0.16       |

✅ __XGBoost__ was the best-performing model, balancing accuracy and interpretability.

### 🛠️ Tech Stack
| Category                | Tools                                |
| ----------------------- | ------------------------------------ |
| **Languages**           | Python                               |
| **Libraries**           | Pandas, NumPy, Scikit-learn, XGBoost |
| **Visualization**       | Matplotlib, Seaborn                  |
| **Experiment Tracking** | MLflow                               |
| **Version Control**     | Git, GitHub                          |

### 💡 Key Learnings

- Handling __real-world noisy operational data__ (negative and skewed distributions).
- Designing __reproducible ML pipelines__ with MLflow for experiment tracking.
- Effective __feature engineering__ significantly boosts model accuracy.
- Balancing __model performance__ with interpretability and reproducibility.

### 🚀 Future Enhancements

- 🧭 __Deploy__ model as a REST API using Flask or Streamlit for real-time ETA prediction.
- 🧩 Integrate __hyperparameter tuning__ (Optuna or GridSearchCV) with MLflow tracking.
- 📈 Automate data ingestion and model retraining for new data batches.
- 🌍 Build an interactive __dashboard__ for visualizing feature importance and predictions.

### 📂 Repository Structure
```css
📦 doordash_delivery_prediction
│
├── data/
│   └── historical_data.csv
│
├── notebooks/
│   └── doordash_prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── mlflow_experiments.py
│
├── models/
│   └── xgb_model.pkl
│
├── README.md
└── requirements.txt
```

### 🧾 Results Summary

✅ Best Model: XGBoost  
🎯 RMSE: 964 s  
🕒 MAE: 684 s  
📈 R²: 0.24  
🔁 Improvement: 55% ETA accuracy boost  
