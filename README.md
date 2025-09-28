# 🚴‍♂️ DoorDash Delivery Duration Prediction

Predicting delivery times is critical for food delivery platforms like DoorDash, UberEats, or Swiggy. Accurate predictions not only enhance customer satisfaction but also improve driver allocation and efficiency.

This project demonstrates an end-to-end machine learning pipeline to predict DoorDash delivery durations using real-world-like dataset.

## 📌 Project Overview

- __Objective:__ Predict actual delivery duration (in seconds) given order, restaurant, and delivery partner features.

- __Dataset:__ DoorDash Delivery Dataset (~100K+ rows, multiple categorical & numerical features). [Link-🔗](https://platform.stratascratch.com/data-projects/delivery-duration-prediction)

- __Challenges:__

  - Skewed distributions of delivery times.

  - Presence of extreme outliers (very short/very long deliveries).

  - Features with negative values that don’t make sense (cleaned during preprocessing).

- __Approach:__

  1. Data cleaning and preprocessing

  2. Exploratory data analysis (EDA)

  3. Feature engineering (encoding, outlier handling, transformations)

  4. Model development & tuning (baseline vs. advanced ML)

  5. Evaluation & results

## ⚙️ Tools & Technologies

- __Python__

- __Libraries:__

  - `pandas`, `numpy` → data wrangling

  - `matplotlib`, `seaborn` → visualizations

  - `scikit-learn` → preprocessing, baseline ML models

  - `xgboost` → gradient boosting model

- __Version Control:__ Git & GitHub

- __Environment:__ Jupyter Notebook

## 📊 Key Steps
### 1. Data Preprocessing

✔️ Removed inconsistent and erroneous entries (e.g., negative delivery durations).  
✔️ Handled right-skewed features with __log transformations__.  
✔️ Applied __IQR-based filtering__ to reduce outlier impact.  
✔️ Encoded categorical variables with `OneHotEncoder`.  

### 2. Exploratory Data Analysis (EDA)

- Identified that many delivery-related features were __heavily skewed__.

- Found correlation between restaurant location, order size, and delivery duration.

- Visualized distributions and checked feature-target relationships.

### 3. Feature Engineering

- Derived new features like __delivery distance buckets__ and __order size categories__.

- Standardized numerical features.

### 4. Modeling

- Tried multiple models: __Linear Regression, Decision Tree, Random Forest, XGBRegressor__.

- Tuned hyperparameters with `GridSearchCV`.

### 5. Evaluation Metrics

- __Baseline Model (Mean Predictor):__

  - MAE: ~717

  - RMSE: ~2247

- __Final Model (XGBRegressor):__

  - MAE: __684 seconds (~11.4 minutes)__

  - RMSE: __964 seconds (~16 minutes)__

  - R² Score: __0.24__

## 🚀 Results

- Reduced RMSE by __~57%__ compared to baseline.

- XGBoost outperformed traditional models due to handling non-linearities & feature interactions.

- Insights: Larger order sizes & higher delivery distances significantly increase delivery duration.

## 🔮 Future Enhancements

- Apply __feature importance analysis__ (SHAP values) to interpret model decisions.

- Experiment with __ensemble models__ (stacking/blending).

- Use __deep learning (LSTM)__ for time-series-like prediction of delivery duration.

- Deploy as an __API with Flask/Streamlit__ for real-time delivery time predictions.

## 📁 Repository Structure
```bash
📦 doordash-delivery-prediction  
 ┣ 📂 data/             # Dataset (raw + cleaned)  
 ┣ 📂 notebooks/        # Jupyter notebooks for EDA & modeling  
 ┣ 📂 src/              # Scripts for preprocessing, training, evaluation  
 ┣ 📜 requirements.txt  # Dependencies  
 ┣ 📜 README.md         # Project overview  
 ┗ 📜 results.png       # Sample visualizations & model results
```

## ✅ Key Takeaways

- Demonstrated __real-world ML problem-solving__ with skewed data and outliers.

- Built an end-to-end __data science pipeline__.

- Showcased usage of __modern ML tools (XGBoost, sklearn)__ for predictive modeling.
##
⭐ If you found this project interesting, feel free to fork, star ⭐, or contribute!
