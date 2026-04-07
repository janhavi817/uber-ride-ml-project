import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

print("Loading dataset...")
# Load dataset
df = pd.read_csv('uber_data.csv')

print("Initial data shape:", df.shape)

# Handle missing values
df = df.dropna()
print("After dropna shape:", df.shape)

# Convert datetime columns
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Feature engineering
print("Feature engineering...")
# Convert trip_duration to minutes
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0

# Extract hour and day of week from pickup_datetime
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

# Filter out unrealistic values just in case (e.g. <=0 duration, negative distance, large durations)
df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 300)]
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 10)]

print("After filtering shape:", df.shape)

# Select features
features = ['trip_distance', 'passenger_count', 'pickup_hour', 'day_of_week']
target = 'trip_duration'

X = df[features]
y = df[target]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Model: Linear Regression
print("Training Linear Regression (Baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("Linear Regression Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, lr_preds):.2f}")
print(f"R2: {r2_score(y_test, lr_preds):.4f}")

# Final Model: Random Forest
print("Training Random Forest...")
# Use a sensible number of estimators / max depth to keep training time reasonable
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("\nRandom Forest Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, rf_preds):.2f}")
print(f"R2: {r2_score(y_test, rf_preds):.4f}")

# Save the trained model
print("Saving model to app/model.pkl...")
import os
os.makedirs('../app', exist_ok=True)
with open('../app/model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model saved successfully!")
