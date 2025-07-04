# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Dummy data
data = {
    'aod': [0.3, 0.4, 0.5, 0.10],
    'temp': [28, 30, 32, 35],
    'humidity': [60, 65, 70, 75],
    'pm25': [100, 120, 135, 160]
}
df = pd.DataFrame(data)

X = df[['aod', 'temp', 'humidity']]
y = df['pm25']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("Model trained and saved.")
# This script trains a Random Forest model on dummy data and saves it to 'model.pkl'.
# You can replace the dummy data with your actual dataset for training.