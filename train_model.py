import pandas as pd
import numpy as np
import pickle
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def cal_SOi(so2):
    if so2 <= 40:
        return so2 * (50/40)
    elif so2 <= 80:
        return 50 + (so2-40)*(50/40)
    elif so2 <= 380:
        return 100 + (so2-80)*(100/300)
    elif so2 <= 800:
        return 200 + (so2-380)*(100/420)
    elif so2 <= 1600:
        return 300 + (so2-800)*(100/800)
    else:
        return 400 + (so2-1600)*(100/800)

def cal_Noi(no2):
    if no2 <= 40:
        return no2 * 50/40
    elif no2 <= 80:
        return 50 + (no2-40)*(50/40)
    elif no2 <= 180:
        return 100 + (no2-80)*(100/100)
    elif no2 <= 280:
        return 200 + (no2-180)*(100/100)
    elif no2 <= 400:
        return 300 + (no2-280)*(100/120)
    else:
        return 400 + (no2-400)*(100/120)

def cal_RSPMT(rspm):
    if rspm <= 30:
        return rspm * 50/30
    elif rspm <= 60:
        return 50 + (rspm-30) * 50/30
    elif rspm <= 90:
        return 100 + (rspm-60) * 100/30
    elif rspm <= 120:
        return 200 + (rspm-90) * 100/30
    elif rspm <= 250:
        return 300 + (rspm-120) * (100/130)
    else:
        return 400 + (rspm-250) * (100/130)

def cal_SPMi(spm):
    if spm <= 50:
        return spm * 50/50
    elif spm <= 100:
        return 50 + (spm-50) * (50/50)
    elif spm <= 250:
        return 100 + (spm-100) * (100/150)
    elif spm <= 350:
        return 200 + (spm-250) * (100/100)
    elif spm <= 430:
        return 300 + (spm-350) * (100/80)
    else:
        return 400 + (spm-430) * (100/430)

def cal_aqi(si, ni, rpi, spi):
    return max(si, ni, rpi, spi)

def main():
    
    df = pd.read_csv('C:/Users/hp/AirQualityPredictionWebApp/data/data.csv', encoding='latin1')

    # 🔥 Data Cleaning
    columns_to_drop = ['agency', 'stn_code', 'date', 'sampling_date', 'location_monitoring_station']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    df['location'] = df['location'].fillna(df['location'].mode()[0])
    df['type'] = df['type'].fillna(df['type'].mode()[0])
    df.fillna(0, inplace=True)

    # 🔥 Calculate individual pollutant indices
    df['SOi'] = df['so2'].apply(cal_SOi)
    df['Noi'] = df['no2'].apply(cal_Noi)
    df['Rpi'] = df['rspm'].apply(cal_RSPMT)
    df['SPMi'] = df['spm'].apply(cal_SPMi)

    # 🔥 Calculate AQI
    df['AQI'] = df.apply(lambda row: cal_aqi(row['SOi'], row['Noi'], row['Rpi'], row['SPMi']), axis=1)

    # 🔥 Features and labels
    X = df[['SOi', 'Noi', 'Rpi', 'SPMi']]
    y = df['AQI']

    # 🔥 Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔥 Train the Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # 🔥 Save the model
    with open('aqi_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved successfully!")

    # 🔥 Predictions and Evaluation
    y_pred = model.predict(X_test)

    # 🔥 Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 🔥 Displaying the evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # 🔥 Plotting the Actual vs Predicted AQI
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI')
    plt.show()

    # 🔥 Plotting residuals
    plt.figure(figsize=(8, 6))
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
