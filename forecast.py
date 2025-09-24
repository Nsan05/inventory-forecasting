import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from datetime import datetime
from datetime import date
import holidays
import joblib

import warnings
warnings.filterwarnings('ignore')

# ----- Reading the data -----

df = pd.read_csv("data/train.csv")

# ----- Feature Engineering -----

parts = df["date"].str.split("/", n = 3, expand = True)
df["day"] = parts[0].astype(int)
df["month"] = parts[1].astype(int)
df["year"] = parts[2].astype(int)

# weekend feature

def is_weekend(day, month, year):
    d = datetime(year, month, day)
    if (d.weekday() > 4):
        return 1
    else:
        return 0
    
df["weekend"] = df.apply(lambda row: is_weekend(row['day'], row['month'], row['year']), axis=1)
# print(df.head())

# public holiday feature

uae_holidays = holidays.country_holidays('AE', years = [2024])
df["holiday"] = df.apply(lambda row: 1 if uae_holidays.get(row["date"]) else 0, axis=1)

# adding cyclic features for month
df["m1"] = np.sin(df["month"] * (2 * np.pi / 12))
df["m2"] = np.cos(df["month"] * (2 * np.pi / 12))

# adding day of the week feature

def day_of_the_week(day, month, year):
    d = datetime(year,month,day)
    return d.weekday()

df["weekday"] = df.apply(lambda x: day_of_the_week(x["day"], x["month"], x["year"]), axis = 1)

df.drop(columns = ["date"], inplace=True, axis = 1)

# ----- Exploratory Data Analysis -----

features = ['store', 'year', 'month', 'weekday', 'weekend', 'holiday']
plt.subplots(figsize=(20,10))
for ind, feature in enumerate(features):
    plt.subplot(2, 3, ind + 1)
    df.groupby(feature).mean()['sales'].plot.bar()
plt.show()

plt.figure(figsize=(10,5))
df.groupby("day").mean()["sales"].plot()
plt.show()

# sma
plt.figure(figsize=(15,10))

window_size = 30
data = df[df["year"] == 2013]
windows = data["sales"].rolling(window_size)
sma = windows.mean()
sma = sma[window_size - 1:]

data["sales"].plot()
sma.plot()
plt.show()

# identifying distribution and outliers

plt.subplots(figsize=(16,5))
plt.subplot(1,2,1)
sb.distplot(df["sales"])

plt.subplot(1,2,2)
sb.boxplot(df["sales"])
plt.show()

# checking if there are high features having high coorelation 
plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar= False)
plt.show()

# removing outliers
df = df[df['sales'] < 140]

# ----- Model Training -----
features = df.drop(columns = ['sales', 'year'], axis = 1)
target = df['sales']
x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.05, random_state=22)

# standardization of data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

#selecting a model
models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]

for i in range(4):
    print(f'{models[i]}: ')
    models[i].fit(x_train, y_train)

    train_pred = models[i].predict(x_train)
    print("Training error: ", mae(y_train, train_pred))

    val_pred = models[i].predict(x_val)
    print("Validation error: ", mae(y_val, val_pred))
    print()

# Saving the model
best_model = models[1]
joblib.dump(best_model, "models/xgb_sales_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")


# future prediction
today = datetime.today().date()
future_dates = pd.date_range(start=today, periods=7)
stores = df['store'].unique() 
items = df['item'].unique()

future_rows = []
for store in stores:
    for item in items:
        for date in future_dates:
            d = date.day
            m = date.month
            y = date.year
            weekend = is_weekend(d, m, y)
            holiday = 1 if uae_holidays.get(date) else 0
            m1 = np.sin(m * (2 * np.pi / 12))
            m2 = np.cos(m * (2 * np.pi / 12))
            weekday = date.weekday()

            future_rows.append([store, item, d, m, weekend, holiday, m1, m2, weekday])

future_features = pd.DataFrame(future_rows, columns=[
    "store", "item", "day", "month", "weekend", "holiday", 
    "m1", "m2", "weekday"
])

future_features_scaled = scaler.transform(future_features)

future_features['predicted_sales'] = best_model.predict(future_features_scaled)

# now displayhing the prediction and data in a more readbale format
inventory_forecast = future_features.groupby(['store','item'])['predicted_sales'].sum().reset_index()
inventory_forecast.rename(columns={'predicted_sales':'forecasted_stock'}, inplace=True)

store_num = int(input("Enter a store number (1-10): "))
while (store_num < 0 or store_num > 10):
    print("please check input again")
    store_num = int(input("Enter a store number (1-10): "))

selected_forecast = inventory_forecast[inventory_forecast['store'] == store_num]

plt.figure(figsize = (15,5))
sb.barplot(x = 'item', y = 'forecasted_stock', data=selected_forecast)
plt.title(f"Forcasted stock quantities for store {store_num}")
plt.show()