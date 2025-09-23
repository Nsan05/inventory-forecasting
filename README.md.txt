# Inventory Demand Forecasting using Machine Learning

This project demonstrates how to use machine learning to forecast inventory demand for multiple products across multiple stores. The goal is to help vendors maintain optimal stock levels, avoid shortages, and reduce overstocking costs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Selection](#model-training-and-selection)
- [Forecasting Future Inventory](#forecasting-future-inventory)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Project Overview

Vendors selling everyday items need to keep their stock updated so that customers don’t leave empty-handed. Maintaining the right stock levels prevents shortages and avoids overstocking costs. This project uses historical sales data and machine learning models to predict stock needs for different products across multiple stores.

---

## Dataset

The dataset `train.csv` contains sales data for 10 stores and 50 products over multiple years.  
Key columns include:
- `store`: Store number
- `item`: Product ID
- `sales`: Number of items sold
- `date`: Transaction date

> The dataset is not included in this repository due to size/privacy. Users need to provide their own dataset in the same format.

---

## Feature Engineering

Several features were added to improve model accuracy:
1. **Day, Month, Year** extracted from the date column.
2. **Weekend** – whether the date is a weekend.
3. **Holiday** – whether the date is a UAE public holiday.
4. **Cyclical features** for month (`m1 = sin(month * 2π/12)`, `m2 = cos(month * 2π/12)`) to capture seasonality.
5. **Weekday** – day of the week (0 = Monday, 6 = Sunday).

---

## Exploratory Data Analysis

EDA was performed to understand trends and patterns:
- Average sales per store, month, weekday, weekend, and holiday.
- Sales trends over days of the month.
- Simple moving average over 30-day windows.
- Distribution and outlier analysis.
- Correlation heatmap to identify highly correlated features.

Outliers in the `sales` column were removed (sales ≥ 140).

---

## Model Training and Selection

Four models were trained and compared:
- Linear Regression
- Lasso Regression
- Ridge Regression
- **XGBoost Regressor** (best performing)

### Evaluation Metrics
Models were evaluated using Mean Absolute Error (MAE) on training and validation data.

| Model              | Training MAE | Validation MAE  |
|--------------------|--------------|-----------------|
| Linear Regression  | 20.90        | 20.97           |
| Lasso Regression   | 21.01        | 21.07           |
| Ridge Regression   | 20.90        | 20.97           |
| **XGBoost**        | 6.88         | 6.90            |

> XGBoost achieved the lowest validation error and was selected as the final model. The trained model is saved as `xgb_sales_model.pkl`.

---

## Forecasting Future Inventory

The project generates inventory forecasts for a 7-day horizon:
1. Input store number.
2. Script predicts sales for each product in that store.
3. Outputs a bar chart showing forecasted stock quantities for each item.

Predictions are generated using:
- `xgb_sales_model.pkl` (trained XGBoost model)
- `scaler.pkl` (StandardScaler for feature normalization)

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Inventory-Demand-Forecasting.git
cd Inventory-Demand-Forecasting

2. Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install required packages:

pip install -r requirements.txt

4. Add the dataset:

Put train.csv into the data\ folder

---

## Usage

Run the forecasting script:

python "Inventory Demand Forecasting using Machine Learning.py"


Follow the prompt to input a store number (1–10). The script will display a bar chart with forecasted stock for each item in that store.

---

## File Structure
Projects/
│
├─ Inventory Demand Forecasting using Machine Learning.py  # Main script
├─ train.csv                                              # Sales dataset
├─ xgb_sales_model.pkl                                    # Saved XGBoost model
├─ scaler.pkl                                             # Saved scaler
├─ requirements.txt                                       # Python dependencies
├─ README.md                                              # Project documentation
└─ .gitignore                                             # Files/folders to ignore in Git

---

