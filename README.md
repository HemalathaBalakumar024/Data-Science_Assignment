# # Data Science Internship Assignment: Sales Forecasting

## Introduction
Welcome to the Data Science Internship Assignment. In this project, you will work with real-world retail sales data to develop a forecasting model that predicts future sales for thousands of product families across different stores in Ecuador. This assignment will help you understand how external factors like promotions, holidays, economic conditions, and events impact sales, and how machine learning models can be used to improve demand forecasting.

This assignment is structured into two main parts:
1. **Data Processing and Feature Engineering (Day 1)** - Cleaning, transforming, and exploring the dataset.
2. **Model Selection, Forecasting, and Evaluation (Day 2)** - Training different forecasting models, comparing their performance, and presenting insights.

## Dataset Overview
The dataset consists of multiple files providing sales data and additional influencing factors:
- **train.csv** - Historical sales data.
- **test.csv** - The test set for which sales need to be predicted.
- **stores.csv** - Metadata about store locations and clusters.
- **oil.csv** - Daily oil prices (affecting Ecuador's economy).
- **holidays_events.csv** - Information about holidays and special events.

Your task is to forecast daily sales for each product family at each store for the next 15 days after the last training date.

---
## Part 1: Data Processing and Feature Engineering (Day 1)
### 1. Data Cleaning
- Load the dataset using Pandas.
   train = pd.read_csv('train.csv', parse_dates=['date'])
   stores = pd.read_csv('stores.csv')
   oil = pd.read_csv('oil.csv', parse_dates=['date'])
   holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])

- Handle missing values in oil prices by filling gaps with interpolation.
   oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear')
  
-  Convert date columns to proper datetime formats
   train['date'] = pd.to_datetime(train['date'])
   oil['date'] = pd.to_datetime(oil['date'])
   holidays_events['date'] = pd.to_datetime(holidays_events['date'])
   
- Merge data from `stores.csv`, `oil.csv`, and `holidays_events.csv` into the main dataset.
  train = train.merge(stores, on='store_nbr', how='left')
  train = train.merge(oil, on='date', how='left')
  train = train.merge(holidays, on='date', how='left')

### 2. Feature Engineering
#### Time-based Features:
- Extract **day, week, month, year, and day of the week**.
- Identify **seasonal trends** (e.g., are sales higher in December?).
train['day'] = train['date'].dt.day
train['week'] = train['date'].dt.isocalendar().week
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['day_of_week'] = train['date'].dt.dayofweek


#### Event-based Features:
- Create **binary flags** for holidays, promotions, and economic events.
- Identify if a day is a **government payday** (15th and last day of the month).
- Consider **earthquake impact (April 16, 2016)** as a separate feature.
train['is_holiday'] = train['type'].notna().astype(int)
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
train['is_gov_payday'] = train['day'].isin([15, train['date'].dt.days_in_month]).astype(int)
train['earthquake_impact'] = (train['date'] == '2016-04-16').astype(int)


#### Rolling Statistics:
- Compute **moving averages** and **rolling standard deviations** for past sales.
- Include **lagged features** (e.g., sales from the previous week, previous month).
train['sales_lag_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)
train['sales_lag_30'] = train.groupby(['store_nbr', 'family'])['sales'].shift(30)
train['rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
train['rolling_std_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).std().reset_index(level=[0,1], drop=True)


#### Store-Specific Aggregations:
- Compute **average sales per store type**.
- Identify **top-selling product families per cluster**.
store_avg_sales = train.groupby('store_nbr')['sales'].mean().rename('avg_store_sales')
train = train.merge(store_avg_sales, on='store_nbr', how='left')

### 3. Exploratory Data Analysis (EDA)
- Visualize **sales trends over time**.
print("Exploratory Data Analysis")
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='sales', data=train, label='Sales Trend')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.show()

- Analyze **sales before and after holidays and promotions**.
holiday_sales = train.groupby(['date', 'is_holiday'])['sales'].mean().reset_index()
sns.boxplot(x='is_holiday', y='sales', data=holiday_sales)
plt.title('Sales Before and After Holidays')
plt.xlabel('Is Holiday')
plt.ylabel('Sales')
plt.show()

- Check **correlations between oil prices and sales trends**.
correlation = train[['sales', 'dcoilwtico']].corr()
print("Correlation between Sales and Oil Prices:")
print(correlation)
sns.scatterplot(x='dcoilwtico', y='sales', data=train)
plt.title('Oil Prices vs Sales')
plt.xlabel('Oil Price')
plt.ylabel('Sales')
plt.show()

- Identify **anomalies in the data**.
sns.boxplot(y='sales', data=train)
plt.title('Sales Anomalies Detection')
plt.ylabel('Sales')
plt.show()

train.to_csv('processed_train.csv', index=False)

print("Data processing and feature engineering completed.")


## Part 2: Model Selection, Forecasting, and Evaluation (Day 2)
### 1. Model Training
Train at least five different time series forecasting models:
- **Baseline Model (Naïve Forecasting)** - Assume future sales = previous sales.

- **ARIMA (AutoRegressive Integrated Moving Average)** - A traditional time series model.
- **Random Forest Regressor** - Tree-based model to capture non-linear relationships.
- **XGBoost or LightGBM** - Gradient boosting models to improve accuracy.
- **LSTM (Long Short-Term Memory Neural Network)** - A deep learning-based forecasting model.

**Bonus Challenge:** If comfortable, implement a **Prophet model** for handling seasonality.

### 2. Model Evaluation
Compare models based on:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R-Squared Score**
- **Visual Inspection** (Plot actual vs. predicted sales)

### 3. Visualization
- Plot **historical sales and predicted sales**.
- Compare **model performances** using error metrics.
- Visualize **feature importance** (for Random Forest/XGBoost).

### 4. Interpretation and Business Insights
- Summarize **which model performed best and why**.
- Discuss **how external factors (holidays, oil prices, promotions) influenced predictions**.
- Suggest **business strategies** to improve sales forecasting (e.g., inventory planning, targeted promotions).





