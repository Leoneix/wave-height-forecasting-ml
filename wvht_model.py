#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[63]:


df = pd.read_csv(r"C:\Users\parth\Documents\ML\train.csv")


# In[64]:


#Data Cleaning
df.rename(columns={'#YY': 'year'}, inplace=True)
df['datetime'] = pd.to_datetime({
    'year': df['year'],
    'month': df['MM'],
    'day': df['DD'],
    'hour': df['hh'],
    'minute': df['mm']
})
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)
df.drop(columns=['year','MM','DD','hh','mm'], inplace=True)


# In[65]:


print(df.index)
print(df.index.dtype)


# In[66]:


df.head()


# In[67]:


#Rolling and lagging features
df['WSPD_lag1'] = df['WSPD(m/s)'].shift(1)
df['WSPD_lag3'] = df['WSPD(m/s)'].shift(3)
df['WVHT_lag1'] = df['WVHT(m)'].shift(1)
df.dropna(inplace=True)


# In[68]:


df['WSPD_roll3'] = df['WSPD(m/s)'].rolling(3).mean()
df['PRES_roll6'] = df['PRES(hPa)'].rolling(6).mean()
df['WVHT_roll3'] = df['WVHT(m)'].rolling(3).mean()
df['WSPD_std3'] = df['WSPD(m/s)'].rolling(3).std()
#Avg wspd over 3 hrs etc


# In[69]:


df.dropna(inplace=True)
df.head()


# In[70]:


#Converting direction to sin and cos 
df['WDIR_sin'] = np.sin(np.deg2rad(df['WDIR(degT)']))
df['WDIR_cos'] = np.cos(np.deg2rad(df['WDIR(degT)']))

df['MWD_sin'] = np.sin(np.deg2rad(df['MWD(degT)']))
df['MWD_cos'] = np.cos(np.deg2rad(df['MWD(degT)']))

df.drop(columns=['WDIR(degT)','MWD(degT)'], inplace=True)
df.drop(columns=['ID'], inplace=True)


# In[71]:


df.head()


# In[72]:


df['WSPD_sq'] = df['WSPD(m/s)']**2
df['wind_pressure'] = df['WSPD(m/s)'] * df['PRES(hPa)']


# In[73]:


df['month'] = df.index.month
df['hour'] = df.index.hour


# In[74]:


df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)


# In[75]:


df.head()


# In[76]:


import matplotlib.pyplot as plt

plt.figure()
df['WVHT(m)'].hist(bins=50)
plt.title("Wave Height Distribution")
plt.xlabel("WVHT (m)")
plt.ylabel("Frequency")
plt.show()


# In[77]:


plt.figure(figsize=(12,5))
plt.plot(df.index, df['WVHT(m)'])
plt.title("WVHT Over Time")
plt.ylabel("Wave Height (m)")
plt.show()


# In[78]:


monthly_avg = df['WVHT(m)'].groupby(df.index.month).mean()
print(monthly_avg)


# In[79]:


monthly_avg.plot()
plt.title("Average Monthly Wave Height")
plt.xlabel("Month")
plt.ylabel("WVHT (m)")
plt.show()


# In[80]:


df['month'] = df.index.month

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# In[81]:


df.drop(columns=['month'], inplace=True)


# In[82]:


monthly_max = df['WVHT(m)'].groupby(df.index.month).max()
print(monthly_max)


# In[83]:


import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()


# In[84]:


plt.figure()
plt.scatter(df['WSPD(m/s)'], df['WVHT(m)'], alpha=0.3)
plt.xlabel("Wind Speed")
plt.ylabel("Wave Height")
plt.show()


# In[85]:


extreme = df[df['WVHT(m)'] > 2]  # adjust threshold
print(extreme[['WSPD(m/s)','GST(m/s)','PRES(hPa)']].describe())


# In[86]:


from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df['WVHT(m)'])
plt.show()


# In[87]:


X = df.drop(columns=['WVHT(m)'])
y = df['WVHT(m)']


# In[88]:


#ML
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# In[89]:


y_pred_persistence = y_test.shift(1)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse_persistence = np.sqrt(mean_squared_error(y_test[1:], y_pred_persistence[1:]))

print("Persistence RMSE:", rmse_persistence)


# In[91]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", rmse_lr)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression")
print("RMSE:", rmse_lr)
print("MAE :", mae_lr)
print("R2  :", r2_lr)


# In[92]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", rmse_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest")
print("RMSE:", rmse_rf)
print("MAE :", mae_rf)
print("R2  :", r2_rf)


# In[93]:


import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", rmse_xgb)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost")
print("RMSE:", rmse_xgb)
print("MAE :", mae_xgb)
print("R2  :", r2_xgb)


# In[94]:


plt.figure(figsize=(12,5))
plt.plot(y_test.values[:500], label='Actual')
plt.plot(y_pred_xgb[:500], label='XGB Prediction')
plt.legend()
plt.show()


# In[95]:


storm_mask = y_test > 2  # adjust threshold

rmse_storm_rf = np.sqrt(mean_squared_error(
    y_test[storm_mask],
    y_pred_rf[storm_mask]
))

print("Storm RMSE (RF):", rmse_storm_rf)


# In[96]:


#Using LSTM
features = df.columns  # include everything including WVHT
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


# In[97]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)


# In[98]:


df.head()


# In[99]:


target_index = df.columns.get_loc("WVHT(m)")


# In[102]:


def create_sequences(data, target_index, time_steps=24):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, target_index])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, target_index, 24)
X_test, y_test = create_sequences(test_scaled, target_index, 24)


# In[103]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


# In[104]:


history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# In[105]:


y_pred_scaled = model.predict(X_test)

temp = np.zeros((len(y_pred_scaled), train_scaled.shape[1]))
temp[:, target_index] = y_pred_scaled[:,0]
y_pred = scaler.inverse_transform(temp)[:, target_index]

temp_true = np.zeros((len(y_test), train_scaled.shape[1]))
temp_true[:, target_index] = y_test
y_true = scaler.inverse_transform(temp_true)[:, target_index]


# In[106]:


from sklearn.metrics import mean_squared_error
import numpy as np

rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred))
print("LSTM RMSE:", rmse_lstm)


# In[107]:


plt.plot(y_true[:500])
plt.plot(y_pred[:500])
plt.show()


# In[108]:


rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred))
mae_lstm = mean_absolute_error(y_true, y_pred)
r2_lstm = r2_score(y_true, y_pred)

print("LSTM")
print("RMSE:", rmse_lstm)
print("MAE :", mae_lstm)
print("R2  :", r2_lstm)


# In[110]:


models = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
rmse_values = [rmse_lr, rmse_rf, rmse_xgb, rmse_lstm]

plt.figure()
plt.bar(models, rmse_values)

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Model Comparison (Lower is Better)")
plt.xticks(rotation=45)

plt.show()


# In[115]:


models = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
r2_values = [r2_lr, r2_rf, r2_xgb, r2_lstm]

plt.figure()
plt.bar(models, rmse_values)

plt.xlabel("Model")
plt.ylabel("R^2 Score")
plt.title("Model Comparison (Higher is Better)")
plt.xticks(rotation=45)

plt.show()


# In[116]:


print(models)
print(r2_values)


# In[ ]:




