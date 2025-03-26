import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("DAYTON_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.values.flatten().reshape(-1, 1)).flatten()


# CUSUM метод для виявлення різких змін
def cusum(data, threshold=0.01):
    pos_sum, neg_sum = np.zeros_like(data), np.zeros_like(data)
    change_points = []
    for i in range(1, len(data)):
        diff = data[i] - data[i - 1]
        pos_sum[i] = max(0, pos_sum[i - 1] + diff)
        neg_sum[i] = min(0, neg_sum[i - 1] + diff)
        if pos_sum[i] > threshold:
            change_points.append(i)
            pos_sum[i] = 0
        elif neg_sum[i] < -threshold:
            change_points.append(i)
            neg_sum[i] = 0
    return np.array(change_points)


change_points = cusum(df_scaled, threshold=0.01)

# Seasonal decomposition
result = seasonal_decompose(df.iloc[:, 0], model='additive', period=24)
trend = result.trend

# Ковзне середнє
rolling_avg = df.iloc[:, 0].rolling(window=24).mean()

# Побудова графіків по роках
years = df.index.year.unique()
for year in years:
    df_year = df[df.index.year == year]
    trend_year = trend[df.index.year == year]
    rolling_avg_year = rolling_avg[df.index.year == year]

    # Отримання індексів змін CUSUM для поточного року
    change_points_year = df.index[change_points]
    change_points_year = change_points_year[change_points_year.year == year]

    plt.figure(figsize=(12, 6))
    plt.plot(df_year.index, df_year.iloc[:, 0], label="Оригінальні дані", alpha=0.5)
    plt.plot(df_year.index, trend_year, label="Тренд (seasonal_decompose)", linestyle="dashed")
    plt.plot(df_year.index, rolling_avg_year, label="Ковзне середнє (24 години)", linestyle="dotted")
    plt.scatter(change_points_year, df_year.loc[change_points_year].iloc[:, 0], color="red", label="CUSUM точки змін")
    plt.xlabel("Дата")
    plt.ylabel("Споживання енергії")
    plt.title(f"Аналіз змін тренду за {year} рік")
    plt.legend()
    plt.show()
