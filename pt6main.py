import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df_led = pd.read_csv("Raw Data1.csv")
df_inc = pd.read_csv("Raw Data 2.csv")

df_led.columns = ["Time", "Illuminance"]
df_inc.columns = ["Time", "Illuminance"]

def create_windows(df, label, window_size=5):
    X, y = [], []
    for i in range(len(df) - window_size):
        window = df["Illuminance"].iloc[i:i + window_size].values
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)

X_led, y_led = create_windows(df_led, label=1)
X_inc, y_inc = create_windows(df_inc, label=0)

X = np.vstack((X_led, X_inc))
y = np.concatenate((y_led, y_inc))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nТочність на тестовій вибірці: {acc:.2f}")

y_pred = model.predict(X_test).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

avg_led = df_led["Illuminance"].mean()
avg_inc = df_inc["Illuminance"].mean()
print(f"Середній рівень освітленості LED: {avg_led:.2f} lx")
print(f"Середній рівень освітленості лампи розжарювання: {avg_inc:.2f} lx")

print("\nКласифікаційний звіт:")
print(classification_report(y_test, y_pred_labels, target_names=["Розжарювання", "LED"]))

plt.figure(figsize=(10, 6))
plt.plot(df_led["Time"], df_led["Illuminance"], label="LED", color='green')
plt.plot(df_inc["Time"], df_inc["Illuminance"], label="Лампа розжарювання", color='orange')
plt.xlabel("Час (с)")
plt.ylabel("Освітленість (lx)")
plt.title("Порівняння рівня освітленості: LED vs Розжарювання")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
