import pandas as pd

# Оптимізоване читання великих файлів (зменшуємо використання пам'яті)
dtype_dict = {
    "CustomerID": "int32",
    "Genre": "category",  
    "Age": "int8",
    "Annual Income (k$)": "int16",
    "Spending Score (1-100)": "int8"
}

# Читання даних з файлу
df = pd.read_csv("Mall_Customers.csv", dtype=dtype_dict)

# Перевіряємо та очищаємо назви стовпців від зайвих пробілів
df.columns = df.columns.str.strip()

# Перевіряємо розмір в пам’яті
print(f"Розмір у пам’яті: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"Назви стовпців: {df.columns}")

# Групування за статтю та обчислення середнього доходу та середнього Spending Score
agg_df = df.groupby("Genre", observed=False).agg(  
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending_Score=("Spending Score (1-100)", "mean"),
    Count=("CustomerID", "count")
).reset_index()

# Виведення результату
print(agg_df)
