import pandas as pd
import time

def calculate_energy_consumption(df):
    # Використовуємо векторизовану операцію для швидшого обчислення
    return df['AEP_MW'].sum()

# Завантаження даних
df = pd.read_csv('AEP_hourly.csv', parse_dates=['Datetime'])

# Порівняння часу виконання
start_time = time.time()
total_consumption = calculate_energy_consumption(df)
end_time = time.time()

# Додаткові виводи для аналізу
print(f"Загальне споживання енергії: {total_consumption:.2f} МВт")
print(f"Час виконання Python: {end_time - start_time:.10f} секунд")
print(f"Кількість рядків у датасеті: {len(df)}")
print(f"Перші 5 рядків даних:\n{df.head()}")
