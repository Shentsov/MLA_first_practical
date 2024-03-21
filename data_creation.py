import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_rub_exchange_rate_data(start_date, end_date, anomaly_rate=0.05, trend_factor=0.02, noise_factor=0.1):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    time = np.arange(len(dates))
    
    # Тренд
    trend = trend_factor * time

    # Аномалии
    anomalies = np.random.choice(len(dates), int(anomaly_rate * len(dates)), replace=False)
    trend[anomalies] += np.random.normal(scale=noise_factor * 5, size=len(anomalies))

    # Шум
    noise = np.random.normal(scale=noise_factor, size=len(dates))

    # Курс
    rub_exchange_rate = 70 + trend + noise
    
    data = {'Date': dates, 'Exchange_Rate': rub_exchange_rate}
    return pd.DataFrame(data)


def generate_cny_exchange_rate_data(start_date, end_date, anomaly_rate=0.05, trend_factor=0.02, noise_factor=0.1):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    time = np.arange(len(dates))
    
    # Тренд
    trend = trend_factor * time

    # Аномалии
    anomalies = np.random.choice(len(dates), int(anomaly_rate * len(dates)), replace=False)
    trend[anomalies] += np.random.normal(scale=noise_factor * 5, size=len(anomalies))

    # Шум
    noise = np.random.normal(scale=noise_factor, size=len(dates))

    # Курс
    cny_exchange_rate = 7 + trend + noise
    
    data = {'Date': dates, 'Exchange_Rate': cny_exchange_rate}
    return pd.DataFrame(data)


def generate_bhat_exchange_rate_data(start_date, end_date, anomaly_rate=0.05, trend_factor=0.02, noise_factor=0.1):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    time = np.arange(len(dates))

    # Тренд
    trend = trend_factor * time

    # Аномалии
    anomalies = np.random.choice(len(dates), int(anomaly_rate * len(dates)), replace=False)
    trend[anomalies] += np.random.normal(scale=noise_factor * 5, size=len(anomalies))

    # Шум
    noise = np.random.normal(scale=noise_factor, size=len(dates))

    # Курс
    bhat_exchange_rate = 36 + trend + noise

    data = {'Date': dates, 'Exchange_Rate': bhat_exchange_rate}
    return pd.DataFrame(data)

# Метод для сохраниня данных в файл csv
def save_data(data, folder, filename):

    # Создание директроии 
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    
    # Запись данных в файл 
    data.to_csv(filepath, index=False)
    print(f"Данные записаны в {filepath}")

if __name__ == "__main__":

    # Берем даты с 1 января 2020 до 1 марта 2024
    start_date = '2020-01-01'
    end_date = '2024-03-01'

    # Генерация данных курса рубля
    all_rub_exchange_rate_data = generate_rub_exchange_rate_data(start_date, end_date, trend_factor=0.02, noise_factor=0.2)
    # Генерация данных курса доллара

    start_date = '2024-01-01' 
    end_date = '2024-12-31' 
    all_cny_exchange_rate_data = generate_cny_exchange_rate_data(start_date, end_date, trend_factor=0.03, noise_factor=0.1)

    # Генерация данных курса Тайских-бат
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    all_bhat_exchange_rate_data = generate_bhat_exchange_rate_data(start_date, end_date, trend_factor=0.02, noise_factor=0.1)

    # Разделение данных на train и test в соотношении 80/20
    train_rub_exchange_data, test_rub_exchange_data = train_test_split(all_rub_exchange_rate_data, test_size=0.2, random_state=42)
    train_cny_exchange_data, test_cny_exchange_data = train_test_split(all_cny_exchange_rate_data, test_size=0.2, random_state=42)
    train_bhat_exchange_data, test_bhat_exchange_data = train_test_split(all_bhat_exchange_rate_data, test_size=0.2, random_state=42)

    # Сохранение данных
    save_data(train_rub_exchange_data, 'train', 'train_rub_exchange_rate_data.csv')
    save_data(test_rub_exchange_data, 'test', 'test_rub_exchange_rate_data.csv')

    # Сохранение данных
    save_data(train_cny_exchange_data, 'train', 'train_cny_exchange_rate_data.csv')
    save_data(test_cny_exchange_data, 'test', 'test_cny_exchange_rate_data.csv')

    # Сохранение данных
    save_data(train_bhat_exchange_data, 'train', 'train_bhat_exchange_rate_data.csv')
    save_data(test_bhat_exchange_data, 'test', 'test_bhat_exchange_rate_data.csv')
