import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):

    # преобразование из строки в datetime
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
    # Индекс - дата
    data.index = data['Date']
    # Убираем имя индекса
    data.index.name = None
    # Удаляем столбец с датой (так как он оказывается дублирующим)
    data.drop('Date', axis = 1, inplace = True)
    # сортируем по взрастанию даты
    data.sort_index(inplace=True)
    # заводим столбец количество дней от начала
    data['Days_from_start'] = (data.index - data.index[0]).days

    # Применение стандартного масштабирования к столбцу Days_from_start
    scaler = StandardScaler()
    data['Days_from_start'] = scaler.fit_transform(data['Days_from_start'].values.reshape(-1, 1))
    return data

# Метод для сохраниня данных в файл csv
def save_data(data, folder, filename):

    # Создание директроии 
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    
    # Запись данных в файл 
    data.to_csv(filepath, index=False)
    print(f"Подготовленные данные записаны в {filepath}")


def main():
    # Загрузка предобработанных тренировочных данных
    train_rub_data = pd.read_csv('train/train_rub_exchange_rate_data.csv')
    test_rub_data = pd.read_csv('test/test_rub_exchange_rate_data.csv')

    train_cny_data = pd.read_csv('train/train_cny_exchange_rate_data.csv')
    test_cny_data = pd.read_csv('test/test_cny_exchange_rate_data.csv')


    train_bhat_data = pd.read_csv('train/train_bhat_exchange_rate_data.csv')
    test_bhat_data = pd.read_csv('test/test_bhat_exchange_rate_data.csv')

    # Преобразование данных
    preprocessed_train_rub_data = preprocess_data(train_rub_data)
    preprocessed_test_rub_data = preprocess_data(test_rub_data)

    preprocessed_train_cny_data = preprocess_data(train_cny_data)
    preprocessed_test_cny_data = preprocess_data(test_cny_data)

    preprocessed_train_bhat_data = preprocess_data(train_bhat_data)
    preprocessed_test_bhat_data = preprocess_data(test_bhat_data)
    
    # Сохранение предобработанных данных
    # по курсу рубля
    save_data(preprocessed_train_rub_data, 'preprocessed', 'preprocessed_train_rub_data.csv')
    save_data(preprocessed_test_rub_data, 'preprocessed', 'preprocessed_test_rub_data.csv')

    # по курсу юаня
    save_data(preprocessed_train_cny_data, 'preprocessed', 'preprocessed_train_cny_data.csv')
    save_data(preprocessed_test_cny_data, 'preprocessed', 'preprocessed_test_cny_data.csv')

    # по курсу бат
    save_data(preprocessed_train_bhat_data, 'preprocessed', 'preprocessed_train_bhat_data.csv')
    save_data(preprocessed_test_bhat_data, 'preprocessed', 'preprocessed_test_bhat_data.csv')



if __name__ == "__main__":
    main()
