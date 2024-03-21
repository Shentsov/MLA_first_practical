import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def load_data(filename):
    # Загрузка данных из указанного файла
    data = pd.read_csv(filename)
    X = data.drop(columns=['Exchange_Rate'])  # Исключаем курс рубля из признаков
    y = data['Exchange_Rate']
    return X, y

def load_model(filename):
    with open(f'models/{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

def get_metrics(y_test, predictions):

    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    print("Средняя квадратичная ошибка: ",mse)
    print("Корень из средней квадратичной ошибки: ",rmse)
    print("Cредняя абсолютная ошибка: ",mae)
    print("Cредняя абсолютная процентная ошибка: ",mape)
    

def main():
    # Загрузка тестовых данных
    X_rub_test, y_rub_test = load_data('preprocessed/preprocessed_test_rub_data.csv')
    X_cny_test, y_cny_test = load_data('preprocessed/preprocessed_test_cny_data.csv')
    X_bhat_test, y_bhat_test = load_data('preprocessed/preprocessed_test_bhat_data.csv')
    
    # Загрузка обученной модели (если используется)
    rub_model = load_model('rub_model.pkl')
    cny_model = load_model('cny_model.pkl')
    bhat_model = load_model('bhat_model.pkl')
    
    # Получение прогноза
    rub_predictions = rub_model.predict(X_rub_test)
    cny_predictions = cny_model.predict(X_cny_test)
    bhat_predictions = bhat_model.predict(X_bhat_test)

    # Получение метрик
    print('Метрики модели прогнозирования курса рубля к доллару:')
    get_metrics(y_rub_test, rub_predictions)
    print('_______________________________')
    print('Метрики модели прогнозирования курса юаня к доллару:')
    get_metrics(y_cny_test, cny_predictions)
    print('_______________________________')
    print('Метрики модели прогнозирования курса бат к доллару:')
    get_metrics(y_bhat_test, bhat_predictions)
    print('_______________________________')

   
      

if __name__ == "__main__":
    main()
