import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor

def load_data(filename):
    # Загрузка данных из указанного файла
    data = pd.read_csv(filename)
    X = data.drop(columns=['Exchange_Rate'])  # Исключаем курс рубля из признаков
    y = data['Exchange_Rate']
    return X, y

def train_model(X, y):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }
    model = RandomForestRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def main():
    
    # Загрузка предобработанных тренировочных данных
    X_rub_train, y_rub_train = load_data('preprocessed/preprocessed_train_rub_data.csv')
    X_cny_train, y_cny_train = load_data('preprocessed/preprocessed_train_cny_data.csv')
    X_bhat_train, y_bhat_train = load_data('preprocessed/preprocessed_train_bhat_data.csv')
    
    # Обучение модели
    rub_model = train_model(X_rub_train, y_rub_train)
    cny_model = train_model(X_cny_train, y_cny_train)
    bhat_model = train_model(X_bhat_train, y_bhat_train)

    if not os.path.exists('models'):
    # Если не существует, создаем директорию
        os.makedirs('models') 

    # Сохранение обученных моделей
    with open('./models/rub_model.pkl', 'wb') as f:
        pickle.dump(rub_model, f)
        print('Модель rub_model.pkl сохранена')

    with open('./models/cny_model.pkl', 'wb') as f:
        pickle.dump(cny_model, f)
        print('Модель cny_model.pkl сохранена')


    with open('./models/bhat_model.pkl', 'wb') as f:
        pickle.dump(bhat_model, f)
        print('Модель bhat_model.pkl сохранена')


if __name__ == "__main__":
    main()
