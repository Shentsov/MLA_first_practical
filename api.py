from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Загрузка моделей
with open('models/rub_model.pkl', 'rb') as f:
    rub_model = pickle.load(f)
with open('models/cny_model.pkl', 'rb') as f:
    cny_model = pickle.load(f)
with open('models/bhat_model.pkl', 'rb') as f:
    bhat_model = pickle.load(f)


start_date = pd.to_datetime('2020-01-01')

# Подготовка данных
def preprocess_date(date_str):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    days_from_start = (date - start_date).days
    scaler = StandardScaler()
    scaled_days = scaler.fit_transform([[days_from_start]])
    return pd.DataFrame({'Days_from_start': scaled_days.flatten()})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    currency = data.get('currency')
    date_str = data.get('date')
    
    if not currency or not date_str:
        return jsonify({'error': 'Необходимо указать валюту и дату'}), 400

    X = preprocess_date(date_str)
        
    if currency == 'rub':
        prediction = rub_model.predict(X)
    elif currency == 'cny':
        prediction = cny_model.predict(X)
    elif currency == 'bhat':
        prediction = bhat_model.predict(X)
    else:
        return jsonify({'error': 'Неподдерживаемая валюта'}), 400
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000)
