# Используем официальный образ Python
FROM python:3.9-slim

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем все файлы проекта в рабочую директорию контейнера
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Установка переменной окружения для предотвращения создания .pyc файлов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Выполнение скриптов для генерации данных, предобработки, подготовки модели и тестирования
RUN python data_creation.py
RUN python model_preprocessing.py
RUN python model_preparation.py
RUN python model_testing.py

# Открываем порт для Flask
EXPOSE 12000

# Запуск Flask API
CMD ["python", "api.py"]