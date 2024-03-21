#!/bin/bash

# Выполнение скрипта создания данных
python data_creation.py

# Выполнение скрипта предобработки
python model_preprocessing.py

# Выполнение скрипта подготовки модели
python model_preparation.py

# Выполнение скрипта тестирования модели
python model_testing.py
