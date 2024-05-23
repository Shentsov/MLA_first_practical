## Автоматический конвейер проекта машинного обучения с API

### Проект: предсказание курса валют (рубля к доллару, юаня к доллару, бат к доллару) и API на Flask

## Установка

Перед запуском необходимо склонировать проект и создать Docker-образ:

```
docker build -t exchange-rate-prediction .
```

Для запуска проекта необходимо запустить Docker-контейнер. Для этого выполните следующую команду:

```
docker run -p 12000:12000 --rm exchange-rate-prediction
```

API будет доступно по адресу: **http://127.0.0.1:12000/**

Для использования API необходимо выполнить POST запрос на следующий эндпоинт:

```
http://127.0.0.1:12000/predict
```

В теле запроса укажите два параметра

-   currency (валюта): rub, cny или bhat
-   date (дата) в формате год-месяц-число (пример: "2024-05-23")

## Команда:

-   Шенцов Ярослав
-   Медведев Денис
-   Волчок Всеволод
