#!/bin/bash

echo "Создание данных..."
python data_creation.py
if [ $? -ne 0 ]; then
    echo "Ошибка при создании данных"
    exit 1
fi

echo "Предобработка данных..."
python model_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Ошибка при предобработке данных"
    exit 1
fi

echo "Обучение модели..."
python model_preparation.py
if [ $? -ne 0 ]; then
    echo "Ошибка при обучении модели"
    exit 1
fi

echo "Тестирование модели..."
python model_testing.py
if [ $? -ne 0 ]; then
    echo "Ошибка при тестировании модели"
    exit 1
fi