# Heart Disease Prediction


## Introduction

Добро пожаловать в репозиторий проекта по прогнозированию сердечно-сосудистых заболеваний! Цель этого проекта - разработать модель, которая поможет выявлять у пациентов высокий риск ССЗ на основе их клинических данных.

## Objective

Основная задача - создать точную предсказательную систему для ранней диагностики ССЗ. Это позволит врачам оперативно принимать решения и проводить необходимые профилактические меры.

## Project Overview

Проект включает несколько этапов:

1. **Импорт библиотек и загрузка данных**
2. **Data understanding & exploration** (EDA): гистограммы, боксплоты, heatmap
3. **Data preprocessing**: удаление дубликатов, обработка категорий, one-hot encoding
4. **Correlation & multicollinearity analysis**: VIF-анализ, отбор признаков
5. **Train-test split** и масштабирование числовых признаков
6. **Hyperparameter tuning** для моделей через GridSearchCV
7. **Training & evaluation** трёх моделей:

   * Logistic Regression
   * Random Forest
   * Neural Network (PyTorch)
8. **Inference script**: единый скрипт для получения прогнозов на новых данных

## Libraries & Dataset

Используемые библиотеки:

* Pandas
* NumPy
* Scikit‑learn
* Matplotlib & Seaborn
* Statsmodels
* PyTorch
* Joblib

Датасет хранится в CSV `heart.csv` и содержит следующие признаки:

```
- age: возраст пациента
- sex: пол (0 = женщина, 1 = мужчина)
- cp: тип болей (1–4)
- trtbps: артериальное давление в покое (мм Hg)
- chol: уровень холестерина (mg/dl)
- fbs: уровень глюкозы натощак (> 120 mg/dl)
- restecg: результаты ЭКГ в покое (0–2)
- thalachh: максимальная частота сердцебиения
- exng: стенокардия при нагрузке (0/1)
- oldpeak: депрессия ST-сегмента
- slp: наклон пикового ST-сегмента (1–3)
- caa: число окрашенных крупных сосудов (0–3)
- thall: результаты теста таллием (3,6,7)
- output: наличие ССЗ (0 = нет, 1 = есть)
```

## Steps

1. **Data loading & initial checks**: проверка формата, NaN, дубликатов
2. **EDA**: распределения, выбросы, взаимосвязи
3. **Preprocessing**: one-hot encoding категорий, удаление мультиколлинеарных фич (VIF)
4. **Split & scaling**: StratifiedSplit + StandardScaler
5. **Modeling**:

   * Logistic Regression (GridSearchCV)
   * Random Forest (GridSearchCV)
   * Deep Neural Network (PyTorch)
6. **Evaluation**: AUC-ROC, confusion matrix, отчёт по классификации
7. **Inference**: `src/inference.py` для массового предсказания

## Conclusion

Этот проект демонстрирует полный цикл разработки ML-пайплайна для задачи классификации: от предобработки до продакшн-инференса. 
На наших данных Logistic Regression показала лучший результат (AUC=0.93), за ней следуют Random Forest и DNN. 

## Спасибо за внимание!