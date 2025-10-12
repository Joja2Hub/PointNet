# PointNet++ Semantic Segmentation для LAS облаков точек

Проект для семантической сегментации облаков точек формата LAS с использованием глубокого обучения (PointNet++).

## 📋 Структура проекта

```
├── model.py              # Архитектура PointNet++
├── dataset.py            # Загрузчик данных
├── train.py              # Скрипт обучения
├── predict.py            # Скрипт предсказания
├── visualize.py          # Визуализация
├── evaluate.py           # Оценка качества
├── main.py               # Главное меню
├── quick_start.py        # Быстрый старт
├── create_unlabeled.py   # Создание unlabeled.las
└── requirements.txt      # Зависимости
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

```bash
# Создать unlabeled.las из Univer2019.las
python create_unlabeled.py
```

### 3. Обучение и предсказание

**Вариант А: Автоматический запуск**

```bash
python quick_start.py
```

**Вариант Б: Пошагово**

```bash
# Обучение
python train.py

# Предсказание
python predict.py
```

**Вариант В: Интерактивное меню**

```bash
python main.py
```

## 📊 Визуализация

```bash
# Визуализация оригинала
python visualize.py Univer2019.las

# Визуализация предсказания
python visualize.py predicted.las

# Сравнение
python main.py  # выбрать пункт 5
```

## 🎯 Оценка качества

```bash
python evaluate.py Univer2019.las predicted.las
```

## ⚙️ Параметры обучения

В файле `train.py` можно настроить:

- `NUM_POINTS = 4096` - количество точек в блоке
- `BATCH_SIZE = 4` - размер батча
- `EPOCHS = 100` - количество эпох
- `LEARNING_RATE = 0.001` - скорость обучения
- `BLOCK_SIZE = 50.0` - размер блока (метры)
- `STRIDE = 25.0` - шаг блока (метры)

## 🎨 Классы (1-8)

| Класс | Цвет             | Описание       |
| ----- | ---------------- | -------------- |
| 1     | Коричневый       | -              |
| 2     | Темно-коричневый | -              |
| 3     | Зеленый          | Растительность |
| 4     | Темно-зеленый    | -              |
| 5     | Желтый           | -              |
| 6     | Красный          | Здания         |
| 7     | Синий            | -              |
| 8     | Фиолетовый       | Земля          |

## 💾 Выходные файлы

- `checkpoints/best_model.pth` - лучшая модель
- `predicted.las` - размеченное облако
- `confusion_matrix.png` - матрица ошибок

## 🔧 Требования к системе

- **GPU**: NVIDIA с CUDA (рекомендуется 8GB+ VRAM)
- **RAM**: 16GB+
- **Диск**: 5GB+ свободного места

## 🎯 Инструкция по запуску

### Вариант 1: Быстрый старт (рекомендуется)

```bash
# 1. Создать unlabeled.las
python create_unlabeled.py

# 2. Запустить обучение и предсказание
python quick_start.py
```

### Вариант 2: Пошаговый запуск

```bash
# 1. Обучение (займет несколько часов)
python train.py

# 2. Предсказание
python predict.py

# 3. Оценка качества
python evaluate.py Univer2019.las predicted.las

# 4. Визуализация
python visualize.py predicted.las
```

### Вариант 3: Интерактивное меню

```bash
python main.py
```

## 📈 Ожидаемые результаты

- **Точность**: 70-85% (зависит от данных)
- **Время обучения**: 2-6 часов на RTX 4060 Ti
- **Время предсказания**: 10-30 минут
