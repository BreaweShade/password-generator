# -*- coding: utf-8 -*-
"""
Скрипт для визуализации важности признаков модели Random Forest.
Сохраняет график и показывает его на экране.
Запускать после того, как основная программа обучила модель.
"""

import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

MODEL_PATH = "password_strength_model.pkl"

# Проверяем, существует ли файл модели
if not os.path.exists(MODEL_PATH):
    print("Файл модели не найден! Сначала запустите password_generator.py для обучения.")
    exit(1)

# Загружаем модель
model = joblib.load(MODEL_PATH)

# Получаем важность признаков (массив чисел)
importance = model.feature_importances_

# Названия признаков (в том же порядке, как в extract_features)
features = [
    'Длина',
    'Цифры',
    'Строчные',
    'Заглавные',
    'Спецсимволы',
    'Энтропия',
    'Классов',
    'Последовательность',
    'Уникальность'
]

# Сортируем по убыванию важности
indices = np.argsort(importance)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importance = importance[indices]

# Вывод в консоль числовых значений (для отчёта)
print("Важность признаков (в порядке убывания):")
for name, imp in zip(sorted_features, sorted_importance):
    print(f"{name}: {imp:.3f}")

# Построение графика
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_importance, color='skyblue')
plt.xlabel('Важность (importance)', fontsize=12)
plt.title('Важность признаков для оценки надёжности пароля\n(модель Random Forest)', fontsize=14)
plt.gca().invert_yaxis()  # самый важный сверху

# Подписываем значения справа от столбцов
for bar, val in zip(bars, sorted_importance):
    plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()

# Сохраняем график в файл (опционально)
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("График сохранён в файл feature_importance.png")

# Показываем график на экране
plt.show()