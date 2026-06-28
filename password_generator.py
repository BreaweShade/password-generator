
"""
Генератор паролей с оценкой сложности на основе машинного обучения
Графический интерфейс: Tkinter
Автор: Размазина Мария Андреевна
Программа обучения: Python-разработчик с использованием инструментов ИИ
Период обучения: 25.05.2026-29.06.2026
Дата проведения итоговой аттестации: 28.06.2026
Поддержка реального датасета rockyou.txt
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
import string
import re
import math
import json
import os
import hashlib
from datetime import datetime
from collections import Counter

# Для машинного обучения
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # для сохранения/загрузки модели

# Для шифрования (сохранение паролей)
from cryptography.fernet import Fernet

 
# 1. Функции для извлечения признаков из пароля
 

def entropy(password: str) -> float:
    """Вычисление энтропии Шеннона (бит на символ)"""
    if not password:
        return 0.0
    prob = [float(password.count(c)) / len(password) for c in set(password)]
    return -sum(p * math.log2(p) for p in prob)

def count_classes(password: str) -> int:
    """Количество типов символов: цифры, строчные, заглавные, спецсимволы"""
    has_digit = any(c.isdigit() for c in password)
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_special = any(not c.isalnum() for c in password)
    return sum([has_digit, has_lower, has_upper, has_special])

def has_sequence(password: str) -> int:
    """Проверка наличия клавиатурных последовательностей (qwerty, 123, abc)"""
    sequences = [
        'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
        '1234567890', 'abcdefghijklmnopqrstuvwxyz',
        'йцукенгшщзхъ', 'фывапролджэ', 'ячсмитьбю'
    ]
    lower_pwd = password.lower()
    for seq in sequences:
        for i in range(len(seq)-2):
            sub = seq[i:i+3]
            if sub in lower_pwd:
                return 1
    return 0

def repeat_ratio(password: str) -> float:
    """Доля повторяющихся символов (отношение уникальных к длине)"""
    if not password:
        return 0.0
    return len(set(password)) / len(password)

def extract_features(password: str) -> np.ndarray:
    """
    Извлекает вектор признаков для пароля:
    - длина
    - количество цифр
    - количество строчных
    - количество заглавных
    - количество спецсимволов
    - энтропия
    - количество классов символов
    - есть ли последовательность (0/1)
    - доля уникальных символов
    """
    length = len(password)
    digits = sum(c.isdigit() for c in password)
    lower = sum(c.islower() for c in password)
    upper = sum(c.isupper() for c in password)
    special = sum(not c.isalnum() for c in password)
    ent = entropy(password)
    classes = count_classes(password)
    seq = has_sequence(password)
    uniq_ratio = repeat_ratio(password)

    return np.array([length, digits, lower, upper, special, ent, classes, seq, uniq_ratio], dtype=float)

 
# 2. Обучение модели (с поддержкой rockyou.txt)
 

MODEL_PATH = "password_strength_model.pkl"
ROCKYOU_PATH = "rockyou.txt"   # имя файла датасета

def load_rockyou_passwords(max_lines=10000):
    """
    Загружает пароли из файла rockyou.txt (если он существует).
    Возвращает список строк (паролей), очищенных от пробелов.
    """
    if not os.path.exists(ROCKYOU_PATH):
        return []
    passwords = []
    try:
        with open(ROCKYOU_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                pwd = line.strip()
                if pwd:  # не пустая строка
                    passwords.append(pwd)
        print(f"Загружено {len(passwords)} паролей из rockyou.txt")
    except Exception as e:
        print(f"Ошибка загрузки rockyou.txt: {e}")
    return passwords

def generate_synthetic_dataset(size=10000):
    """
    Создаёт синтетический набор данных для обучения.
    Если rockyou.txt найден, его пароли добавляются в список слабых.
    """
    # Базовый список самых частых паролей (топ-100)
    common_passwords = [
        "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234567",
        "111111", "1234567890", "123123", "000000", "12345678910",
        "qwertyuiop", "123321", "666666", "a123456", "123456789a", "qwerty123",
        "1q2w3e4r", "123qwe", "zaq12wsx", "password123", "admin", "letmein",
        "welcome", "monkey", "dragon", "master", "sunshine", "princess", "iloveyou",
        "fuckyou", "7777777", "password1", "555555"
    ]
    weak_list = common_passwords[:]

    # Добавляем пароли из rockyou.txt (первые 5000, чтобы не перегружать)
    rockyou_pwds = load_rockyou_passwords(max_lines=5000)
    if rockyou_pwds:
        # Берём только уникальные, чтобы не дублировать
        weak_list.extend([p for p in rockyou_pwds if p not in weak_list])

    # Ограничим общее количество слабых, чтобы сбалансировать классы
    if len(weak_list) > 5000:
        weak_list = weak_list[:5000]

    # Добавим ещё сгенерированные простые (короткие, только буквы или цифры)
    for _ in range(max(0, 5000 - len(weak_list))):
        length = random.randint(4, 6)
        if random.choice([True, False]):
            pwd = ''.join(random.choices(string.digits, k=length))
        else:
            pwd = ''.join(random.choices(string.ascii_lowercase, k=length))
        weak_list.append(pwd)

    weak_list = weak_list[:5000]  # фиксируем 5000 слабых

    # Генерация сильных паролей (длина 12-20, все типы символов)
    strong_list = []
    for _ in range(5000):
        length = random.randint(12, 20)
        pwd = []
        pwd.append(random.choice(string.ascii_lowercase))
        pwd.append(random.choice(string.ascii_uppercase))
        pwd.append(random.choice(string.digits))
        pwd.append(random.choice(string.punctuation))
        all_chars = string.ascii_letters + string.digits + string.punctuation
        for _ in range(length - 4):
            pwd.append(random.choice(all_chars))
        random.shuffle(pwd)
        strong_list.append(''.join(pwd))

    # Собираем данные
    passwords = weak_list + strong_list
    labels = [0] * len(weak_list) + [1] * len(strong_list)  # 0 - слабый, 1 - сильный

    # Извлекаем признаки
    X = np.array([extract_features(p) for p in passwords])
    y = np.array(labels)

    # Перемешиваем
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    print(f"Датасет сгенерирован: {len(X)} примеров, из них слабых: {sum(y==0)}, сильных: {sum(y==1)}")
    return X, y

def train_and_save_model():
    """Обучает модель и сохраняет в файл"""
    print("Обучение модели...")
    X, y = generate_synthetic_dataset(10000)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение случайного леса
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Точность на тесте: {acc:.3f}")

    # Сохраняем модель
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

def load_model():
    """Загружает модель, если есть, иначе обучает"""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Модель загружена из файла.")
    else:
        train_and_save_model()
        model = joblib.load(MODEL_PATH)
    return model

 
# 3. Функции генерации паролей и оценки
 

def generate_password(length=12, use_digits=True, use_special=True,
                     use_upper=True, use_lower=True, avoid_ambiguous=False):
    """
    Генерирует пароль по заданным параметрам.
    Если avoid_ambiguous=True, исключает символы 'il1Lo0O'
    """
    chars = ''
    if use_lower:
        chars += string.ascii_lowercase
    if use_upper:
        chars += string.ascii_uppercase
    if use_digits:
        chars += string.digits
    if use_special:
        chars += string.punctuation

    if not chars:
        raise ValueError("Не выбрано ни одного типа символов!")

    if avoid_ambiguous:
        ambiguous = 'il1Lo0O'
        chars = ''.join(c for c in chars if c not in ambiguous)

    password = ''.join(random.choice(chars) for _ in range(length))
    return password

def evaluate_password(password, model):
    """
    Оценивает сложность пароля с помощью модели.
    Возвращает вероятность принадлежности к классу "сильный" и текстовую оценку.
    """
    features = extract_features(password).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]  # вероятность класса 1 (сильный)
    if prob >= 0.7:
        strength = "Очень надёжный"
    elif prob >= 0.5:
        strength = "Средней надёжности"
    elif prob >= 0.3:
        strength = "Слабый"
    else:
        strength = "Очень слабый"
    return prob, strength

 
# 4. Шифрование и сохранение истории
 

def get_cipher():
    """Генерирует или загружает ключ шифрования"""
    key_file = "secret.key"
    if os.path.exists(key_file):
        with open(key_file, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
    return Fernet(key)

def save_password_record(password, strength, prob, filename="history.json"):
    """
    Сохраняет запись о сгенерированном пароле в зашифрованном виде.
    Пароль шифруется, остальные поля – в открытом виде.
    """
    cipher = get_cipher()
    encrypted_pwd = cipher.encrypt(password.encode())

    record = {
        "timestamp": datetime.now().isoformat(),
        "encrypted_password": encrypted_pwd.decode(),
        "strength": strength,
        "probability": float(prob)
    }

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []

    history.append(record)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_history(filename="history.json"):
    """Загружает историю и расшифровывает пароли"""
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except:
            return []

    cipher = get_cipher()
    decrypted = []
    for rec in history:
        try:
            pwd_bytes = rec["encrypted_password"].encode()
            decrypted_pwd = cipher.decrypt(pwd_bytes).decode()
            decrypted.append({
                "timestamp": rec["timestamp"],
                "password": decrypted_pwd,
                "strength": rec["strength"],
                "probability": rec["probability"]
            })
        except Exception as e:
            continue
    return decrypted

 
# 5. Графический интерфейс на Tkinter
 

class PasswordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Генератор паролей с оценкой надёжности")
        self.root.geometry("700x650")
        self.root.resizable(False, False)

        self.model = load_model()

        self.length_var = tk.IntVar(value=14)
        self.use_digits = tk.BooleanVar(value=True)
        self.use_special = tk.BooleanVar(value=True)
        self.use_upper = tk.BooleanVar(value=True)
        self.use_lower = tk.BooleanVar(value=True)
        self.avoid_ambiguous = tk.BooleanVar(value=False)

        self.password_var = tk.StringVar()

        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Настройки генерации", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Label(main_frame, text="Длина:").grid(row=1, column=0, sticky="w", pady=5)
        length_scale = ttk.Scale(main_frame, from_=6, to=30, orient=tk.HORIZONTAL,
                                 variable=self.length_var, length=200)
        length_scale.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(main_frame, textvariable=self.length_var, width=5).grid(row=1, column=2, sticky="w")

        ttk.Checkbutton(main_frame, text="Использовать цифры", variable=self.use_digits).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(main_frame, text="Использовать спецсимволы", variable=self.use_special).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(main_frame, text="Заглавные буквы", variable=self.use_upper).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(main_frame, text="Строчные буквы", variable=self.use_lower).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(main_frame, text="Исключать неоднозначные (il1Lo0O)", variable=self.avoid_ambiguous).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Button(main_frame, text="Сгенерировать пароль", command=self.generate).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

        ttk.Label(main_frame, text="Сгенерированный пароль:", font=("Arial", 11)).grid(row=6, column=0, sticky="w", pady=5)
        password_entry = ttk.Entry(main_frame, textvariable=self.password_var, font=("Courier", 14), width=40)
        password_entry.grid(row=6, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Button(main_frame, text="Копировать", command=self.copy_to_clipboard).grid(row=6, column=3, padx=5)

        self.strength_label = ttk.Label(main_frame, text="Надёжность: ---", font=("Arial", 12))
        self.strength_label.grid(row=7, column=0, columnspan=4, sticky="w", pady=5)

        ttk.Button(main_frame, text="Сохранить в историю", command=self.save_current).grid(row=8, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Показать историю", command=self.show_history).grid(row=8, column=2, columnspan=2, pady=5, sticky="ew")

        ttk.Label(main_frame, text="Статус:", font=("Arial", 10)).grid(row=9, column=0, sticky="w", pady=5)
        self.status_text = scrolledtext.ScrolledText(main_frame, height=6, width=80, state=tk.NORMAL)
        self.status_text.grid(row=10, column=0, columnspan=4, pady=5, sticky="ew")
        self.status_text.config(state=tk.DISABLED)

        self.current_password = ""
        self.current_strength = ""
        self.current_prob = 0.0

    def log_message(self, msg):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def generate(self):
        try:
            pwd = generate_password(
                length=self.length_var.get(),
                use_digits=self.use_digits.get(),
                use_special=self.use_special.get(),
                use_upper=self.use_upper.get(),
                use_lower=self.use_lower.get(),
                avoid_ambiguous=self.avoid_ambiguous.get()
            )
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.current_password = pwd
        self.password_var.set(pwd)

        prob, strength = evaluate_password(pwd, self.model)
        self.current_strength = strength
        self.current_prob = prob

        color = "green" if prob >= 0.7 else "orange" if prob >= 0.5 else "red"
        self.strength_label.config(text=f"Надёжность: {strength} (вероятность {prob:.2f})", foreground=color)

        self.log_message(f"Сгенерирован пароль: {pwd} | Оценка: {strength} ({prob:.2f})")

    def copy_to_clipboard(self):
        pwd = self.password_var.get()
        if pwd:
            self.root.clipboard_clear()
            self.root.clipboard_append(pwd)
            self.log_message("Пароль скопирован в буфер обмена")
        else:
            messagebox.showinfo("Информация", "Сначала сгенерируйте пароль")

    def save_current(self):
        if not self.current_password:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте пароль")
            return

        save_password_record(self.current_password, self.current_strength, self.current_prob)
        self.log_message(f"Пароль сохранён в историю (зашифрован)")

    def show_history(self):
        history = load_history()
        if not history:
            messagebox.showinfo("История", "История пуста")
            return

        hist_win = tk.Toplevel(self.root)
        hist_win.title("История паролей")
        hist_win.geometry("600x400")

        frame = ttk.Frame(hist_win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, yscrollcommand=scroll.set, font=("Courier", 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=listbox.yview)

        for rec in history:
            dt = rec["timestamp"][:19]
            pwd = rec["password"]
            strength = rec["strength"]
            prob = rec["probability"]
            listbox.insert(tk.END, f"{dt} | {pwd} | {strength} ({prob:.2f})")

        ttk.Button(hist_win, text="Закрыть", command=hist_win.destroy).pack(pady=5)

 
# 6. Запуск приложения
 

if __name__ == "__main__":
    root = tk.Tk()
    app = PasswordApp(root)
    root.mainloop()