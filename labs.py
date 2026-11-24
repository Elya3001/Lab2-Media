import os
import cv2
import numpy
from datetime import datetime

BASE = "Lab2_Media" # текущая папка
INPUT_DIR = os.path.join(".", "data", "input")
OUTPUT_DIR = os.path.join(".", "data", "output")
LOG_FILE = os.path.join(".", "logs", "history.txt")
FILTER_NAME = "Свёртка + Выравнивание гистограммы"

# 1. Ввод имени файла
filename = input("Введите имя файла для обработки (например, image1.jpg): ").strip()
path = os.path.join(INPUT_DIR, filename)

def write_log(status):
    """Дописываем строку в журнал в конец, сохраняя историю (режим 'a')"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) # создает logs, если ее нет
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    # время и дата
    line = f"[{now}] Файл: {filename} | Фильтр: {FILTER_NAME} | Статус: {status}\n"
    with open(LOG_FILE, "a", encoding="UTF-8") as f:
        f.write(line)

try:
    # 2. Поиск файла + 3. обработка ошибок доступа
    if not os.path.exists(path):
        print("Ошибка: Файл не найден")
        write_log("Ошибка: файл не найден")
    else:
        img = cv2.imread(path)          # может бросить PermissionError/IOError
        if img is None:
            raise PermissionError()     # imread вернул None — нет прав на чтение

        # 4. Обработка: свёртка (ядро Гаусса) + выравнивание гистограммы
        kernel = numpy.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=numpy.float32) / 16.0
        conv = cv2.filter2D(img, -1, kernel)       # свёртка

        gray = cv2.cvtColor(conv, cv2.COLOR_BGR2GRAY) # переводим в чб для equalizeHist
        eq = cv2.equalizeHist(gray)                # выравнивание гистограммы

        # 5. Сохранение
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(BASE, "data", "output", "result.jpg")
        cv2.imwrite(out_path, eq)   # сохраняет картинку по указанному пути
        print(f"Готово! Результат сохранён в {out_path}")
        write_log("Успешно")

except PermissionError:
    print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
    write_log("Ошибка доступа")
except Exception as e:
    print(f"Ошибка: {e}")
    write_log(f"Ошибка: {e}")
