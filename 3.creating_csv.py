# Импорт необходимых модулей
import pandas as pd  # для работы с данными
from PIL import Image  # для обработки изображений
import os  # для работы с файловой системой

# Задаем путь к директории, где находятся обрезанные изображения
NEW_DATA_DIR = './cropped_imgs'

# Задаем размер изображений руки
image_width = 128
image_height = 128

# Создаем список для хранения меток и пикселей изображений
data = []

for dir_ in os.listdir(NEW_DATA_DIR):
    # Проходим по каждому файлу в папке
    for filename in os.listdir(os.path.join(NEW_DATA_DIR, dir_)):
        if filename.endswith('.jpg'):
            # Извлекаем метку из названия файла
            label = int(dir_.split(".")[0])
            # Открываем изображение и приводим его к заданному размеру
            img = Image.open(os.path.join(NEW_DATA_DIR, dir_, filename)).resize((image_width, image_height))
            # Преобразуем изображение в массив пикселей
            pixels = list(img.getdata())
            # Преобразуем кортежи из трех чисел в одно число яркости
            pixels = [int(0.2989 * r + 0.5870 * g + 0.1140 * b) for r, g, b in pixels]
            # Добавляем метку и пиксели в список данных
            data.append((label, *pixels))

# Создаем DataFrame из списка данных
df = pd.DataFrame(data, columns=['label'] + [f'pixel{i + 1}' for i in range(image_width * image_height)])

# Сохраняем DataFrame в csv файл
df.to_csv('dataset.csv', index=False)
