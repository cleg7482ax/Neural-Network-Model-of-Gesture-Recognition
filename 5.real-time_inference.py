# Импорт необходимых модулей
import cv2  # для работы с видео и изображениями
import mediapipe as mp  # для обработки руки
import tensorflow as tf  # для работы с нейросетями
import numpy as np  # для работы с массивами

# Загружается ранее обученная модель
# model = tf.keras.models.load_model('model1.h5')
model = tf.keras.models.load_model('model2.h5')

# Задаем размер изображений руки
image_width = 128
image_height = 128

# Инициализируется захват видео
cap = cv2.VideoCapture(0)

# Инициализируются объекты для обработки рук и рисования на кадре
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Создается словарь для соответствия меток жестов и их названий
labels_dict = {0: 'кулак', 1: 'опущенный большой палец',
               2: 'сжатие всех пальцев', 3: 'Поднятая ладонь',
               4: 'ок', 5: 'Верхний указательный палец',
               6: 'Нижний указательный палец', 7: 'Указательный палец влево',
               8: 'Указательный палец вправо', 9: 'Опущенная ладонь'}

# В цикле while читается каждый кадр из видео
while True:
    ret, frame = cap.read()

    # Инвертируем изображение по горизонтали
    frame = cv2.flip(frame, 1)

    # Кадр переводится из формата BGR в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка кадра для обнаружения рук
    results = hands.process(frame_rgb)

    # Если руки обнаружены, происходит рисование на кадре
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Вычисляются координаты прямоугольника, ограничивающего руку, на основе landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame_rgb.shape[1]), \
                int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame_rgb.shape[0])
            x_max, y_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame_rgb.shape[1]), \
                int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame_rgb.shape[0])

        # Обрезка и изменение размера изображения
        cropped_image = frame[y_min-30:y_max+30, x_min-30:x_max+30]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (image_width, image_height))

        # Преобразование изображения в массив numpy
        img = np.array(resized_image).reshape(-1, image_width, image_height, 1)

        # Нормализация изображения
        img = img / 255.0

        # Выполнение предсказания с помощью модели
        prediction = model.predict(img)

        # Получение индекса метки с наибольшей вероятностью
        top_prediction = np.argmax(prediction)

        # Вывод названия метки на экран
        label = labels_dict[top_prediction]

        cv2.rectangle(frame, (x_min - 30, y_min - 30), (x_max + 30, y_max + 30), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (100, 0, 100), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recognition', frame)
    cv2.waitKey(1)

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
