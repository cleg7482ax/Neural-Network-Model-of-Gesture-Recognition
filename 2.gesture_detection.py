# Импорт необходимых модулей
import mediapipe as mp  # для работы с рукой и обнаружения ключевых точек
import cv2  # для обработки изображений и работы с файлами
import os  # для работы с файловой системой

# Инициализация модели для обнаружения ключевых точек на руке
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Указываем путь к исходной директории с собранными изображениями
DATA_DIR = './data'
# и директории, куда будут сохранены обрезанные изображения
NEW_DATA_DIR = './cropped_imgs'

number_of_classes = 10  # Количество классов жестов

if not os.path.exists(NEW_DATA_DIR):  # Проверяется наличие директории
    os.makedirs(NEW_DATA_DIR)  # если ее нет, она создается.

for j in range(number_of_classes):  # Для каждого класса жестов выполняется следующее:
    # Проверяется наличие директории для данного класса
    if not os.path.exists(os.path.join(NEW_DATA_DIR, str(j))):
        os.makedirs(os.path.join(NEW_DATA_DIR, str(j)))  # если ее нет, она создается

for dir_ in os.listdir(DATA_DIR):  # Для каждого изображения выполняется следующее:
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

        # Изображение конвертируется из формата BGR в RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Запускается обработка изображения с использованием модели
        results = hands.process(img_rgb)

        # Если обнаружены ключевые точки рук на изображении, выполняется следующее:
        if results.multi_hand_landmarks:
            # Для каждой обнаруженной руки на изображении выполняется следующее:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                # Определяются минимальные и максимальные координаты по осям x и y с использованием ключевых точек руки
                x_min = int(min([landmark.x * img_rgb.shape[1] for landmark in hand_landmarks.landmark]))
                y_min = int(min([landmark.y * img_rgb.shape[0] for landmark in hand_landmarks.landmark]))
                x_max = int(max([landmark.x * img_rgb.shape[1] for landmark in hand_landmarks.landmark]))
                y_max = int(max([landmark.y * img_rgb.shape[0] for landmark in hand_landmarks.landmark]))
                # Создается обрезанное изображение
                cropped_image = img_rgb[y_min:y_max, x_min:x_max]
                # Обрезанное изображение сохраняется в файл
                cv2.imwrite(os.path.join(NEW_DATA_DIR, dir_, img_path), cropped_image)

hands.close()  # Модель закрывается
