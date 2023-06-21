# Импорт необходимых модулей
import os  # для работы с файловой системой
import cv2  # для обработки изображений и работы с камерой

# Указываем путь к директории, где будут сохраняться изображения
DATA_DIR = './data'

number_of_classes = 10   # Количество классов жестов
dataset_size = 500  # Размер набора данных для каждого класса

# Создаем объект cap для работы с камерой (индекс 0 соответствует основной камере компьютера)
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):  # Для каждого класса жестов выполняется следующее:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):  # Проверяется наличие директории для данного класса
        os.makedirs(os.path.join(DATA_DIR, str(j)))  # если ее нет, она создается

    # Выводим сообщение о начале сбора изображений для текущего класса
    print('Collecting images for {}'.format(j))

    # Запускаем цикл, который ожидает нажатие клавиши 'q' для сбора изображений по каждому классу
    while True:
        # Захватываем текущий кадр с камеры
        ret, frame = cap.read()
        # Инвертируем изображение по горизонтали
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Ready? Press "Q" !', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Кадр отображается в окне с названием 'frame'
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    # Запускаем цикл для сбора указанного количества изображений
    while counter < dataset_size:
        # Захватываем текущий кадр с камеры
        ret, frame = cap.read()
        # Инвертируем изображение по горизонтали
        frame = cv2.flip(frame, 1)
        # Кадр отображается в окне с названием 'frame'
        cv2.imshow('frame', frame)
        # Захватываем новый кадр с окном в 25 миллисекунд
        cv2.waitKey(25)
        # Захваченный кадр сохраняем в файл в указанной директории.
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1  # Счетчик изображений увеличивается

cap.release()  # Освобождаем ресурс камеры
cv2.destroyAllWindows()  # Закрываем открытые окна
