# Импорт необходимых модулей
import pandas as pd  # для работы с данными
from sklearn.model_selection import train_test_split  # для разделения данных на обучающую и тестовую выборки
from tensorflow import keras  # для работы с нейросетями
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Задаем размер изображений руки
image_width = 128
image_height = 128

number_of_classes = 10  # Количество классов жестов

# Разделение на метки и пиксели
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Приведение яркости к диапазону от 0 до 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Преобразование каждого жеста в матрицу пикселей
X_train = X_train.reshape(-1, image_width, image_height, 1)
X_test = X_test.reshape(-1, image_width, image_height, 1)

# Создание объекта для аугментации данных
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

# Определение архитектуры нейросети
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(number_of_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Увеличение размера обучающей выборки с помощью аугментации данных
augmented_train_data = datagen.flow(X_train, y_train, batch_size=32)

# Обучение модели
history = model.fit(augmented_train_data, epochs=20, validation_data=(X_test, y_test))

# Оценка качества работы модели на тестовой выборке
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nТочность на тестовых данных:', test_acc)

# Получение предсказаний модели
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанный класс')
plt.ylabel('Фактический класс')
plt.show()

# Визуализация нескольких случайных тестовых изображений и их предсказанных классов
num_images = 5
random_indices = np.random.randint(0, len(X_test), num_images)

plt.figure(figsize=(15, 3))
for i, index in enumerate(random_indices):
    image = X_test[index].reshape(image_width, image_height)
    plt.subplot(1, num_images, i+1)
    plt.imshow(image, cmap='gray')
    plt.title('Предсказано: {}'.format(y_pred[index]))
    plt.axis('off')
plt.show()

# Визуализация результатов обучения модели
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title('Зависимость функции потерь от эпохи')
plt.xlabel('Эпоха')
plt.ylabel('Функция потерь')
plt.legend(['Обучающая выборка', 'Тестовая выборка'])

history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.title('Зависимость точности от эпохи')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend(['Обучающая выборка', 'Тестовая выборка'])

plt.show()
# Сохранение модели
model.save('model2.h5')
