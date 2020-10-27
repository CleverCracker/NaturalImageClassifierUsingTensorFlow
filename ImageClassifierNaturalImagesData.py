# * Import Libraries
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import tensorflow as tf
import numpy as np
import os

# * File Directory Info
natural_images_dir = './natural_images/'
labels = os.listdir(natural_images_dir)

# * Fetching Data of Image
x_data = []
y_data = []
for label in labels:
    path = natural_images_dir + label + '/'
    folder_images = os.listdir(path)
    for image_path in folder_images:
        image = cv2.imread(path+image_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resized = cv2.resize(image, (32, 32))

        x_data.append(image_resized)
        y_data.append(label)

# * Image Preprocessing
x_data = np.array(x_data)
y_data = np.array(y_data)
print('The Shape X = ', x_data.shape, 'Shape Y = ', y_data.shape)

x_data = x_data.astype('float32')/255

# Converting Label in Categorical Data
y_encoded = LabelEncoder().fit_transform(y_data)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# * Spliting The Data and Shuffling..
# Shuffle the Data
r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_categorical[r]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# * Structuring Keras Model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
    5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
    3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(8,   activation='softmax'))

# Model Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=25, validation_split=0.2)

# Model By Clever Cracker
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=32, kernel_size=(
            4, 4), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(
            4, 4), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

y_pred = model.predict_classes(x_test)
y_test = np.argmax(y_test, axis=1)
print('Accracy = ', accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
