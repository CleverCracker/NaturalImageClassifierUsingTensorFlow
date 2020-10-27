# * Import Libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import tensorflow as tf
from IPython.display import Image, display
import numpy as np
import os
from PIL import ImageFile
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

natural_images_dir = './natural_images/'

labels = os.listdir(natural_images_dir)
print(labels)

num = []
for label in labels:
    path = natural_images_dir + label + '/'
    folder_images = os.listdir(path)
    print('', label.upper())
    display(Image(path+folder_images[0]))
    print(label.upper()+' Contains '+str(folder_images.__len__())+' Images...')
    num.append(folder_images.__len__())


plt.figure(figsize=(8, 8))
plt.bar(labels, num)
plt.show()

x_data = []
y_data = []
for label in labels:
    path = natural_images_dir + label + '/'
    folder_images = os.listdir(path)
    for image_path in folder_images:
        image = cv2.imread(path+image_path)
        image_resized = cv2.resize(image, (32, 32))
        x_data.append(image_resized)
        y_data.append(label)
y_data.__len__()

x_data = np.array(x_data)
y_data = np.array(y_data)
print('The Shape X = ', x_data.shape, 'Shape Y = ', y_data.shape)

x_data = x_data.astype('float32')/255
x_data

# Converting Label in Categorical Data
y_encoded = LabelEncoder().fit_transform(y_data)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Shuffle the Data
r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_encoded[r]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Structuring Keras Model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
    5, 5), activation='relu', input_shape=x_train.shpae[1:]))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
    3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(8, activation='softmax'))

print()
