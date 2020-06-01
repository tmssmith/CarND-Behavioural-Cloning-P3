import csv
import tensorflow as tf
import cv2
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

log = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip headers
    for line in reader:
        log.append(line)

frames = []
angles = []

for line in log:
    frame_path = './data/' + line[0]
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
    angles.append(float(line[3]))
    
X = np.array(frames)
y = np.array(angles)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X, y, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')


    