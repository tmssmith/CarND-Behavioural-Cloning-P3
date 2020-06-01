import csv
import tensorflow as tf
import cv2
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

### Data preparation

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip headers
    for line in reader:
        samples.append(line)

# Split samples into training and validation
train_data, valid_data = train_test_split(samples, test_size=0.2)

# Generator to prepare batches of images as required without loading all images into memory
def generator(samples, batch_size=32):
    num_samples = len(samples)    
    steer_correction = [0, 0.35, -0.35]  # offsets for center left and right camera angles
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            frames = []
            angles = []
            
            for batch_sample in batch_samples:
                for cam in range(3):  # Take images from 3 cameras for each sample
                    frame_path = './data/' + batch_sample[cam].strip()
                    frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
                    frame_flip = np.fliplr(frame)
                    frames.extend((frame, frame_flip))
                    angle = float(batch_sample[3]) + steer_correction[cam]
                    angle_flip = -angle
                    angles.extend((angle, angle_flip))

            X = np.array(frames)
            y = np.array(angles)
            
            yield shuffle(X, y)

batch_size = 32
train_gen = generator(train_data, batch_size)
valid_gen = generator(valid_data, batch_size)

### Build NVIDIA model
model = Sequential()
# Normalise inputs to 0 - 1
model.add(Lambda(lambda x: x/255.0, input_shape=(160,320,3)))
# Center mean (roughly)
model.add(Lambda(lambda x: x - 0.5))
# Crop image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Resize image to 66 x 200 as per NVIDIA paper
model.add(Lambda(lambda image: tf.image.resize_images( 
    image, (66, 200))))
# Dropout on inputs to reduce overfit
model.add(Dropout(0.3))
# 3x convolutional layers: k = 5x5, s = 2x2
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.1))
# 2x convolutional layers: k = 3x3, s = 1x1
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Dropout(0.1))
# 3x dense layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

### Train model

epochs = 2

model.fit_generator(train_gen, steps_per_epoch=len(train_data)*6/batch_size, epochs=epochs, 
                    validation_data=valid_gen, validation_steps=len(valid_data)*6/batch_size)

model.save('model_nvidia_drop3.h5')