import csv
import tensorflow as tf
import cv2
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

### Data preparation

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip headers
    for line in reader:
        samples.append(line)

train_data, valid_data = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)    
    steer_correction = [0, 0.5, -0.5]
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
            X = preprocess_input(X)
            y = np.array(angles)
            
            yield shuffle(X, y)

batch_size = 32
train_gen = generator(train_data, batch_size)
valid_gen = generator(valid_data, batch_size)

### Model
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None

mobilenet = MobileNetV2(weights=weights_flag, include_top=False, input_shape=(224,224,3), pooling='avg')

if freeze_flag == True:
    for layer in mobilenet.layers:
        layer.trainable = False
cam_input = Input(shape=(160,320,3))
cropped = Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3))(cam_input)
resized = Lambda(lambda image: tf.image.resize_images( 
    image, (50, 50)))(cropped)
inp = mobilenet(resized)
x = Dense(256, activation='relu')(inp)
x = Dropout(0.5)(x)
predictions = Dense(1)(x)
model = Model(inputs=cam_input, outputs=predictions)
model.compile(loss='mse', optimizer='adam')

### Train model

epochs = 6

model.fit_generator(train_gen, steps_per_epoch=len(train_data)*6/batch_size, epochs=epochs, 
                    validation_data=valid_gen, validation_steps=len(valid_data)*6/batch_size)

model.save('model_mobile.h5')


    