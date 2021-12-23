# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from utils import config

train = True
num_classes = 100
weight_decay = 0.0005
x_shape = [600, 600, 3]

# Implementation of VGG-16 CNN
model = Sequential()
weight_decay = weight_decay

# The First Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(
    Conv2D(64, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The Second Block
model.add(
    Conv2D(128, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(128, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The Third Block
model.add(
    Conv2D(256, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(256, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(256, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The 4th Block
model.add(
    Conv2D(512, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(512, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The 5th Block
model.add(
    Conv2D(512, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(512, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(
    Conv2D(512, (3, 3),
           padding='same',
           kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# The Final Block
model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.summary()

# # Training Process
# def normalize(X_train, X_test):
#     # this function normalize inputs for zero mean and unit variance
#     # it is used when training a model.
#     # Input: training set and test set
#     # Output: normalized training set and test set according to the training set statistics.
#     mean = np.mean(X_train, axis=(0, 1, 2, 3))
#     std = np.std(X_train, axis=(0, 1, 2, 3))
#     print(mean)
#     print(std)
#     X_train = (X_train - mean) / (std + 1e-7)
#     X_test = (X_test - mean) / (std + 1e-7)
#     return X_train, X_test
#
#
# # The data, shuffled and split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train, x_test = normalize(x_train, x_test)
#
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
#
# # Training parameters
# batch_size = 128
# max_epoch = 200
# learning_rate = 0.1
# lr_decay = 1e-6
# lr_drop = 20
#
#
# def lr_scheduler(epoch):
#     return learning_rate * (0.5 ** (epoch // lr_drop))
#
#
# reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
#
# # Data augmentation
# data_gen = ImageDataGenerator(
#     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
#
# # Optimization details
# sgd = optimizers.SGD(
#     learning_rate=learning_rate,
#     decay=lr_decay,
#     momentum=0.9,
#     nesterov=True)
#
# model.compile(
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     optimizer=sgd,
#     metrics=['accuracy'])
#
# # training process in a for loop with learning rate drop every 25 epochs.
# hist = model.fit(
#     data_gen.flow(x_train, y_train, batch_size=batch_size),
#     steps_per_epoch=x_train.shape[0] // batch_size,
#     epochs=max_epoch,
#     validation_data=(x_test, y_test),
#     callbacks=[reduce_lr],
#     verbose=1)
#
# print("Training Accuracy: %.2f%%" % (hist.history['accuracy'][max_epoch - 1] * 100))
# print("Testing Accuracy: %.2f%%" % (hist.history['val_accuracy'][max_epoch - 1] * 100))
#
# # Plot the result
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.legend(['train', 'test'])
# plt.title('loss')
# plt.savefig(os.path.sep.join([config.VGG_PLOT_PATH, "vgg_cifar100_scratch_loss.png"]), dpi=300, format="png")
# plt.figure()
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.legend(['train', 'test'])
# plt.title('accuracy')
# plt.savefig(os.path.sep.join([config.VGG_PLOT_PATH, "vgg_cifar100_scratch_accuracy.png"]), dpi=300, format="png")
#
# # Save model to disk
# print("Save model weights to disk")
# model.save_weights(os.path.sep.join([config.VGG_BASE_PATH, "cifar100vgg_scratch.h5"]))
# model_json = model.to_json()
# with open(os.path.sep.join([config.VGG_BASE_PATH, "cifar100vgg_scratch.json"]), "w") as json_file:
#     json_file.write(model_json)
