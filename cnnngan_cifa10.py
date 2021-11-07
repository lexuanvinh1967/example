#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import matplotlib.image as mp
import numpy as np
from tensorflow.keras import datasets, layers, models

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i][0]])
plt.show()
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3),activation = 'relu',padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))


    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model
model=cnn_model()


opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model_detail=  model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test),
              verbose=1)
   
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=0)
print (scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1]*100)


# In[ ]: Thử

# =============================================================================
# plt.imshow(x_test[0].reshape(32,32,3))
# y_predict = model.predict(x_test[0].reshape(1,32,32,3))
# print('Giá trị dự đoán: ', class_names[np.argmax(y_predict)])
# =============================================================================

# =============================================================================

# =============================================================================
# img = mp.imread('hau.jpg')
# 
# #plt.imshow(img)
# print(img.shape)
# img1=np.array(img)
# img1.resize(32,32,3)
# print(img1.shape)
# y_predict = model.predict(img1.reshape(1,32,32,3))
# print('Giá trị dự đoán: ', np.argmax(y_predict))
# =============================================================================

#plt.imshow(img1)
#plt.show
# 
# y_predict = model.predict(X_test[0].reshape(1,28,28,1))
# print('Giá trị dự đoán: ', np.argmax(y_predict))
# =============================================================================
from skimage.transform import resize
img = mp.imread('xehoi.jpg')
img1=resize(img,(32,32))
print(img1.shape)
y_predict = model.predict(img1.reshape(1,32,32,3))
print('Giá trị dự đoán: ', class_names[np.argmax(y_predict)])
#bottle_resized = resize(bottle, (140, 54))