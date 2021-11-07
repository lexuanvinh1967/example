import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint  
import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPool2D,MaxPooling2D,LSTM
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Dropout,ELU
from keras.models import Model

X_train=[]
y_train=[]

X_test=[]
y_test=[]

width,height=256,256
batch_size = 32

input_shapes = (256,256,1)
n_classes = 5

data_folder = "ecg_img"
folder_test = "ecg_img/test"
folder_train = "ecg_img/train"

for folder_tra in os.listdir(folder_train):
    curr_path_train = os.path.join(folder_train, folder_tra)
    for file_train in os.listdir(curr_path_train):
        curr_path_train2 = os.path.join(curr_path_train, file_train)
        for file_train2 in os.listdir(curr_path_train2):
            curr_file1= os.path.join(curr_path_train2, file_train2)
            images1 = cv2.imread(curr_file1,cv2.IMREAD_GRAYSCALE)
            new_images1 = cv2.resize(images1,(width,height),interpolation=cv2.INTER_LANCZOS4)
            X_train.append(new_images1)
            y_train.append(folder_tra)
            
for folder_tes in os.listdir(folder_test):
    curr_path1 = os.path.join(folder_test, folder_tes)
    for file1 in os.listdir(curr_path1):
        curr_path2 = os.path.join(curr_path1, file1)
        for file2 in os.listdir(curr_path2):
            curr_file2 = os.path.join(curr_path2, file2)
            images2 = cv2.imread(curr_file2,cv2.IMREAD_GRAYSCALE)
            new_images2 = cv2.resize(images2,(width,height),interpolation=cv2.INTER_LANCZOS4)
            X_test.append(new_images2)
            y_test.append(folder_tes)   

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train1:",X_train.shape)    
plt.imshow(X_train[1])

X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)

print("X_train2:",X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.dtype)

X_train /= 255
X_test /= 255

print("Reshape:",X_test[0].shape)

encoder = LabelBinarizer()
y_train =encoder.fit_transform(y_train)
y_test =encoder.fit_transform(y_test)

# # y_train = np_utils.to_categorical(y_train,n_classes)
# # y_test = np_utils.to_categorical(y_test,n_classes)

print(y_train.shape)        

for (i,lab) in enumerate(encoder.classes_):
    print("{}.{}".format(i+1,lab))
 
model = Sequential()
#64 conv
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shapes, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#128 conv
model.add(Conv2D(128, (3, 3), activation='relu', padding='same' ))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#256 conv
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#Dense part
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  

print(model.summary())

filepath="best_weight/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=False,
                             period=1,verbose=1,save_best_only=False)
callbacks_list = [checkpoint]

model.fit(
  X_train,y_train,
  validation_data=(X_test,y_test),
  epochs=1,verbose=2,shuffle=False, 
  batch_size=batch_size,
  callbacks=callbacks_list
)
model.save("model_SVnghia.h5")
 
predictions = model.predict(X_test)
score = accuracy_score(change(y_test), change(predictions))
print(score)
    


















# inputs = Input(input_shapes)

#     # layer 1
# x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(inputs)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer 2
# x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer3
# x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

#     # layer4
# x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer5
# x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer6
# x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

#     # layer7
# x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer 8
# x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
# x = ELU()(x)
# x = BatchNormalization()(x)

#     # layer 9
# x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

# x = Flatten()(x)

#     # layer 10
# x = Dense(2048)(x)
# x = ELU()(x)
# x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
# x = Dense(n_classes, activation='softmax')(x)

# model = Model(inputs, x)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

# print(model.summary())

# filepath="best_weight/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# model.fit(
#   X_train,y_train,
#   validation_data=(X_test,y_test),
#   epochs=30, 
#   batch_size=batch_size,
#   callbacks=callbacks_list
# )
# model.save("models/model_SV.h5")
 
# predictions = model.predict(X_test)
# score = accuracy_score(change(y_test), change(predictions))
# print(score)
    

