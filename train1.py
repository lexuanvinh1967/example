from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.models import  load_model

# Khai báo
X_train=[]
y_train=[]

X_test=[]
y_test=[]

width,height=128,128
batch_size = 64

data_folder = "ecg_img"
folder_test = "ecg_img/test"
folder_train = "ecg_img/train"

for folder_tra in os.listdir(folder_train):
    curr_path_train = os.path.join(folder_train, folder_tra)
    for file_train in os.listdir(curr_path_train):
        curr_path_train2 = os.path.join(curr_path_train, file_train)
        for file_train2 in os.listdir(curr_path_train2):
            curr_file1= os.path.join(curr_path_train2, file_train2)
            images1 = cv2.imread(curr_file1)
            new_images1 = cv2.resize(images1,(width,height),interpolation=cv2.INTER_LANCZOS4)
            X_train.append(new_images1)
            y_train.append(folder_tra)
            
for folder_tes in os.listdir(folder_test):
    curr_path1 = os.path.join(folder_test, folder_tes)
    for file1 in os.listdir(curr_path1):
        curr_path2 = os.path.join(curr_path1, file1)
        for file2 in os.listdir(curr_path2):
            curr_file2 = os.path.join(curr_path2, file2)
            images2 = cv2.imread(curr_file2)
            new_images2 = cv2.resize(images2,(width,height),interpolation=cv2.INTER_LANCZOS4)
            X_test.append(new_images2)
            y_test.append(folder_tes)   

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train1:",X_train.shape)    
plt.imshow(X_train[1])

#X_train = X_train.reshape(X_train.shape[0], width, height, 1)
#X_test = X_test.reshape(X_test.shape[0], width, height, 1)

print("X_train2:",X_train.shape)

# =============================================================================
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# =============================================================================

print(X_train.dtype)

#X_train /= 255
#X_test /= 255

print("Reshape:",X_test[0].shape)

encoder = LabelBinarizer()
y_train =encoder.fit_transform(y_train)
y_test =encoder.fit_transform(y_test)

# # y_train = np_utils.to_categorical(y_train,n_classes)
# # y_test = np_utils.to_categorical(y_test,n_classes)

print(y_train.shape)        

for (i,lab) in enumerate(encoder.classes_):
    print("{}.{}".format(i+1,lab))

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  



filepath="best_weight/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=False,
                             period=1,verbose=1,save_best_only=False)
callbacks_list = [checkpoint]

base_model = VGG16(input_shape=(128,128,3),weights='imagenet', include_top=False)

    # Dong bang cac layer
for layer in base_model.layers:
        layer.trainable = False


    # Them cac layer FC va Dropout
x = Flatten(name='flatten')(base_model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictions')(x)

    # Compile
my_model = Model(base_model.input, x)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(my_model.summary())
my_model.fit(
  X_train,y_train,
  validation_data=(X_test,y_test),
  epochs=1,verbose=2,shuffle=False, 
  batch_size=64,
  callbacks=callbacks_list
)
my_model.save("model_SVnghia.h5")
 
predictions = my_model.predict(X_test)
score = my_model.accuracy_score(encoder.change(y_test), change(predictions))
print(score)

# data_folder = "E:/anhECG/ecg_img/train"
# #Đường dẫn folder
# for folder in os.listdir(data_folder):
#     fd=folder
#     data_folder1 = os.path.join(data_folder, folder)
#     for folder in os.listdir(data_folder1):
#         curr_path = os.path.join(data_folder1, folder)
#         for file in os.listdir(curr_path):
#             curr_file = os.path.join(curr_path, file)
#             images = cv2.imread(curr_file)
#             new_images = cv2.resize(images,(width,height))
#             X.append(new_images)
#             y.append(fd)
    
# # Tiền xử lý dữ liệu
# X = np.array(X)
# y = np.array(y)
# # In xem ảnh như nào
# print(X.shape)
# # Chuyển Label sang dạng nhị phân one hot
# encoder = LabelBinarizer()
# y=encoder.fit_transform(y)

# #Label tương ứng với Class

# for (i,lab) in enumerate(encoder.classes_):
#     print("{}.{}".format(i+1,lab))
    
# # Không biết lý do sao đọc folder ngược đời nên em phải làm vậy cho khỏi nhầm
# class_names = ['F', 'N', 'Q','S','V']

# # CHia dữ liệu thành các tập x,y train test
# #X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# #Vẽ mấy cái ảnh xem thử
# labels = np.array([int(np.where(x==1)[0]) for x in y])
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X[i],cmap=plt.cm.binary)
#     plt.xlabel(class_names[labels[i]])
# plt.show()

# Định nghĩa model (layer.trainable chặn FC của mạng VGG16)
# Khi đó khai báo 1 đoạn FC khác với đầu ra hàm softmax trong vecto lấy cái cao nhất
# =============================================================================
# base_model = VGG16(input_shape=(128,128,3),weights='imagenet', include_top=False)
# 
#     # Dong bang cac layer
# for layer in base_model.layers:
#         layer.trainable = False
# 
# 
#     # Them cac layer FC va Dropout
# x = Flatten(name='flatten')(base_model.output)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
# x = Dense(5, activation='softmax', name='predictions')(x)
# 
#     # Compile
# my_model = Model(base_model.input, x)
# my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     
# 
# =============================================================================
# Dữ dụng data augument lúc train (Không cần làm lúc trc train)
# (phóng to,dịch sang trái,phải 1/10 độ rộng của ảnh,điều chỉnh độ sáng từ 0.2 đến 1.5,méo ảnh )
# =============================================================================
# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
#     rescale=1./255,
#  	width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=45,
#  	horizontal_flip=True,
#     brightness_range=[0.2,1.5],
#     fill_mode="nearest")
# 
# aug_val = ImageDataGenerator(rescale=1./255)
# 
# 
# # Lưu kết quả train tốt hơn sau mỗi lần epochs lưu weight tốt nhất
# filepath="best_weight/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# =============================================================================

# Fit thôi.. cái này sinh data khi train
# my_model.fit_generator(aug.flow(X, y, batch_size=64),
#                                 epochs=10,# steps_per_epoch=len(X)//64,
#                                 validation_data=aug.flow(X,y,
#                                 batch_size=len(X)),
#                                 callbacks=callbacks_list)
"""
my_model.fit(X,y,epochs=10,batch_size=64)
my_model.save("model_SV.h5")

# hiển thị table trực quan
loss_data = my_model.history.history['loss']
accuracy_data = my_model.history.history['accuracy']
loss_val = my_model.history.history['val_loss']
accuracy_val = my_model.history.history['val_accuracy']

epoch_data =  my_model.history.epoch

f = plt.figure(figsize=(10,5))
ch1 = f.add_subplot(121)
ch2 = f.add_subplot(122)
ch1.plot(epoch_data, loss_data, label="train_loss")
ch1.plot(epoch_data, loss_val, label="val_loss")

ch1.set_title("Training Loss")
ch1.set_xlabel("Epoch #")
ch1.set_ylabel("Loss")
ch1.legend()
ch2.plot(epoch_data, accuracy_data, label="train_accuracy")
ch2.plot(epoch_data, accuracy_val, label="val_accuracy")
ch2.set_title("Training Accuracy")
ch2.set_xlabel("Epoch #")
ch2.set_ylabel("Accuracy")
ch2.legend()
plt.show()

# Đánh giá thử nghiệm 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)
print('Test Lost:', test_loss)

predictions = model.predict(X_test)

# # Thử nghiệm trên bộ test 
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])

#     plt.imshow(img, cmap=plt.cm.binary)

#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'

#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)


# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array, true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(3), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)

#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')

# num_rows = 10
# num_cols = 10
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], labels, X_test)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], labels)
# plt.tight_layout()
# plt.show()

"""








