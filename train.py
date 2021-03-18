#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-07-03
#Description:
import tensorflow as tf
keras = tf.keras
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import numpy as np
from liveness import create_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        tf.config.experimental
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

height = 32
width = 32
depth = 3

trainset = r'D:\工作\翻拍\Fakeness-master\data\train'
valset = r'D:\工作\翻拍\Fakeness-master\data\val'
ckp_path = r'models\ckp.h5'
# data=np.load('data/v1/data.npz')
 
# X_train=data['X_train']
# Y_train=data['Y_train']
# X_valid=data['X_valid']
# Y_valid=data['Y_valid']
# X_test= data['X_test']
# Y_test= data['Y_test']
def train():

    generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )

    traindataloader = generator.flow_from_directory(
        trainset,
        batch_size=32,
        target_size=(height,width)
    )

    valdataloader = generator.flow_from_directory(
        valset,
        batch_size=32,
        target_size=(height,width)
    )

    train_ckp = ModelCheckpoint(ckp_path,monitor='val_acc')
    model = create_model()
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('=' * 40 + '开始训练' + '=' * 40)
    model.fit_generator(
        traindataloader,
        epochs=100,
        verbose=1,
        callbacks=[train_ckp],
        validation_data=valdataloader,
        workers=2

    )
    # model.fit(
    #     X_train,Y_train,
    #     batch_size = 16,
    #     epochs=100,
    #     callbacks=[train_ckp],
    #     verbose=1,
    #     validation_data=(X_test,Y_test),
    #     shuffle = True
    # )
    model.save_weights(r'models\liveness4.0.h5')



if __name__ == "__main__":
    train()



