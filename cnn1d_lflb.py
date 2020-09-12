# 1D cnn for SER

from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Activation, Layer
from datasets import escolher_dataset
from tensorflow.keras.utils import to_categorical, normalize
import keras.backend as K
import argparse
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy


def emo1d(input_shape, num_classes, args):
    model = tf.keras.Sequential(name='Emo1D')

    # LFLB1
    model.add(Conv1D(filters=64, kernel_size=(3), strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # LFLB2
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # LFLB3
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # LFLB4
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # LSTM
    model.add(LSTM(units=args.num_fc))

    # FC
    model.add(Dense(units=num_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.SGD(lr=args.learning_rate, decay=args.decay,
                         momentum=args.momentum, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def train(model, x_tr, y_tr, x_val, y_val, args):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    mc = ModelCheckpoint('best_model_cnn1d.h5', monitor='val_categorical_accuracy',
                         mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_tr, y_tr, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
                        callbacks=[es, mc])
    return model


def test(model, x_t, y_t):
    saved_model = load_model('best_model_cnn1d.h5')
    score = saved_model.evaluate(x_t, y_t, batch_size=20)
    print(score)
    return score

def string2num(y):
    classes = ['W', 'L', 'E', 'A', 'F', 'T', 'N']
    y1 = []
    for i in y:
        if(i == classes[0]):
            y1.append(0)
        elif(i == classes[1]):
            y1.append(1)
        elif(i == classes[2]):
            y1.append(2)
        elif(i == classes[3]):
            y1.append(3)
        elif(i == classes[4]):
            y1.append(4)
        elif(i == classes[5]):
            y1.append(5)
        else:
            y1.append(6)
    y1 = np.float32(np.array(y1))
    return y1

def loadData():

    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '1d', 'mfcc')

    y_tr = string2num(y_tr)
    y_t = string2num(y_t)
    x_tr = x_tr.reshape(-1, x_tr.shape[1], 1)
    x_t = x_t.reshape(-1, x_t.shape[1], 1)
    y_tr = to_categorical(y_tr)
    y_t = to_categorical(y_t)


    return x_tr, y_tr, x_t, y_t


emotions = ['W', 'L', 'E', 'A', 'F', 'T', 'N']


if __name__ == "__main__":
    import numpy as np


    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # load data
    x_tr, y_tr, x_t, y_t = loadData()

    args.num_fc = 64
    # args.batch_size = 32
    args.batch_size = 6
    # best model will be saved before number of epochs reach this value
    # args.num_epochs = 1500
    args.num_epochs = 300
    args.learning_rate = 0.0001
    args.decay = 1e-6
    args.momentum = 0.9

    # define model
    model = emo1d(input_shape=x_tr.shape[1:], num_classes=7, args=args)

    model.summary()

    # train model
    model = train(model, x_tr, y_tr, x_t, y_t, args=args)

    # test model
    score = test(model,x_t,y_t)
    # y_t = [numpy.argmax(y, axis=None, out=None) for y in y_t]
    # print(confusionmatrix(model, x_t, y_t))

    predictions = model.predict(x_t, batch_size=10, verbose=0)
    """for prediction in predictions:
        print(prediction)"""
    rounded_predictions = np.argmax(predictions, axis=1)
    rounded_true = np.argmax(y_t, axis=1)
    cm = confusion_matrix(y_true=rounded_true, y_pred=rounded_predictions)
    print(cm)
