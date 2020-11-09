# 1D cnn for SER

from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Conv2D, BatchNormalization, MaxPooling2D, LSTM, Dense, Activation, Reshape
from datasets import escolher_dataset
from tensorflow.keras.utils import to_categorical, normalize
import keras.backend as K
import argparse
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, accuracy_score
import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Banco usado
emo_db=True
ravdess=False
savee=False

#Tem Janela Deslizante
window=True

#Speaker dependent ou independent
speaker="dependent"

def cnn2d(input_shape, num_classes, args):
    model = tf.keras.Sequential(name='CNN_2D')

    # LFLB1
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.005)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


    # LFLB2
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4)))


    # LFLB3
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4)))


    # LFLB4
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4)))


    # LSTM

    model.add(Reshape((1, 128)))
    #model.add(Reshape((1, 384)))
    #model.add(LSTM(units=256))
    model.add(LSTM(units=args.num_fc, kernel_regularizer=tf.keras.regularizers.l2(0.005)))

    

    # FC
    model.add(Dense(units=num_classes, activation='softmax'))

    # Model compilation

    #opt = optimizers.SGD(lr=args.learning_rate, decay=args.decay,momentum=args.momentum, nesterov=True)
    #opt = optimizers.Adam(lr=args.learning_rate)
    opt = optimizers.SGD(lr=0.01, decay=1e-3, momentum=0.8)


    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def train(model, x_tr, y_tr, x_val, y_val, args):
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model_cnn2d.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_tr, y_tr, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
                        callbacks=[es, mc])
    return model


def test(model, x_t, y_t):
    saved_model = load_model('best_model_cnn2d.h5')
    score = saved_model.evaluate(x_t, y_t, batch_size=20)
    print(score)
    return score

def string2num(y):

    if emo_db:
        classes = ['W', 'L', 'E', 'A', 'F', 'T', 'N']
    elif ravdess:
        classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]
    elif savee:
        classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

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

def check_classes(y_tr, y_t):
    treino = np.array(y_tr)
    teste  = np.array(y_t)

    unique_treino, counts_treino = np.unique(treino, return_counts=True)
    result_treino = dict(zip(unique_treino, counts_treino))
    print('Classes para treino: ', result_treino)

    unique_teste, counts_teste = np.unique(teste, return_counts=True)
    result_teste = dict(zip(unique_teste, counts_teste))
    print('Classes para teste: ', result_teste)

def loadData():

    database = ''
    if emo_db:
        database = 'emo'
    elif ravdess:
        database = 'ravdess'
    elif savee:
        database = 'savee'

    print(database)
    x_tr, x_t, y_tr, y_t = escolher_dataset(database, '2d', 'mfcc', window)

    check_classes(y_tr, y_t)

    y_tr = string2num(y_tr)
    y_t = string2num(y_t)

    x_tr = x_tr.reshape(-1,x_tr.shape[1:][0], x_tr.shape[1:][1], 1)
    x_t = x_t.reshape(-1,x_tr.shape[1:][0], x_tr.shape[1:][1], 1)

    y_tr = to_categorical(y_tr)
    y_t = to_categorical(y_t)

    return x_tr, y_tr, x_t, y_t


if __name__ == "__main__":
    import numpy as np


    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.num_fc = 64 #64
    args.batch_size = 32
    # args.batch_size = 5
    # best model will be saved before number of epochs reach this value
    # args.num_epochs = 1500
    args.num_epochs = 1
    args.learning_rate = 0.0001
    args.decay = 1e-6
    args.momentum = 0.9

    f = open("results.txt", "w")
    for i in range(3):
        banco = ""
        if i == 0:
            emo_db=True
            savee=False
            ravdess=False
            banco = "EMO-DB"

        elif i == 1:
            emo_db = False
            savee = True
            ravdess = False
            banco = "SAVEE"
            continue
        elif i == 2:
            emo_db = False
            savee = False
            ravdess = True
            banco = "RAVDESS"
            continue

        final_accuracy = np.array([])
        final_ROC_AUC = np.array([])
        final_F1_Macro = np.array([])
        final_F1_Micro = np.array([])
        final_cm = []
                        #5
        for i in range(1):

            # load data
            x_tr, y_tr, x_t, y_t = loadData()
            print(len(x_tr))

            # define model
            model = cnn2d(input_shape=x_tr.shape[1:], num_classes=7, args=args)

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

            # Accuracy Score
            print('Accuracy Score: ',accuracy_score(rounded_true, rounded_predictions))

            # F1-Score
            f1_micro = f1_score(y_true=rounded_true, y_pred=rounded_predictions, average='micro')
            f1_macro = f1_score(y_true=rounded_true, y_pred=rounded_predictions, average='macro')
            final_F1_Macro = np.append(final_F1_Macro, f1_macro)
            final_F1_Micro = np.append(final_F1_Micro, f1_micro)

            # ROC-AUC
            roc_auc = roc_auc_score(y_t, predictions)
            final_ROC_AUC = np.append(final_ROC_AUC, roc_auc)

            # Matriz de confusão
            emotions_labels = []

            if emo_db:
                #emotions_labels = ['Anger', 'Bordedom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
                emotions_labels = ['Raiva', 'Tedio', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Neutra']
            elif ravdess:
                #emotions_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]
                emotions_labels = ["Neutra", "Calma", "Felicidade", "Tristeza", "Raiva", "Medo", "Nojo"]
            elif savee:
                #emotions_labels = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
                emotions_labels = ["Raiva", "Nojo", "Medo", "Felicidade", "Neutra", "Tristeza", "Surpresa"]

            y_labels = [0,1,2,3,4,5,6]

            cm = confusion_matrix(y_true=rounded_true, y_pred=rounded_predictions, labels=y_labels)
            if len(final_cm) > 0:
                for i in range(len(final_cm)):
                    for j in range(len(final_cm[i])):
                        final_cm[i][j] = final_cm[i][j] + cm[i][j]
            else:
                final_cm = cm


        cmn = final_cm.astype('float') / final_cm.sum(axis=1)[:, np.newaxis]

        # UA
        UA = np.array([])
        for i in range(7):
            for j in range(7):
                if i == j:
                    UA = np.append(UA, cmn[i][j])

        # Metricas de avaliação
        print("UA: ", UA.mean())
        print("ROC_AUC: ", final_ROC_AUC.mean())
        print("F1 Macro: ", final_F1_Macro.mean())
        print("F1 Micro: ", final_F1_Micro.mean())

        # PLOT
        axes = sns.heatmap(cmn, fmt='.1%', square=True, annot=True, cbar=True, cmap=plt.cm.GnBu)


        axes.set_xlabel('Verdadeiro')
        axes.set_ylabel('Predição')

        tick_marks = np.arange(len(emotions_labels)) + 0.5

        axes.set_xticks(tick_marks)
        axes.set_xticklabels(emotions_labels, rotation=30)

        axes.set_yticks(tick_marks)
        axes.set_yticklabels(emotions_labels, rotation=0)

        axes.set_title('Matriz de confusão - '+banco)

        plt.show()

        path = ''

        if window:
            hasJanela = "_Janela_"
        else:
            hasJanela = ""

        if emo_db:
            path = 'confusion_matrix/cnn2d_'+'emo_'+speaker+hasJanela+'.png'
        elif ravdess:
            path = 'confusion_matrix/cnn2d_'+'ravdess'+speaker+hasJanela+'.png'
        elif savee:
            path = 'confusion_matrix/cnn2d_'+'savee'+speaker+hasJanela+'.png'
        axes.figure.savefig(path, dpi=100)

        f.write(banco+": UA="+str(UA.mean())+"; ROC_AUC="+str(final_ROC_AUC.mean())+"; F1 Macro="+str(final_F1_Macro.mean())+"; F1 Micro="+str(final_F1_Micro.mean()) + ";\n")

    f.close()