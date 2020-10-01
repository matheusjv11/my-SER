from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn import svm, metrics
from datasets import escolher_dataset

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
    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '2d', 'mfcc')

    y_tr = string2num(y_tr)
    y_t = string2num(y_t)

    x_tr = x_tr.reshape(-1,x_tr.shape[1:][0], x_tr.shape[1:][1], 1)
    x_t = x_t.reshape(-1,x_tr.shape[1:][0], x_tr.shape[1:][1], 1)

    y_tr = to_categorical(y_tr)
    y_t = to_categorical(y_t)

    return x_tr, y_tr, x_t, y_t


if __name__ == '__main__':
    x_tr, y_tr, x_t, y_t = loadData()

    # Modelo SVM
    model = svm.SVC(kernel='linear')

    #Treinando o modelo
    model.fit(x_tr, y_tr)
    