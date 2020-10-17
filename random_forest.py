from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from datasets import escolher_dataset
from sklearn.metrics import f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

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
    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '1d', 'mfcc')
    check_classes(y_tr, y_t)
    #y_tr = string2num(y_tr)
    #y_t = string2num(y_t)

    #x_tr = x_tr.reshape(-1, x_tr.shape[1:][0] * x_tr.shape[1:][1])
    #x_t = x_t.reshape(-1, x_t.shape[1:][0] * x_t.shape[1:][1])

    #y_tr = to_categorical(y_tr)
    #y_t = to_categorical(y_t)

    return x_tr, y_tr, x_t, y_t


if __name__ == '__main__':
    x_tr, y_tr, x_t, y_t = loadData()

    # Modelo SVM
    model = RandomForestClassifier(max_depth=100, random_state=0)

    # Treinando o modelo
    model.fit(x_tr, y_tr)

    # Predição
    pred = model.predict(x_t)

    #Avaliação
    y_labels = ['W', 'L', 'E', 'A', 'F', 'T', 'N']
    emotions_labels = ['Anger', 'Bordedom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

    print("Acuracy: ", metrics.accuracy_score(y_true=y_t, y_pred=pred))
    print("Precision: ", metrics.precision_score(y_true=y_t, y_pred=pred , average="micro"))
    cm = metrics.confusion_matrix(y_true=y_t, y_pred=pred, labels=y_labels)
    print("Confusion Matrix: \n", cm)

    # F1-Score
    f1_micro = f1_score(y_true=y_t, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=y_t, y_pred=pred, average='macro')
    print('F1 - Macro: ', f1_macro)
    print('F1 - Micro: ', f1_micro)

    # ROC-AUC
    #roc_auc = roc_auc_score(y_t, pred)
    #print('ROC-AUC: ', roc_auc)
    #
    # Matriz de confusão
    axes = sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap=plt.cm.GnBu)

    axes.set_xlabel('Verdadeiro')
    axes.set_ylabel('Predição')

    tick_marks = np.arange(len(emotions_labels)) + 0.5

    axes.set_xticks(tick_marks)
    axes.set_xticklabels(emotions_labels, rotation=30)

    axes.set_yticks(tick_marks)
    axes.set_yticklabels(emotions_labels, rotation=0)

    axes.set_title('Matriz de confusão')

    plt.show()
    axes.figure.savefig('confusion_matrix/random_forest.png', dpi=100)