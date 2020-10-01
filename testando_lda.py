from datasets import escolher_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def LDA_MFCC_1D():
    print('----- MFCC 1D -----')
    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '2d', 'mfcc')

    print('Shape 2d Mfcc: ',x_tr.shape)
    x_tr = x_tr.reshape(x_tr.shape[0], -1)
    x_t = x_t.reshape(x_t.shape[0], -1)
    print('Shape 1d Mfcc: ', x_tr.shape)

    model = LinearDiscriminantAnalysis()
    model.fit(x_tr, y_tr)

    transformed = model.transform(x_tr)
    print('Shape Mfcc pós LDA: ', transformed.shape)

    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_true=y_t, y_pred=y_pred)

    print("Precisão: {:.2f}%".format(accuracy * 100))



def LDA_AUDIO_1D():
    print('----- AUDIO 1D -----')
    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '1d', 'audio')
    print('Shape 1d Audio: ', x_tr.shape)

    model = LinearDiscriminantAnalysis()
    model.fit(x_tr, y_tr)

    transformed = model.transform(x_tr)
    print('Shape Audio pós LDA: ', transformed.shape)

    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_true=y_t, y_pred=y_pred)

    print("Precisão: {:.2f}%".format(accuracy * 100))

if __name__ == '__main__':

    LDA_MFCC_1D()
    LDA_AUDIO_1D()

