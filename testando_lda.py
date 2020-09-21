from datasets import escolher_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    x_tr, x_t, y_tr, y_t = escolher_dataset('emo', '2d', 'mfcc')

    model = LinearDiscriminantAnalysis(n_components=3)
    model.fit(x_tr, y_tr)

    y_pred = model.predict(x_t)
    accuracy = accuracy_score(y_true=y_t, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))

    transformed = model.transform(x_tr)
    print(transformed)