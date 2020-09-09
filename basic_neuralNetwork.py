from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from datasets import escolher_dataset, confusionmatrix
from models import escolher_modelo
import os
import pickle

# Carregar dados do dataset selecionado
# Datasets disponiveis: emo, ravdess.
X_train, X_test, y_train, y_test = escolher_dataset("emo")


# Numero de amostra nos dados de treino
print("[+] Numero de amostras para treino:", X_train.shape[0])

# number of samples in testing data
print("[+] Numero de amostras para teste:", X_test.shape[0])

#Numero de features utilizadas
print("[+] Numero de features:", X_train.shape[1])

# Configurando o modelo
# Modelos disponiveis: padrao.
model = escolher_modelo("padrao")

# Treinamento do modelo
print("[*] Treinando o modelo...")
model.fit(X_train, y_train)

# predicção de 20% dos dados para ver quão bom o modelo está
y_pred = model.predict(X_test)

# calcular a acurácia
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print(confusionmatrix(model,X_test,y_test))
print("Accuracy: {:.2f}%".format(accuracy*100))

predictions = model.predict(X_test)
for prediction in predictions:
	print(prediction)


# Salvando o modelo
# Cria um diretorio result, caso nao criado antes
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))
