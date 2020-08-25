"""
Esse arquivo serve para retonar o modelo e os parametros escolhidos
para a execução do código
"""

from sklearn.neural_network import MLPClassifier


def escolher_modelo(modelo):
    if modelo == "padrao":
        return padrao()
    return False




def padrao():

    model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 1000,
    }
    # Inicia Multi Layer Perceptron classifier
    model = MLPClassifier(**model_params)
    return model