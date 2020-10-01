from tensorflow.keras.models import load_model

if __name__ == '__main__':
    saved_model = load_model('best_model_cnn2d.h5')
    saved_model.summary()