import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix 

# Todas emoções no RAVDESS 
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Todas emoções no EMO-DB
emo_db = {
    "W":"W",
    "L":"L",
    "E":"E",
    "A":"A",
    "F":"F",
    "T":"T",
    "N":"N"
}


# Emoções escolhidas para execução RAVDESS
AVAILABLE_EMOTIONS_RAVDESS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

# Emoções escolhidas para execução EMO-DB
AVAILABLE_EMOTIONS_EMO = {
    "W",
    "S",
    "N",
    "F",
    "L",
    "E",
    "A"
}
emotions = ["anger","sadness","neutral","boredom","disgust","fear","happiness"]

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - MEL Spectrogram Frequency (mel)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    mel = kwargs.get("mel")
    dimension = kwargs.get("dimension")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        #result = np.array([])
        #result = np.ndarray(shape=)
        result = None

        if mfcc:
            # Verifica a dimensão dos dados que serão retornados
            if dimension == '1d':
                # O np mean é utilizado para transformar a matriz em vetor, tirando a media de cada linha
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)
            else:
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128, )

            #result = np.hstack((result, mfccs))
            result = mfccs
        if mel:
            # Verifica a dimensão dos dados que serão retornados
            if dimension == '1d':
                mel1d = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                mel = librosa.power_to_db(mel1d ** 2)
            else:
                mel2d = librosa.feature.melspectrogram(X, sr=sample_rate, n_fft=2048, hop_length=512)
                mel = librosa.power_to_db(mel2d ** 2)
            #result = np.hstack((result, mel))
            result = mel

    return result



def load_data_ravdess(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]

        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS_RAVDESS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True , chroma=False, mel=False)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def load_data_emo(test_size=0.2, **kwargs):
    X, y = [], []

    dimension = kwargs.get("dimension")
    feature = kwargs.get("feature")

    if feature =='mfcc':
        mfcc=True
        mel=False
    else:
        mfcc = False
        mel = True


    for file in glob.glob("emo_db/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
       
        # get the emotion label
        emotion = emo_db[basename[5]]
    
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS_EMO:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=mfcc, mel=mel, dimension=dimension)

        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


def escolher_dataset(dataset, dimension, feature):
    if dataset == "emo":
        return load_data_emo(dimension=dimension, feature=feature)
    elif dataset == "ravdess":
        return load_data_ravdess()
    else:
        raise NotImplementedError("Dataset não foi selecionado")


x_tr, x_t, y_tr, y_t  = escolher_dataset("emo", "2d", "mel")