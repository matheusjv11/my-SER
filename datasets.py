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

# Todas emoções no SAVEE
savee = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}

# Emoções escolhidas para execução RAVDESS
AVAILABLE_EMOTIONS_RAVDESS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust"
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

def extract_feature_1d(file_name, **kwargs):
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
    audio = kwargs.get("audio")

    y, sr = librosa.load(file_name, duration=8, sr=16000, dtype=np.float32)
    result = np.array([])

    if mfcc:
        # O np mean é utilizado para transformar a matriz em vetor, tirando a media de cada linha
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T, axis=0)
        result = np.hstack((result, mfccs))

    if mel:
        mel1d = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
        mel = librosa.power_to_db(mel1d ** 2)

        result = np.hstack((result, mel))
    if audio:
        result = np.hstack((result, y))

    return result

def extract_feature_2d(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    mel = kwargs.get("mel")
    y, sr = librosa.load(file_name, duration=8, sr=16000, dtype=np.float32)
    result = []

    if mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128 )
        result = mfccs

    if mel:
        mel2d = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=512)
        logmel = librosa.power_to_db(mel2d, ref=np.max)
        result = logmel

    return result


def load_data_savee(test_size=0.2, **kwargs):
    X, y = [], []

    dimension = kwargs.get("dimension")
    feature = kwargs.get("feature")
    window = kwargs.get("window")

    if feature == 'mfcc':
        mfcc = True
        mel = False
        audio = False
    elif feature == 'mel':
        mfcc = False
        mel = True
        audio = False
    else:
        mfcc = False
        mel = False
        audio = True

    if window == True:
        path = 'savee_window_8sec/*.wav'
    else:
        path = "savee_8sec/*.wav"

    for file in glob.glob(path):
        # get the base name of the audio file
        basename = os.path.basename(file)

        # get the emotion label
        emotion = savee[basename.split("_")[0]]

        # extract speech features
        if dimension == '1d':
            features = extract_feature_1d(file, mfcc=mfcc, mel=mel, audio=audio, dimension=dimension)
        else:
            features = extract_feature_2d(file, mfcc=mfcc, mel=mel)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def load_data_ravdess(test_size=0.2, **kwargs):
    X, y = [], []

    dimension = kwargs.get("dimension")
    feature = kwargs.get("feature")
    window = kwargs.get("window")

    if feature == 'mfcc':
        mfcc = True
        mel = False
        audio = False
    elif feature == 'mel':
        mfcc = False
        mel = True
        audio = False
    else:
        mfcc = False
        mel = False
        audio = True

    if window == True:
        path = 'ravdess_window_8sec/*.wav'
    else:
        path = "ravdess_8sec/*.wav"

    for file in glob.glob(path):
        # get the base name of the audio file
        basename = os.path.basename(file)
        
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]

        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS_RAVDESS:
            continue
        # extract speech features
        if dimension == '1d':
            features = extract_feature_1d(file, mfcc=mfcc, mel=mel, audio=audio, dimension=dimension)
        else:
            features = extract_feature_2d(file, mfcc=mfcc, mel=mel)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def load_data_emo(test_size=0.2, **kwargs):
    X, y = [], []

    dimension = kwargs.get("dimension")
    feature = kwargs.get("feature")
    window = kwargs.get("window")

    if feature =='mfcc':
        mfcc=True
        mel=False
        audio=False
    elif feature =='mel':
        mfcc = False
        mel = True
        audio = False
    else:
        mfcc = False
        mel = False
        audio = True

    if window == True:
        path = 'emo_db_window_8sec/*.wav'
    else:
        path = "emo_db_8sec/*.wav"

    for file in glob.glob(path):
        # get the base name of the audio file
        basename = os.path.basename(file)
       
        # get the emotion label
        emotion = emo_db[basename[5]]

        # we allow only AVAILABLE_EMOTIONS we set
        #if emotion not in AVAILABLE_EMOTIONS_EMO:
            #continue
        # extract speech features
        if dimension == '1d':
            features = extract_feature_1d(file, mfcc=mfcc, mel=mel, audio=audio, dimension=dimension)
        else:
            features = extract_feature_2d(file, mfcc=mfcc, mel=mel)
        # add to data
        X.append(features)
        y.append(emotion)

    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, stratify=y) # NORMAL
    #return np.array(X), y # KFOLD


def escolher_dataset(dataset, dimension, feature, window):
    if dataset == "emo":
        return load_data_emo(dimension=dimension, feature=feature, window=window)
    elif dataset == "ravdess":
        return load_data_ravdess(dimension=dimension, feature=feature, window=window)
    elif dataset == "savee":
        return load_data_savee(dimension=dimension, feature=feature, window=window)
    else:
        raise NotImplementedError("Dataset não foi selecionado")


x_tr, x_t, y_tr, y_t  = escolher_dataset("emo", "2d", "mfcc", True)
