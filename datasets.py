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
    "W":"anger",
    "L":"boredom",
    "E":"disgust",
    "A":"fear",
    "F":"happiness",
    "T":"sadness",
    "N":"neutral"
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
    "anger",
    "sadness",
    "neutral",
    "happiness",
    "boredom",
    "disgust",
    "fear"
}
emotions = ["anger","sadness","neutral","boredom","disgust","fear","happiness"]

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
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
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def load_data_emo(test_size=0.2):
    X, y = [], []
    for file in glob.glob("emo_db/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
       
        # get the emotion label
        emotion = emo_db[basename[5]]
    
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS_EMO:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


def escolher_dataset(dataset):
    if dataset == "emo":
        return load_data_emo()

    elif dataset == "ravdess":
        return load_data_ravdess()
    else:
        raise NotImplementedError("Dataset não foi selecionado")


##-- POE EM OUTRO LUGAR
def confusionmatrix(model,X_test, y_test, percentage=True, labeled=True):
        """
        Computes confusion matrix to evaluate the test accuracy of the classification
        and returns it as numpy matrix or pandas dataframe (depends on params).
        params:
            percentage (bool): whether to use percentage instead of number of samples, default is True.
            labeled (bool): whether to label the columns and indexes in the dataframe.
        """
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred, labels=emotions).astype(np.float32)
        #matrix = confusion_matrix().astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in emotions ],
                                    columns=[ f"predicted_{e}" for e in emotions ])
        return matrix