import librosa
from scipy.io.wavfile import write
import glob
import os
import soundfile
import numpy as np

emo_db = {
        "W": "Anger(W)",
        "L": "Boredom(L)",
        "E": "Disgust(E)",
        "A": "Fear(A)",
        "F": "Happiness(F)",
        "T": "Sadness(S)",
        "N": "Neutral"
    }

def duration_analisys():
    # Essa função faz uma analise focada na duração dos audios

    # Coletando a duração dos audios em segundos
    audios_seconds = np.array([])

    Anger_Seconds = np.array([])
    Boredom_Seconds = np.array([])
    Disgust_Seconds = np.array([])
    Fear_Seconds = np.array([])
    Happiness_Seconds = np.array([])
    Neutral_Seconds = np.array([])
    Sadness_Seconds = np.array([])

    for file in glob.glob("emo_db/*.wav"):
        basename = os.path.basename(file)
        emotion = emo_db[basename[5]]
        y, sr = librosa.load(file, duration=8, sr=16000, dtype=np.float32)
        duration_seconds = float(len(y)) / sr
        audios_seconds = np.hstack((audios_seconds, duration_seconds))

        if emotion == "Anger(W)":
            Anger_Seconds = np.append(Anger_Seconds, duration_seconds)
        elif emotion == "Boredom(L)":
            Boredom_Seconds = np.append(Boredom_Seconds, duration_seconds)
        elif emotion == "Disgust(E)":
            Disgust_Seconds = np.append(Disgust_Seconds, duration_seconds)
        elif emotion == "Fear(A)":
            Fear_Seconds = np.append(Fear_Seconds, duration_seconds)
        elif emotion == "Happiness(F)":
            Happiness_Seconds = np.append(Happiness_Seconds, duration_seconds)
        elif emotion == "Sadness(S)":
            Sadness_Seconds = np.append(Sadness_Seconds, duration_seconds)
        else:
            Neutral_Seconds = np.append(Neutral_Seconds, duration_seconds)

    seconds1to2 = []
    seconds2to3 = []
    seconds3to4 = []
    seconds4toMax = []

    print('---- Análise dos audios ----')
    print('Mais curto:', audios_seconds.min())
    print('Mais longo:', audios_seconds.max())
    print('Media:', audios_seconds.mean())
    print('Mediana:', np.median(audios_seconds))
    print('Variância:', np.var(audios_seconds))
    print('Desvio Padrão:', np.std(audios_seconds))

    for audio in audios_seconds:
        if audio > 1 and audio <= 2:
            seconds1to2.append(audio)
        if (audio > 2) and (audio <=3):
            seconds2to3.append(audio)
        if audio > 3 and audio <=4:
            seconds3to4.append(audio)
        if audio > 4:
            seconds4toMax.append(audio)

    print('---- Quantitativo ----')
    print('Total:', len(audios_seconds))
    print('De 1 a 2 segundos:', len(seconds1to2))
    print('De 2 a 3 segundos:', len(seconds2to3))
    print('De 3 a 4 segundos:', len(seconds3to4))
    print('Maiores de 4 segundos:', len(seconds4toMax))

    print('---- Media de segundos por emoção  ----')
    print('Anger: ', Anger_Seconds.mean())
    print('Boredom: ', Boredom_Seconds.mean())
    print('Disgust: ', Disgust_Seconds.mean())
    print('Fear: ', Fear_Seconds.mean())
    print('Happinnes: ', Happiness_Seconds.mean())
    print('Neutral: ', Neutral_Seconds.mean())
    print('Sadness: ', Sadness_Seconds.mean())



def resize_audio(audio):

    new_audio = audio
    feature = 0

    while(len(new_audio)<128000):
        new_audio = np.append(new_audio, audio[feature])
        feature += 1
        if feature == len(audio):
            feature = 0

    return new_audio

def emodb_to_8ec():
    for file in glob.glob("emo_db_window/*.wav"):
        basename = os.path.basename(file)
        y, sr = librosa.load(file, sr=16000, dtype=np.float32)
        audio8sec=resize_audio(y)
        write("emo_db_window_8sec/"+basename, 16000, audio8sec)


def novoAudio_JalenaDeslizante(audio, basename):

    limite = len(audio)
    #jump_janela = 8000  # 0,5 segundos
    jump_janela = int(len(audio)/5)
    comeco_janela = 0
    fim_janela = 32000


    i=0
    while(fim_janela<=limite):
        novo_audio = audio[comeco_janela:fim_janela]
        write("emo_db_window/" + basename + "_" + str(fim_janela) + ".wav", 16000, novo_audio)

        comeco_janela += jump_janela
        fim_janela += jump_janela
        i+=1


    print(basename, float(len(audio)) / 16000, i, jump_janela)
    return (i)



def janela_deslizante():
    audios = []
    audios_gerados = np.array([])
    i=0
    x=0
    for file in glob.glob("emo_db/*.wav"):
        basename = os.path.basename(file)
        basename = basename.split('.')[0]

        y, sr = librosa.load(file, sr=16000, dtype=np.float32)

        #Se o audio tiver menos de 2,5 segundos, não faz nada com ele
        if len(y) < 40000 :
            i+=1
            continue

        gerados=novoAudio_JalenaDeslizante(y, basename)
        audios_gerados = np.append(audios_gerados, gerados)

        x+=1


    print("nao mudados:", i, "mudados:", x)
    print("media de novos audios:", audios_gerados.mean(), "menor:", audios_gerados.min(), "maior:", audios_gerados.max())

def classes_emo():
    # Essa função coleta informações das classes de emoções do Emo-db

    emotions_originalDB = np.array([])
    emotions_windowDB = np.array([])

    # Estudo das classes do Emo db original
    for file in glob.glob("emo_db/*.wav"):
        basename = os.path.basename(file)
        emotion = emo_db[basename[5]]
        emotions_originalDB = np.append(emotions_originalDB, emotion)

    unique_original, counts_original = np.unique(emotions_originalDB, return_counts=True)
    result_original = dict(zip(unique_original, counts_original))
    print(result_original)

    # Estudo das classes do Emo db com janela deslizante
    for file in glob.glob("emo_db_window/*.wav"):
        basename = os.path.basename(file)
        emotion = emo_db[basename[5]]
        emotions_windowDB = np.append(emotions_windowDB, emotion)

    unique_window, counts_window = np.unique(emotions_windowDB, return_counts=True)
    result_window = dict(zip(unique_window, counts_window))
    print(result_window)

    # Porcentagem de crescimento das classes
    porcentagem = []
    for mood in unique_window:
        float_value = ((result_window[mood] - result_original[mood]) / result_original[mood])
        percentage = "{:.0%}".format(float_value)
        porcentagem.append(percentage)

    result_porcentagem = dict(zip(unique_window, porcentagem))
    print(result_porcentagem)

if __name__ == '__main__':

    # Essas funções modificação os audios
    #janela_deslizante()
    #emodb_to_8ec()

    # Essas funções analisam as bases de dados
    #classes_emo()
    duration_analisys()
    """""
    audios = []
    audio_seconds = np.array([])
    
    for file in glob.glob("emo_db_8sec/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        y, sr = librosa.load(file, duration=8, sr=16000,dtype=np.float32)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, )
        audios.append(mfccs)

        duration_seconds = float(len(y)) / sr
        #print(duration_seconds, float(len(y)), len(mfccs[0]))

        audio_seconds = np.hstack((audio_seconds, duration_seconds))

    audios = np.ndarray(audios)
    analysis(audio_seconds)
    """""