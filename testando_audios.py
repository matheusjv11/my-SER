import librosa
from scipy.io.wavfile import write
import glob
import os
import soundfile
import numpy as np

def analysis(audios):

    seconds1to2 = []
    seconds2to3 = []
    seconds3to4 = []
    seconds4toMax = []

    print('---- Análise dos audios ----')
    print('Mais curto:', audios.min())
    print('Mais longo:', audios.max())
    print('Media:', audios.mean())
    print('Mediana:', np.median(audios))

    for audio in audios:
        if audio > 1 and audio <= 2:
            seconds1to2.append(audio)
        if (audio > 2) and (audio <=3):
            seconds2to3.append(audio)
        if audio > 3 and audio <=4:
            seconds3to4.append(audio)
        if audio > 4:
            seconds4toMax.append(audio)

    print('---- Quantitativo ----')
    print('Total:', len(audios))
    print('De 1 a 2 segundos:', len(seconds1to2))
    print('De 2 a 3 segundos:', len(seconds2to3))
    print('De 3 a 4 segundos:', len(seconds3to4))
    print('Maiores de 4 segundos:', len(seconds4toMax))

def resize_audio(audio):

    new_audio = audio
    feature = 0

    while(len(new_audio)<128000):
        new_audio = np.append(new_audio, audio[feature])
        feature += 1
        if feature == len(audio):
            feature = 0

    return new_audio

if __name__ == '__main__':

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

    """"with soundfile.SoundFile(file) as sound_file:
        x = sound_file.read(dtype="float32", )
        sample_rate = sound_file.samplerate
        duration_seconds = float(len(x)) / sample_rate
        if basename == '08b03Tc.wav':
            print(duration_seconds, float(len(x)))
            # 8.97825 143652.0
        audio_seconds = np.hstack((audio_seconds, duration_seconds))
        # print(duration_seconds)"""