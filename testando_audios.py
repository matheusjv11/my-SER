import librosa
import glob
import os
import soundfile

if __name__ == '__main__':

    audios = []
    for file in glob.glob("emo_db/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        y, sr = librosa.load(file, duration=1)
        print(sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, )
        audios.append(mfccs)
        with soundfile.SoundFile(file) as sound_file:
            x = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            duration_seconds =float(len(x)) / sample_rate
            #print(duration_seconds)