import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sample = "bgaudio/1.wav"
data, sample_rate = librosa.load(sample)

plt.title(" Gelombang Suara ")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

mfccs = librosa.feature.mfcc(y=data,sr = sample_rate, n_mfcc=40)
print("Bentuk MfCC: ", mfccs.shape)

plt.title('MFCC')
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

all_data = []

lokasi_data = {
    0:["bgaudio/" + lokasi_file for lokasi_file in os.listdir("bgaudio/")],
    1:["dataaudio/" + lokasi_file for lokasi_file in os.listdir("dataaudio/")]
}

for label_kelas, list_file in lokasi_data.items():
    for data_tunggal in list_file:
        data, sample_rate = librosa.load(data_tunggal)
        mfccs = librosa.feature.mfcc(y=data,sr = sample_rate, n_mfcc=40)
        mfcc_preprocess = np.mean(mfccs.T, axis=0)
        all_data.append([mfcc_preprocess, label_kelas])
    
    print(f"Sukses Melabeli Data {label_kelas}")
    
df = pd.DataFrame(all_data, columns=["fitur", 'label_kelas'])

df.to_pickle('proper_dataset/audio_data.csv')