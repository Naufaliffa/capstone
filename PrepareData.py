import sounddevice as sd
from scipy.io.wavfile import write

def rekam_wakeword_audio(save_path, n_kali=20):
    input("Untuk merekam audio wake word tekan Enter : ")
    for i in range (n_kali):
        fs = 44100
        detik = 3
        rekaman = sd.rec(int(detik*fs), samplerate = fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, rekaman)
        input(f"Tekan Untuk merekam Kembali atau stop dengan ctrl + c ({i+1}/{n_kali})")


def rekam_audio_background(save_path, n_kali=5):
    input("Untuk merekam audio background tekan Enter : ")
    for i in range (n_kali):
        fs = 44100
        detik = 3
        rekaman = sd.rec(int(detik*fs), samplerate = fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, rekaman)
        print(f"Sekarang Sedang Berada Di : {i+1}/{n_kali}")
        
        
#print("Merekam WakeWord Audio: \n")
#rekam_wakeword_audio("dataaudio/")

#print("Merekam WakeWord Audio: \n")
#rekam_audio_background("bgaudio/")

