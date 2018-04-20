import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

except_list = []
file_path = '/Users/gta/Desktop/heartbeat-sounds/set_b/'
save_file_path = '/Users/gta/Desktop/heartbeat-sounds/b_images/'
for root, sub, files in os.walk(file_path):
    files = sorted(files)
    for f in files:

        try:
            fig = plt.figure()
            sample_rate, samples = wavfile.read(file_path + f)
            frequencies, times, spectogram = signal.spectrogram(samples, sample_rate, nfft=512, nperseg=512)
            plt.pcolormesh(times, frequencies, spectogram)
            plt.imshow(spectogram)
            plt.axis('off')
            fig.savefig(save_file_path + f.split('.')[0], bbox_inches='tight')
            plt.close('all')
            print('File: ', f)

        except:
            except_list.append(f)
print(except_list)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
# https://stackoverflow.com/questions/48582986/scipy-signal-spectrogram-frequency-resolution
# https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
# https://stackoverflow.com/questions/16032389/pad-inches-0-and-bbox-inches-tight-makes-the-plot-smaller-than-declared-figsiz
# https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
# https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
# https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
