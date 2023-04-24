import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

# Load the audio file

file = 'blues.00000.wav'

# waveform
# sr is sample rate

signal, sr = librosa.load(file, sr=22050) # sr * T = 22050 * 30 = 661500 samples in 30 seconds of audio file || we get the amplitude of the signal at each sample

librosa.display.waveshow(signal, sr=sr)

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Fourier Transform
# we want to convert the signal into the frequency domain

fft = np.fft.fft(signal) 

magnitude = np.abs(fft) # np.abs() returns the absolute value of the complex number || magnitude is the distance from the origin to the point on the complex plane || magnitude is the amplitude of the signal at each frequency
frequency = np.linspace(0, sr, len(magnitude)) # np.linspace() returns evenly spaced numbers over a specified interval || frequency is the x-axis of the graph || give us the number of evenly spaced number in the interval, between 0 and sample rate itself. The number of evenly spaced numbers is the same as the number of samples in the signal

left_frequency = frequency[:int(len(frequency)/2)] # we only want the first half of the frequencies
left_magnitude = magnitude[:int(len(magnitude)/2)] # we only want the first half of the magnitudes


plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.plot(left_frequency, left_magnitude)
plt.show()

# Short Time Fourier Transform
# we want to convert the signal into the frequency domain, but we want to do it in small chunks of time
n_fft = 2048 # number of samples per frame || 2048 samples per frame | window size a single fourier transform is computed on
hop_length = 512 # number of samples between successive frames || 512 samples between successive frames | how much we shift the window by each time shifting sliding window

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft) # stft is a complex-valued matrix of short-term Fourier transform coefficients

spectrogram = np.abs(stft) # spectrogram is the magnitude of the complex-valued matrix of short-term Fourier transform coefficients

log_spectrogram = librosa.amplitude_to_db(spectrogram) # log_spectrogram is the log of the spectrogram


librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('Frequency')
# plt.colorbar(mappable=None, cax=None, ax=None,) # add a color bar to the graph
plt.show()


# Mel-Frequency Cepstral Coefficients
y = signal

MFFCs = librosa.feature.mfcc(y=y, n_fft=n_fft, hop_length=hop_length, n_mfcc=13) # MFFCs are the Mel-frequency cepstral coefficients
librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('MFFC')
# plt.colorbar(mappable=None, cax=None, ax=None,) # add a color bar to the graph
plt.show()