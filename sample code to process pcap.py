import numpy as np
import matplotlib.pyplot as plt
import interleaved as decoder

activity_samples = decoder.read_pcap("any.pcap") # upload pcap file path

print("Activity CSI shape:", activity_samples.csi.shape)
print("Bandwidth:", empty_samples.bandwidth)

# creating windown to analyze the data and pattern

duration_activity = 600

fs_activity = amp_activity.shape[0] / duration_activity

def segment_windows(amplitude, fs, window_size=60):
    samples_per_window = int(fs * window_size)
    windows = []

    for start in range(0, len(amplitude) - samples_per_window, samples_per_window):
        windows.append(amplitude[start:start + samples_per_window])

    return windows

activity_windows = segment_windows(amp_activity, fs_activity)

print("Empty windows:", len(empty_windows))
print("Activity windows:", len(activity_windows))


# checking variance

def preprocess(window):
    window = window - np.mean(window, axis=0)
    return window

activity_var = []

for w in activity_windows:
    activity_var.append(np.var(preprocess(w)))

plt.plot(activity_var, label="Activity")
plt.legend()
plt.title("Variance per 60-sec Window")
plt.show()

# spectogram

from scipy.signal import stft

sample_window = preprocess(activity_windows[0])
signal = sample_window[:, 10]  # choose subcarrier index

f, t, Zxx = stft(signal, fs=fs_activity)

plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title("Spectrogram - Activity")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.colorbar()
plt.show()

