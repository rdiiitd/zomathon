import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import spectrogram

def __find_bandwidth(incl_len):
    pkt_len = int.from_bytes(incl_len, byteorder='little', signed=False)
    pkt_len += (128 - 60)
    bandwidth = 20 * int(pkt_len // (20 * 3.2 * 4))
    return bandwidth

def __find_nsamples_max(pcap_filesize, nsub):
    return int((pcap_filesize - 24) / (12 + 46 + 18 + (nsub*4)))

class SampleSet(object):
    def __init__(self, samples, bandwidth):
        self.rssi, self.fctl, self.mac, self.seq, self.css, self.csi = samples
        self.nsamples = self.csi.shape[0]
        self.bandwidth = bandwidth

def read_pcap(pcap_filepath, bandwidth=0, nsamples_max=0):
    pcap_filesize = os.stat(pcap_filepath).st_size
    with open(pcap_filepath, 'rb') as f:
        fc = f.read()

    if bandwidth == 0:
        bandwidth = __find_bandwidth(fc[32:36])

    nsub = int(bandwidth * 3.2)

    if nsamples_max == 0:
        nsamples_max = __find_nsamples_max(pcap_filesize, nsub)

    rssi = bytearray(nsamples_max)
    fctl = bytearray(nsamples_max)
    mac = bytearray(nsamples_max * 6)
    seq = bytearray(nsamples_max * 2)
    css = bytearray(nsamples_max * 2)
    csi = bytearray(nsamples_max * nsub * 4)

    ptr = 24
    nsamples = 0

    while ptr < pcap_filesize:
        ptr += 8
        frame_len = int.from_bytes(fc[ptr: ptr+4], byteorder='little', signed=False)
        ptr += 50

        rssi[nsamples] = fc[ptr+2]
        fctl[nsamples] = fc[ptr+3]
        mac[nsamples*6:(nsamples+1)*6] = fc[ptr+4:ptr+10]
        seq[nsamples*2:(nsamples+1)*2] = fc[ptr+10:ptr+12]
        css[nsamples*2:(nsamples+1)*2] = fc[ptr+12:ptr+14]
        csi[nsamples*(nsub*4):(nsamples+1)*(nsub*4)] = fc[ptr+18:ptr+18+nsub*4]

        ptr += (frame_len - 42)
        nsamples += 1

    csi_np = np.frombuffer(csi, dtype=np.int16, count=nsub * 2 * nsamples)
    csi_np = csi_np.reshape((nsamples, nsub * 2))

    csi_cmplx = np.fft.fftshift(
        csi_np[:, ::2] + 1.j * csi_np[:, 1::2],
        axes=(1,)
    )

    rssi_np = np.frombuffer(rssi, dtype=np.int8, count=nsamples)

    return SampleSet((rssi_np, fctl, mac, seq, css, csi_cmplx), bandwidth)

# PIPELINE STEP 1: Load CSI

def load_real_csi(filepath, assumed_fs=100):
    print(f"Loading {filepath}...")
    sample_set = read_pcap(filepath)
    csi_matrix = sample_set.csi

    num_samples = csi_matrix.shape[0]
    timestamps = np.arange(num_samples) / assumed_fs
    print(f"Loaded {num_samples} packets. Estimated duration: {num_samples/assumed_fs:.2f} seconds.")

    return timestamps, csi_matrix, assumed_fs


# PIPELINE STEP 2: Amplitude Conversion

def get_amplitude(csi_matrix):
    return np.abs(csi_matrix)

# PIPELINE STEP 3: 60-Second Segmentation

def segment_data(timestamps, amplitude_matrix, window_size=60):
    segments = []
    start_time = timestamps[0]
    end_time = timestamps[-1]

    current_start = start_time
    while current_start < end_time:
        current_end = current_start + window_size
        idx = np.where((timestamps >= current_start) & (timestamps < current_end))[0]

        if len(idx) > 0:
            segments.append({
                'start': current_start,
                'end': current_end,
                'data': amplitude_matrix[idx, :]
            })
        current_start = current_end
    return segments

# PIPELINE STEP 4: Preprocessing

def preprocess_segment(segment_data, moving_avg_window=5, top_k_subcarriers=10):
    dc_removed = segment_data - np.mean(segment_data, axis=0)

    kernel = np.ones(moving_avg_window) / moving_avg_window
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=dc_removed)

    variances = np.var(smoothed, axis=0)
    top_indices = np.argsort(variances)[-top_k_subcarriers:]
    selected_subcarriers = smoothed[:, top_indices]

    return selected_subcarriers

# PIPELINE STEP 5 & 6: Feature Extraction
  
def extract_features(preprocessed_data, fs):
    features = {}
    features['variance'] = np.mean(np.var(preprocessed_data, axis=0))
    # You can add the rest of the features back here when you move to classification
    return features

# PIPELINE STEP 7: Visualization

def run_pipeline():
    # --- CONFIGURATION ---
    # Update these if your uploaded files have different names
    FILE_EMPTY = "any.pcap" # use empty room or baseline activity to set a reference
    FILE_ACTIVITY = "any.pcap" # live files to see rush

    # Update this to match your actual ping/packet rate
    ASSUMED_FS = 100

    if not os.path.exists(FILE_EMPTY) or not os.path.exists(FILE_ACTIVITY):
        print("ERROR: PCAP files not found. Please upload them to the Colab files pane (left sidebar).")
        return

    # 1 & 2. Load Data & Get Amplitude
    ts_empty, csi_empty, fs = load_real_csi(FILE_EMPTY, assumed_fs=ASSUMED_FS)
    ts_act, csi_act, fs = load_real_csi(FILE_ACTIVITY, assumed_fs=ASSUMED_FS)

    amp_empty = get_amplitude(csi_empty)
    amp_act = get_amplitude(csi_act)

    # 3. Segment (60s)
    seg_empty = segment_data(ts_empty, amp_empty, window_size=60)
    seg_act = segment_data(ts_act, amp_act, window_size=60)

    # Process features
    var_empty = []
    var_act = []

    for s in seg_empty:
        processed = preprocess_segment(s['data'])
        feats = extract_features(processed, fs)
        var_empty.append(feats['variance'])

    for s in seg_act:
        processed = preprocess_segment(s['data'])
        feats = extract_features(processed, fs)
        var_act.append(feats['variance'])

    # --- PLOTTING ---
    plt.figure(figsize=(15, 12))

    # Plot 1: Amplitude vs Time
    plt.subplot(3, 1, 1)
    if len(seg_empty) > 0:
        plt.plot(np.linspace(0, 60, len(seg_empty[0]['data'])), seg_empty[0]['data'][:, 0], label='Empty Room', alpha=0.7)
    if len(seg_act) > 0:
        plt.plot(np.linspace(0, 60, len(seg_act[0]['data'])), seg_act[0]['data'][:, 0], label='Activity', alpha=0.7)
    plt.title("Amplitude vs Time (First 60s Window, Single Subcarrier)")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.legend()

    # Plot 2: Variance Comparison
    plt.subplot(3, 1, 2)
    max_len = max(len(var_empty), len(var_act))
    x = np.arange(max_len)
    width = 0.35

    var_empty_padded = var_empty + [0] * (max_len - len(var_empty))
    var_act_padded = var_act + [0] * (max_len - len(var_act))

    plt.bar(x - width/2, var_empty_padded, width, label='Empty')
    plt.bar(x + width/2, var_act_padded, width, label='Activity', color='orange')
    plt.title("Window Variance Comparison (Activity Detector Baseline)")
    plt.ylabel("Average Variance")
    plt.xticks(x, [f"Win {i+1}" for i in range(max_len)])
    plt.legend()

    # Plot 3: Spectrogram of Activity
    plt.subplot(3, 1, 3)
    if len(seg_act) > 0:
        active_processed = preprocess_segment(seg_act[0]['data'])
        f, t, Sxx = spectrogram(active_processed[:, 0], fs, nperseg=256, noverlap=128)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        plt.title("Spectrogram of Activity Window")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar(label='Power/Frequency [dB/Hz]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()
