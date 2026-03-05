import numpy as np
import matplotlib.pyplot as plt
import interleaved as decoder

activity_samples = decoder.read_pcap("any.pcap") // upload pcap file path

print("Activity CSI shape:", activity_samples.csi.shape)
print("Bandwidth:", empty_samples.bandwidth)
