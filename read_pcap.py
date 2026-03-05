import os
import numpy as np

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

    rssi = np.frombuffer(rssi, dtype=np.int8, count=nsamples)

    return SampleSet((rssi, fctl, mac, seq, css, csi_cmplx), bandwidth)
