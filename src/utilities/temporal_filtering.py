import math
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, TR, order=5):
    fs = 1.0 / TR
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, TR, order=5):
    fs = 1.0 / TR
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def discrete_cosine(nvol, TR, lowcut=None, highcut=None):
    n = np.arange(0, nvol)

    if lowcut is None:
        range_lowcut = []
    else:
        range_lowcut = np.arange(math.ceil(2 * (nvol * TR) / (1 / lowcut)), nvol)

    if highcut is None:
        range_highcut = []
    else:
        range_highcut = np.arange(1, math.ceil(2 * (nvol * TR) / (1 / highcut)))

    K = np.hstack((range_highcut, range_highcut))
    
    if len(K) > 0:
        C = np.sqrt(2 / nvol) * np.cos(np.pi * (2 * n[:, None] + 1) * K / (2 * nvol))
    else:
        C = np.empty((nvol, 0))

    return C