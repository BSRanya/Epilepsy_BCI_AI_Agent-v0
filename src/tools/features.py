import numpy as np
from scipy.signal import butter , filtfilt , welch

_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def _bandpass(data , low=0.5, high =45.0 , fs=256, order=4):
    nyq = 0.5 *fs
    #nyquist 
    b ,a =butter(order,[low / nyq , high/nyq], btype="band")
    return filtfilt(b,a,data , axis=-1) #filter along time

"""What it does: Applies a 4th-order Butterworth band-pass filter to keep only frequencies between 0.5 Hz and 45 Hz.

nyq = 0.5 * fs → Nyquist frequency = 128 Hz. This normalizes the cutoff frequencies to the range [0, 1] that butter expects.

low / nyq = 0.5 / 128 ≈ 0.0039; high / nyq = 45 / 128 ≈ 0.3516. These are the normalized cutoffs.

btype="band" → band-pass (keeps frequencies between low and high).

filtfilt → zero-phase filtering. It runs the filter forward and backward, canceling out phase distortion. Critical for EEG, where timing matters.

Why 0.5–45 Hz?

Below 0.5 Hz: DC drift, electrode movement artifacts.

Above 45 Hz: Muscle activity (EMG), power line noise (50/60 Hz).

The 0.5–45 Hz band captures all clinically relevant EEG rhythms: delta (0.5–4), theta (4–8), alpha (8–13), beta (13–30), and low gamma (30–45).

"""
     


def _extract_features(window , fs=256):
    window = _bandpass(window, fs=fs)
    feats =[]
    for ch in range(window.shape[0]):
        freqs ,psd = welch(window[ch] , fs=fs , nperseg= fs*2)
        total =np.trapezoid(psd , freqs) + 1e-12
        for _, (lo, hi) in _BANDS.items():
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            bp = np.trapezoid(psd[idx], freqs[idx]) if len(idx) else 0
            feats.append(bp / total)
        act = np.var(window[ch])
        mob = np.sqrt(np.var(np.diff(window[ch])) / (act + 1e-12))
        comp = np.sqrt(np.var(np.diff(np.diff(window[ch]))) / (np.var(np.diff(window[ch])) + 1e-12)) / (mob + 1e-12)
        feats.extend([act, mob, comp])
    return np.array(feats).reshape(1, -1)


