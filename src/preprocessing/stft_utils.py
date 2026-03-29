import numpy as np
from scipy.signal import stft


def compute_log_spectrogram(signal, fs=1.0, nperseg=512, noverlap=256, eps=1e-10):
    """Compute normalized log-magnitude spectrogram from a 1D signal."""
    f, t, Zxx = stft(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )

    S = np.abs(Zxx)
    S_log = np.log10(S + eps)
    S_log = (S_log - S_log.mean()) / (S_log.std() + eps)

    return f, t, S_log.astype(np.float32)
