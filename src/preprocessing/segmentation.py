import numpy as np


def segment_boundaries(signal_length, window_size=131072, hop_size=65536):
    """Return (start, end) boundaries for sliding window segments."""
    boundaries = []
    for start in range(0, signal_length - window_size + 1, hop_size):
        boundaries.append((start, start + window_size))
    return boundaries


def segment_signal(signal, window_size=131072, hop_size=65536):
    """Split a 1D signal into sliding window segments."""
    boundaries = segment_boundaries(len(signal), window_size, hop_size)
    return [signal[start:end] for start, end in boundaries]


if __name__ == "__main__":
    x = np.arange(20)
    segs = segment_signal(x, window_size=8, hop_size=4)
    for i, s in enumerate(segs):
        print(f"Segment {i}: {s}")