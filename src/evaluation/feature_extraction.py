"""Extraction de caractéristiques manuelles depuis les spectrogrammes pour les baselines ML."""

import numpy as np
from scipy.stats import kurtosis, skew


def extract_spectrogram_features(spec):
    # Extraire le vecteur de caractéristiques d'un spectrogramme log-magnitude 2D
    features = []

    # Statistiques globales
    features.append(spec.mean())
    features.append(spec.std())
    features.append(spec.max())
    features.append(spec.min())
    features.append(np.median(spec))
    features.append(kurtosis(spec.flatten()))
    features.append(skew(spec.flatten()))

    # Caractéristiques d'énergie
    total_energy = np.sum(spec ** 2)
    features.append(total_energy)
    features.append(total_energy / spec.size)  # énergie moyenne

    # Caractéristiques spectrales (moyennées sur les trames temporelles)
    freq_means = spec.mean(axis=1)  # puissance moyenne par bande de fréquence
    n_freq = len(freq_means)
    freq_axis = np.arange(n_freq)

    # Centroïde spectral
    if freq_means.sum() != 0:
        spectral_centroid = np.sum(freq_axis * np.abs(freq_means)) / np.sum(np.abs(freq_means))
    else:
        spectral_centroid = 0.0
    features.append(spectral_centroid)

    # Largeur de bande spectrale
    if freq_means.sum() != 0:
        spectral_bw = np.sqrt(
            np.sum(((freq_axis - spectral_centroid) ** 2) * np.abs(freq_means))
            / np.sum(np.abs(freq_means))
        )
    else:
        spectral_bw = 0.0
    features.append(spectral_bw)

    # Rolloff spectral (fréquence en dessous de laquelle 85% de l'énergie est concentrée)
    cumsum = np.cumsum(np.abs(freq_means))
    if cumsum[-1] > 0:
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        spectral_rolloff = rolloff_idx / n_freq
    else:
        spectral_rolloff = 0.0
    features.append(spectral_rolloff)

    # Planéité spectrale (moyenne géométrique / moyenne arithmétique)
    abs_means = np.abs(freq_means) + 1e-10
    log_mean = np.mean(np.log(abs_means))
    spectral_flatness = np.exp(log_mean) / np.mean(abs_means)
    features.append(spectral_flatness)

    # Caractéristiques temporelles (moyennées sur les bandes de fréquence)
    time_means = spec.mean(axis=0)
    features.append(time_means.std())  # variation temporelle
    features.append(kurtosis(time_means))
    features.append(skew(time_means))

    # Rapports d'énergie par bande (spectre divisé en 4 bandes)
    band_size = n_freq // 4
    for i in range(4):
        start = i * band_size
        end = (i + 1) * band_size if i < 3 else n_freq
        band_energy = np.sum(spec[start:end, :] ** 2)
        features.append(band_energy / (total_energy + 1e-10))

    # Contraste spectral (difference entre pics et creux par bande)
    for i in range(4):
        start = i * band_size
        end = (i + 1) * band_size if i < 3 else n_freq
        band = spec[start:end, :]
        if band.size > 0:
            features.append(band.max() - band.min())
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def extract_features_from_dataset(dataset, max_samples=None):
    # Extraire les caractéristiques d'un dataset PyTorch de spectrogrammes, retourner (X, y)
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    X_list = []
    y_list = []

    for i in range(n):
        spec_tensor, label = dataset[i]
        spec = spec_tensor.squeeze(0).numpy()  # supprimer la dimension du canal
        features = extract_spectrogram_features(spec)
        X_list.append(features)
        y_list.append(label.item())

    return np.array(X_list), np.array(y_list)
