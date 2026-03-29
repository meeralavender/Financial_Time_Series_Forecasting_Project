import numpy as np
from scipy.signal import stft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def compute_multichannel_spectrogram(window_data, nperseg=16, noverlap=8):
    channels = []

    for feature_idx in range(window_data.shape[1]):
        signal = window_data[:, feature_idx]
        _, _, zxx = stft(signal, nperseg=nperseg, noverlap=noverlap)
        spectrogram = np.abs(zxx)
        channels.append(spectrogram)

    return np.stack(channels, axis=-1)


def prepare_spectrogram_dataset(X_windows, nperseg=16, noverlap=8):
    specs = []
    for window in X_windows:
        spec = compute_multichannel_spectrogram(window, nperseg=nperseg, noverlap=noverlap)
        specs.append(spec)

    return np.array(specs)


def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def get_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )