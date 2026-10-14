"""
Dysarthria AI — Audio Feature Extraction Utilities

Additional feature extraction functions beyond Mel Spectrograms.
Currently provides helper functions for audio inspection and
visualization used in exploration notebooks.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.utils.config import (
    TARGET_SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    FIGURES_DIR,
)
from src.data.preprocessing import load_audio, extract_mel_spectrogram


# ============================================================
# Visualization Helpers
# ============================================================

def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Axes:
    """
    Plot the waveform of an audio signal.

    Args:
        audio: 1D audio array.
        sr: Sampling rate.
        title: Plot title.
        ax: Matplotlib axes. If None, creates a new figure.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_mel_spectrogram(
    mel_spec_db: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    title: str = "Mel Spectrogram (dB)",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Axes:
    """
    Plot a Mel Spectrogram in dB scale.

    Args:
        mel_spec_db: 2D array (n_mels, n_frames) in dB scale.
        sr: Sampling rate.
        hop_length: Hop length used in spectrogram extraction.
        title: Plot title.
        ax: Matplotlib axes. If None, creates a new figure.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title(title, fontsize=12)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_comparison(
    filepath_control: str | Path,
    filepath_dysarthric: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create a side-by-side comparison of waveform and Mel Spectrogram
    for a control and dysarthric speaker sample.

    Args:
        filepath_control: Path to control speaker WAV file.
        filepath_dysarthric: Path to dysarthric speaker WAV file.
        target_sr: Sampling rate for loading.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    for col, (filepath, label) in enumerate([
        (filepath_control, "Control"),
        (filepath_dysarthric, "Dysarthric"),
    ]):
        audio, sr = load_audio(filepath, target_sr=target_sr, mono=True)

        # Waveform
        plot_waveform(audio, sr, title=f"{label} — Waveform", ax=axes[0, col])

        # Mel Spectrogram
        mel_spec = extract_mel_spectrogram(audio, sr)
        plot_mel_spectrogram(mel_spec, sr, title=f"{label} — Mel Spectrogram", ax=axes[1, col])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ============================================================
# Batch Audio Info Collection
# ============================================================

def collect_audio_info(
    filepaths: list[str | Path],
    max_files: Optional[int] = None,
) -> list[dict]:
    """
    Collect audio properties for a list of files without loading full audio.

    Args:
        filepaths: List of WAV file paths.
        max_files: Maximum number of files to inspect. None = all.

    Returns:
        List of dicts with: filepath, sample_rate, channels, frames, duration.
    """
    from src.data.preprocessing import get_audio_info

    if max_files is not None:
        filepaths = filepaths[:max_files]

    results = []
    for fp in filepaths:
        try:
            info = get_audio_info(fp)
            info["filepath"] = str(fp)
            results.append(info)
        except Exception as e:
            print(f"Warning: Could not read info for {fp}: {e}")
            results.append({
                "filepath": str(fp),
                "sample_rate": None,
                "channels": None,
                "frames": None,
                "duration": None,
                "error": str(e),
            })

    return results
