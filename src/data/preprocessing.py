"""
Dysarthria AI — Audio Preprocessing

Reusable functions for loading, preprocessing, and converting audio
into model-ready Mel Spectrogram tensors.

Pipeline:
    WAV → Load → Mono → Resample → Normalize → Mel Spec → dB → Pad/Crop → Tensor
"""

from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from src.utils.config import (
    TARGET_SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    MAX_DURATION_SEC,
    TARGET_FRAMES,
)


# ============================================================
# Audio Loading
# ============================================================

def load_audio(
    filepath: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and optionally resample.

    Args:
        filepath: Path to the .wav file.
        target_sr: Target sampling rate. If None, uses the file's native rate.
                   If provided and different from native, resamples.
        mono: If True, converts to mono by averaging channels.

    Returns:
        Tuple of (audio_array, sampling_rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be loaded.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        # Load with librosa (handles resampling internally)
        audio, sr = librosa.load(
            str(filepath),
            sr=target_sr,  # None = native rate; int = resample
            mono=mono,
        )
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio '{filepath}': {e}") from e


def get_audio_info(filepath: str | Path) -> dict:
    """
    Get audio file properties without loading the full audio.

    Args:
        filepath: Path to the .wav file.

    Returns:
        Dict with: sample_rate, channels, frames, duration, subtype.
    """
    filepath = Path(filepath)
    info = sf.info(str(filepath))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "subtype": info.subtype,
    }


# ============================================================
# Audio Normalization
# ============================================================

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak amplitude normalization to [-1, 1].

    Avoids division by zero for silent audio.

    Args:
        audio: 1D numpy array of audio samples.

    Returns:
        Normalized audio array.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


# ============================================================
# Mel Spectrogram Extraction
# ============================================================

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute Mel Spectrogram and convert to dB scale.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sampling rate of the audio.
        n_mels: Number of Mel frequency bands.
        n_fft: FFT window size.
        hop_length: Hop length between successive frames.

    Returns:
        2D numpy array of shape (n_mels, n_frames) in dB scale.
    """
    # Compute Mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert power to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


# ============================================================
# Fixed-Size Padding/Cropping
# ============================================================

def fix_length_frames(
    mel_spec: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Pad or crop a Mel Spectrogram to a fixed number of time frames.

    Padding uses the minimum value in the spectrogram (silence in dB).
    Cropping takes the first target_frames.

    Args:
        mel_spec: 2D array of shape (n_mels, n_frames).
        target_frames: Desired number of time frames.

    Returns:
        2D array of shape (n_mels, target_frames).
    """
    n_mels, n_frames = mel_spec.shape

    if n_frames == target_frames:
        return mel_spec
    elif n_frames > target_frames:
        # Crop to target length
        return mel_spec[:, :target_frames]
    else:
        # Pad with minimum value (silence in dB scale)
        pad_width = target_frames - n_frames
        pad_value = mel_spec.min()
        padded = np.full((n_mels, target_frames), pad_value, dtype=mel_spec.dtype)
        padded[:, :n_frames] = mel_spec
        return padded


# ============================================================
# Full Preprocessing Pipeline
# ============================================================

def preprocess_single(
    filepath: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_frames: int = TARGET_FRAMES,
    normalize: bool = True,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single audio file.

    Pipeline:
        WAV → Load (mono) → Resample → Normalize → Mel Spec → dB → Pad/Crop

    Args:
        filepath: Path to the .wav file.
        target_sr: Target sampling rate.
        n_mels: Number of Mel bands.
        n_fft: FFT window size.
        hop_length: Hop length.
        target_frames: Fixed number of time frames.
        normalize: Whether to apply peak normalization.

    Returns:
        3D numpy array of shape (n_mels, target_frames, 1) ready for CNN input.
    """
    # Step 1: Load and resample
    audio, sr = load_audio(filepath, target_sr=target_sr, mono=True)

    # Step 2: Check for empty audio
    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {filepath}")

    # Step 3: Normalize amplitude
    if normalize:
        audio = normalize_audio(audio)

    # Step 4: Extract Mel Spectrogram (in dB)
    mel_spec = extract_mel_spectrogram(
        audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    # Step 5: Fix to target frames
    mel_spec = fix_length_frames(mel_spec, target_frames=target_frames)

    # Step 6: Add channel dimension for CNN: (n_mels, target_frames, 1)
    mel_spec = mel_spec[..., np.newaxis]

    return mel_spec


def preprocess_batch(
    filepaths: list[str | Path],
    **kwargs,
) -> np.ndarray:
    """
    Preprocess a batch of audio files.

    Args:
        filepaths: List of paths to .wav files.
        **kwargs: Arguments forwarded to preprocess_single.

    Returns:
        4D numpy array of shape (batch_size, n_mels, target_frames, 1).
    """
    specs = []
    for fp in filepaths:
        spec = preprocess_single(fp, **kwargs)
        specs.append(spec)
    return np.array(specs)
