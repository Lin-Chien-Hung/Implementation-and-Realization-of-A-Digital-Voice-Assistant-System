"""This module is utils of speech enhancement.
"""

import numpy as np
import torch

from .demucs import Demucs


def try_enhancement(
    model: Demucs, audio, snr_threshold: int, device="cpu", dry=0
) -> tuple[float, np.ndarray]:
    """If 'snr' < 'snr_threshold' then perform speech enhancement; otherwise, no enhancement.

    Args:
        model (Demucs): Speech enhancement model.
        audio (_type_): Input audio signal.
        snr_threshold (int): Threshold of signal-to-noise ratio.
        device (str, optional): Inference device.
        dry (int, optional): Dry/Wet knob coefficient.
                             0 is only denoised, 1 only input signal.
                             Defaults to 0.

    Returns:
        tuple[float, np.ndarray]: (snr, enhanced_audio)
    """
    audio_tensor = torch.tensor(audio, dtype=torch.float32).view(1, -1)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        estimate = (1 - dry) * model(audio_tensor) + dry * audio_tensor

    noise = audio_tensor - estimate
    snr = signaltonoise(estimate.numpy(), noise.numpy())
    if snr < snr_threshold:
        audio_tensor = estimate

    audio_tensor = audio_tensor.view(audio_tensor.size(0), -1)  # 3D to 2D
    audio_tensor = audio_tensor / max(
        audio_tensor.abs().max().item(), 1
    )  # Normalization
    return (snr, audio_tensor.flatten().numpy())


def signaltonoise(signal, noise, db=True) -> float:
    """The signal-to-noise ratio of the input data.

    Arguments:
        signal: An array_like object.
        noise: An array_like object.
        db (optional): Defaults to True.

    Returns:
        float: Signal-to-noise ratio.
    """
    signal = np.asanyarray(signal)
    noise = np.asanyarray(noise)
    snr = np.sum(signal**2) / np.sum(noise**2)
    return 10 * np.log10(snr) if db else snr


def from_pretrained(url: str, *args, **kwargs) -> Demucs:
    """Load pretrained model.

    Arguments:
        url (str): URL of the object to download.

    Returns:
        Demucs: Pretrained model.
    """
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    kwargs["hidden"] = 64
    kwargs["sample_rate"] = 16000
    model = Demucs(*args, **kwargs)
    model.load_state_dict(state_dict)
    return model
