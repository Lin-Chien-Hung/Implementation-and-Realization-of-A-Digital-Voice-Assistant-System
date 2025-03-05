"""This module is utils of text-to-speech
"""

from os import PathLike
from pathlib import Path
from typing import Generator, Iterable

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def from_pretrained(checkpoint_dir: PathLike, device="cpu") -> tuple[XttsConfig, Xtts]:
    """Load pretrained model.

    Args:
        checkpoint_dir (PathLike): XTTS path.
        device (str, optional): Inference device. Defaults to "cpu".

    Returns:
        tuple[XttsConfig, Xtts]: (config, model)
    """
    config_path = Path.joinpath(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        use_deepspeed=True,
    )

    device = device.lower()
    if device in ("cuda", "gpu"):
        model.cuda()

    return (config, model)


def synthesize(
    config: XttsConfig, model: Xtts, inputs: str, speaker_wav: Iterable, *args, **kwargs
) -> Generator:
    """_summary_

    Args:
        config (XttsConfig): XTTS config.
        model (Xtts): XTTS model.
        inputs (str): Input text.
        speaker_wav (Iterable): Cloned speaker.

    Returns:
        Iterable: Stream output.
    """
    language = "zh-cn"
    if language not in config.languages:
        raise ValueError(f"{language} isn't supported language.")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(speaker_wav)
    chunks = model.inference_stream(
        inputs,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        enable_text_splitting=True,
        *args,
        **kwargs,
    )
    return chunks
