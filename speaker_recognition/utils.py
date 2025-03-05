"""This module is utils of speaker recognition.
"""

from os import PathLike
from typing import Union

import torch

from .model import BackgroundResnet


def from_pretrained(model_path: Union[str, PathLike], device="cpu") -> BackgroundResnet:
    """Load pretrained model.

    Args:
        model_path (Union[str, PathLike]): Pretrained model path.
        device (str, optional): Inference device. Defaults to "cpu".

    Returns:
        BackgroundResnet: Pretrained model.
    """
    model = BackgroundResnet(2099)

    device = device.lower()
    if device in  ("cuda", "gpu"):
        model.cuda()
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model
