import torch
import numpy

xxx = torch.load("checkpoint_033.pth")
torch.save(state_dict, "checkpoint_033.pth", _use_new_zipfile_serialization=False)
