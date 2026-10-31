"""Small helpers shared by the NHMM training and evaluation scripts."""
import json
import random

import numpy as np
import torch


def seed_everything(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)


def json_default(value):
    return value.item() if isinstance(value, np.generic) else str(value)


def write_json(path, obj, indent=2):
    path.write_text(json.dumps(obj, indent=indent, default=json_default))


def save_checkpoint(path, model, config, normalization, **extra):
    torch.save({"model": model.state_dict(), "config": config, "duration_normalization": normalization, **extra}, path)
