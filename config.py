import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

def resolve_data_dir(garment="majca"):
    """Return data dir for garment: HDD_data/data/<g> if HDD_data/ exists, else data/<g>."""
    if "DATA_ROOT" in os.environ:
        return os.path.join(os.environ["DATA_ROOT"], garment)
    if os.path.isdir(os.path.join(_HERE, "HDD_data")):
        return os.path.join(_HERE, "HDD_data", "data", garment)
    return os.path.join(_HERE, "data", garment)

class Config:
    def __init__(self, data_dir=None):
        self.data    = DataConfig(data_dir)
        self.train   = TrainConfig()
        self.record  = RecordConfig()


class DataConfig:
    def __init__(self, data_dir=None):
        base = data_dir or resolve_data_dir("majca")
        self.partial_dir  = os.path.join(base, "partial")
        self.full_dir     = os.path.join(base, "full")
        self.train_split  = 0.9
        self.num_workers  = 8


class TrainConfig:
    def __init__(self):
        self.device       = torch.device("cuda")
        self.batch_size   = 32
        self.epochs       = 200
        self.lr           = 1e-3
        self.weight_decay = 1e-5
        self.cd_alpha     = 0.5    # fine CD weight  : total = cd_coarse + cd_alpha * cd_fine + uv_beta * uv_mse
        self.uv_beta      = 0.5    # UV MSE weight
        self.uv_warmup    = 30     # epochs before UV loss is enabled
        self.save_every   = 10     # epochs between checkpoints


class RecordConfig:
    def __init__(self):
        self.log_interval = 50   # batches between wandb log entries
