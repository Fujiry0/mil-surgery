import os
import random

import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(root_model, store_name, state, epoch=None):
    filename = f"{root_model}/{store_name}/ckpt.pth.tar"
    torch.save(state, filename)
    if epoch:
        filename = f"{root_model}/{store_name}/{epoch}_ckpt.pth.tar"
        torch.save(state, filename)
