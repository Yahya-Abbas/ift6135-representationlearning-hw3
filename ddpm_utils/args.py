import os

import torch
from easydict import EasyDict


args = {
    "image_channels": 1,
    "image_size": 32,
    "n_steps": 1000,
    "nb_save": 5,
    "batch_size": 256,
    "n_samples": 16,
    "learning_rate": 2e-4,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fp16_precision": True,
    "show_every_n_epochs": 2,
    "save_every_n_epochs": 2,
    "num_workers": 0,
    "project_root": os.getcwd(),
    "ddpm_model_path": os.path.join(os.getcwd(), "q1_ddpm_model.pkl"),
    "flow_model_path": os.path.join(os.getcwd(), "q2_flow_matching_model.pkl"),
}

args["MODEL_PATH"] = args["ddpm_model_path"]
args = EasyDict(args)
