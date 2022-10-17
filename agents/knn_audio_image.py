from sklearn.neighbors import KDTree
import argparse
import joblib
import numpy as np
import os
import pickle
import sys
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(1, "/home/vdean/jared_contact_mic/avid-glove")

# avid-glove imports
from inference import LivePrep, get_feature_extractor
from main import LightningModel
from utils import misc


class KNNAudioImage(object):
    def __init__(self, k, backbone_cfg, extract_dir, device="cuda:0"):
        # loading backbone
        cfg = misc.convert2namespace(yaml.safe_load(open(backbone_cfg)))
        print(f"cfg save paths: {cfg.save_path}")
        model = LightningModel.load_from_checkpoint(
            os.path.join(cfg.save_path, cfg.name, "checkpoints/last.ckpt")
        )
        self.fe_model = get_feature_extractor(model.model, unimodal=False).to(device)
        self.img_audio_prep = LivePrep(cfg.dataset, None, device)

        # loading fitted knn
        knn_model_path = os.path.join(extract_dir, "knn_model.pkl")
        actions_path = os.path.join(extract_dir, "actions.txt")

        self.knn_model = joblib.load(knn_model_path)
        self.actions = np.loadtxt(actions_path)
        self.k = 1
        self.device = device

    def predict(self, sample):
        img = sample["cam0c"]
        print(sample["audio"])

        return np.array([0.142, 0.613, -0.1899, -0.6505, 0.0947, -0.3041, 0.522])

def identity_transform(img):
    return img

def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudioImage................")
    transforms = identity_transform
    knn_agent = KNNAudioImage(config.knn.k, config.agent.backbone_cfg, config.data.extract_dir)
    return knn_agent, transforms
