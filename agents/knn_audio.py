from collections import deque
from sklearn.neighbors import KDTree
from .knn_audio_image import KNNAudioImage
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
from scipy import special

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from knn_teleop_features import load_model

sys.path.insert(1, "/home/vdean/jared_contact_mic/avid-glove")
# avid-glove imports
from inference import LivePrepAudio, get_feature_extractor
from main import LightningModel
from utils import misc


class KNNAudio(KNNAudioImage):
    def __init__(
        self,
        k,
        backbone_cfg,
        extract_dir,
        H=1,
        pretraining="finetuned",
        device="cuda:0",
    ):
        # loading backbone
        cfg = misc.convert2namespace(yaml.safe_load(open(backbone_cfg)))
        print(f"cfg save paths: {cfg.save_path}")
        model = load_model(cfg, pretraining)
        self.fe_model = get_feature_extractor(model.model, unimodal=True).to(device)
        self.audio_prep = LivePrepAudio(
            db_cfg=cfg.dataset, image_path_lists=None, device=device
        )

        self.KDTree, self.actions, self.traj_ids = self.load_expert(extract_dir)
        self.k = k
        self.H = H  # horizon length
        self.action_idx = 0  # 0 -> 50
        self.traj_id = None

        self.device = device
        self.num_cat = 8
        self.audio_window = deque([])
        self.start_position = np.array(
            [0.142, 0.613, -0.1899, -0.6505, 0.0947, -0.3041, 0.522]
        )

    def update_window(self, sample):
        audio_tuple, _ = sample["audio"]
        audio_arr = np.array(list(audio_tuple), dtype=np.float64).T

        self.audio_window.append(audio_arr)

        if len(self.audio_window) < self.num_cat + 1:
            return False
        else:
            self.audio_window.popleft()

        return True

    def get_features(self, sample):
        audio_input = np.concatenate(list(self.audio_window), axis=1)
        sample_prepped = self.audio_prep(audio_input, predict=True)

        self.fe_model.eval()
        with torch.no_grad():
            features = self.fe_model(sample_prepped)
        sample_features = features.detach().cpu()
        return sample_features


def identity_transform(img):
    return img


def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudio................")
    transforms = identity_transform
    finetuned = config.agent.finetuned.lower() == "true"
    knn_agent = KNNAudio(
        k=config.knn.k,
        backbone_cfg=config.agent.backbone_cfg,
        extract_dir=config.data.extract_dir,
        H=config.knn.H,
        finetuned=finetuned,
    )
    return knn_agent, transforms
