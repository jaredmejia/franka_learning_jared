from collections import deque
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
from scipy import special

sys.path.insert(1, "/home/vdean/jared_contact_mic/avid-glove")

# avid-glove imports
from inference import LivePrepAudio, get_feature_extractor
from main import LightningModel
from utils import misc


class KNNAudio(object):
    def __init__(self, k, backbone_cfg, extract_dir, device="cuda:0"):
        # loading backbone
        cfg = misc.convert2namespace(yaml.safe_load(open(backbone_cfg)))
        print(f"cfg save paths: {cfg.save_path}")
        model = LightningModel.load_from_checkpoint(
            os.path.join(cfg.save_path, cfg.name, "checkpoints/last.ckpt")
        )
        self.fe_model = get_feature_extractor(model.model, unimodal=True).to(device)
        self.audio_prep = LivePrepAudio(
            db_cfg=cfg.dataset, image_paths=None, device=device
        )

        # loading fitted knn
        knn_model_path = os.path.join(extract_dir, "knn_model.pkl")
        actions_path = os.path.join(extract_dir, "actions.txt")

        self.KDTree = joblib.load(knn_model_path)
        self.actions = np.loadtxt(actions_path)
        self.k = k
        self.device = device
        self.num_cat = 8
        self.audio_window = deque([])
        self.start_position = np.array(
            [0.142, 0.613, -0.1899, -0.6505, 0.0947, -0.3041, 0.522]
        )

    def get_features(self, input_prepped):
        self.fe_model.eval()
        with torch.no_grad():
            features = self.fe_model(input_prepped)
        return features.cpu()

    def predict(self, sample):
        audio_tuple, _ = sample["audio"]
        audio_arr = np.array(list(audio_tuple), dtype=np.float64).T
        self.audio_window.append(audio_arr)

        if len(self.audio_window) < self.num_cat + 1:
            print(f"returning to starting position")
            return self.start_position
        else:
            self.audio_window.popleft()

        audio_input = np.concatenate(list(self.audio_window), axis=1)
        sample_prepped = self.audio_prep(audio_input, predict=True)
        sample_features = self.get_features(sample_prepped)

        knn_dis, knn_idx = self.KDTree.query(sample_features, k=self.k)

        return_action = self.actions[
            knn_idx[0][0]
        ]  ### TODO: works for k=1 only so far!!!
        print(f"action_idx: {knn_idx[0][0]}")
        print(knn_idx)

        actions = [self.actions[i] for i in knn_idx[0]]
        weights = [-1 * (knn_dis[0][i]) for i in range(self.k)]
        weights = special.softmax(weights)
        return_action = np.zeros(7)
        for i in range(self.k):
            return_action += weights[i] * actions[i]

        return return_action


def identity_transform(img):
    return img


def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudio................")
    transforms = identity_transform
    knn_agent = KNNAudio(
        config.knn.k, config.agent.backbone_cfg, config.data.extract_dir
    )
    return knn_agent, transforms
