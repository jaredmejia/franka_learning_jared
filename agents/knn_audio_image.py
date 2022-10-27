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
from inference import LivePrep, get_feature_extractor
from main import LightningModel
from utils import misc


class KNNAudioImage(object):
    def __init__(self, k, backbone_cfg, extract_dir, H=1, device="cuda:0"):
        # loading backbone
        cfg = misc.convert2namespace(yaml.safe_load(open(backbone_cfg)))
        print(f"cfg save paths: {cfg.save_path}")
        model = LightningModel.load_from_checkpoint(
            os.path.join(cfg.save_path, cfg.name, "checkpoints/last.ckpt")
        )
        self.fe_model = get_feature_extractor(model.model, unimodal=False).to(device)
        self.img_audio_prep = LivePrep(
            db_cfg=cfg.dataset, image_paths=None, device=device
        )

        self.KDTree, self.actions, self.traj_ids = self.load_expert(extract_dir)
        self.k = k
        self.H = H  # horizon length
        self.action_idx = 0  # 0 -> 50
        self.traj_id = None

        self.device = device
        self.num_cat = 8
        self.image_window = deque([])
        self.audio_window = deque([])
        self.start_position = np.array(
            [0.142, 0.613, -0.1899, -0.6505, 0.0947, -0.3041, 0.522]
        )

    def load_expert(self, extract_dir):
        knn_model_path = os.path.join(extract_dir, "knn_model.pkl")
        actions_path = os.path.join(extract_dir, "action_trajs.pkl")
        traj_ids_path = os.path.join(extract_dir, "traj_ids.pkl")
        knn_model = joblib.load(knn_model_path)
        with open(actions_path, "rb") as f:
            actions = pickle.load(f)
        with open(traj_ids_path, "rb") as f:
            traj_ids = pickle.load(f)
        return knn_model, actions, traj_ids

    def get_features(self, sample):
        img = sample["cam0c"]
        audio_tuple, _ = sample["audio"]
        audio_arr = np.array(list(audio_tuple), dtype=np.float64).T

        self.image_window.append(img)
        self.audio_window.append(audio_arr)

        if len(self.image_window) < self.num_cat + 1:
            return None
        else:
            self.image_window.popleft()
            self.audio_window.popleft()

        img_input = list(self.image_window)
        audio_input = np.concatenate(list(self.audio_window), axis=1)
        sample_prepped = self.img_audio_prep(img_input, audio_input, predict=True)

        self.fe_model.eval()
        with torch.no_grad():
            features, _, _ = self.fe_model(sample_prepped)
        sample_features = features.cpu()
        return sample_features

    def predict(self, sample):
        if self.H == 1:
            sample_features = self.get_features(sample)

            if sample_features is None:
                print(f"returning to starting position")
                return self.start_position

            knn_dis, knn_idx = self.KDTree.query(sample_features, k=self.k)

            if self.k == 1:
                traj_id, traj_sub_idx = self.traj_ids[knn_idx[0][0]]
                return_action = self.actions[traj_id][traj_sub_idx]
            else:  # weighting top k by distances
                k_actions = []
                for i in knn_idx[0]:
                    traj_id, traj_sub_idx = self.traj_ids[i]
                    k_actions.append(self.actions[traj_id][traj_sub_idx])
                # actions = [self.actions[i] for i in knn_idx[0]]
                weights = [-1 * (knn_dis[0][i]) for i in range(self.k)]
                weights = special.softmax(weights)
                return_action = np.zeros(7)
                for i in range(self.k):
                    return_action += weights[i] * k_actions[i]
        else:  # open loop
            # begin trajectory if previous horizon reached, or first trajectory
            if self.traj_id is None:
                print(f"action_idx: {self.action_idx}")
                sample_features = self.get_features(sample)

                if sample_features is None:
                    print(f"returning to starting position")
                    return self.start_position

                knn_dis, knn_idx = self.KDTree.query(sample_features, k=self.k)
                self.traj_id, self.start_action_idx = self.traj_ids[knn_idx[0][0]]
                self.action_idx = 0
                print(f"Beginning trajectory: {self.traj_id}")

            if (
                self.action_idx + self.start_action_idx
                < self.actions[self.traj_id].shape[0]
            ):
                return_action = self.actions[self.traj_id][
                    self.action_idx + self.start_action_idx
                ]
                self.action_idx += 1
                if (
                    self.action_idx + self.start_action_idx
                    == self.actions[self.traj_id].shape[0]
                    or self.action_idx >= self.H
                ):
                    print(f"Finished KNN trajectory")
                    self.traj_id = None

        return return_action


def identity_transform(img):
    return img


def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudioImage................")
    transforms = identity_transform
    knn_agent = KNNAudioImage(
        k=config.knn.k,
        backbone_cfg=config.agent.backbone_cfg,
        extract_dir=config.data.extract_dir,
        H=config.knn.H,
    )
    return knn_agent, transforms
