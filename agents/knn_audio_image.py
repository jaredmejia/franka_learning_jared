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

        self.KDTree = joblib.load(knn_model_path)
        self.actions = np.loadtxt(actions_path)
        self.k = 1
        self.device = device
        self.curr_audio_len = 0

    def get_features(self, input_prepped):
        self.fe_model.eval()
        with torch.no_grad():
            features, _, _ = self.fe_model(input_prepped)
        assert features.shape == torch.Size([1, 1024])
        return features.cpu()


    def predict(self, sample):
        img = sample["cam0c"]

        # audio_tuple, _  = sample["audio"]
        # audio_arr = np.array(list(audio_tuple)).T
        audio_list = [np.array(list(audio_tuple)) for audio_tuple, _ in sample['audio'][2:]]  # skipping 'start
        self.curr_audio_len = len(audio_list) - self.curr_audio_len
        audio_list = audio_list[:max(256, self.curr_audio_len)]
        audio_arr = np.vstack(audio_list).T

        sample_prepped = self.img_audio_prep(img, audio_arr, predict=True)
        sample_features = self.get_features(sample_prepped)

        knn_dis, knn_idx = self.KDTree.query(sample_features, k=self.k)

        return_action = self.actions[knn_idx[0][0]] ### TODO: works for k=1 only so far!!!
        print(f'action_idx: {knn_idx[0][0]}')
        print(knn_idx)

        actions = [self.actions[i] for i in knn_idx[0]]
        weights = [-1*(knn_dis[0][i]) for i in range(self.k)]
        weights = special.softmax(weights)
        return_action = np.zeros(7)
        for i in range(self.k):
            return_action += weights[i] * actions[i]


        return return_action
        # return np.array([0.142, 0.613, -0.1899, -0.6505, 0.0947, -0.3041, 0.522])  # start_position

def identity_transform(img):
    return img

def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudioImage................")
    transforms = identity_transform
    knn_agent = KNNAudioImage(config.knn.k, config.agent.backbone_cfg, config.data.extract_dir)
    return knn_agent, transforms
