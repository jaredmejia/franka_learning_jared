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


parser = argparse.ArgumentParser(
    description="Extract features from expert teleop demonstrations and fit to KNN"
)
parser.add_argument(
    "--backbone_cfg",
    default="/home/vdean/jared_contact_mic/avid-glove/config/gloveaudio/avcat-avid-ft-jointloss.yaml",
    help="path to config file associated with backbone model for feature extracting",
)
parser.add_argument(
    "--teleop_pickle",
    default="/home/vdean/franka_learning_jared/jared_chopping_exps.pkl",
    help="path to teleop demonstration data pickled object",
)
parser.add_argument(
    "--teleop_dir", default="/home/vdean/franka_demo/logs/jared_chopping_exps/"
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory to save fitted KNN model and corresponding actions to",
)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", default=256)


def get_teleop_data(teleop_paths, teleop_dir):
    actions = []
    img_paths = []
    for path in teleop_paths:
        actions.append(path["actions"])
        data_path = os.path.join(teleop_dir, path["traj_id"])

        for img_file in path["cam0c"]:
            img_path = os.path.join(data_path, img_file)
            img_paths.append(img_path)

    actions = np.vstack(actions)

    return actions, img_paths


def get_knn_features(dl, fe_model):
    feature_list = []
    fe_model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dl)):
            features, _, _ = fe_model(data)
            feature_list.append(features)
    representations = torch.cat(feature_list, dim=0)
    return representations.cpu()


def main():
    args = parser.parse_args()

    backbone_cfg = misc.convert2namespace(yaml.safe_load(open(args.backbone_cfg)))
    model = LightningModel.load_from_checkpoint(
        os.path.join(backbone_cfg.save_path, backbone_cfg.name, "checkpoints/last.ckpt")
    )
    fe_model = get_feature_extractor(model.model, unimodal=False).to(args.device)

    with open(args.teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    actions, img_paths = get_teleop_data(teleop_paths, args.teleop_dir)
    print(f"Total number of images: {len(img_paths)}")
    print(f"Actions shape: {actions.shape}")

    img_audio_prep = LivePrep(backbone_cfg.dataset, img_paths, args.device)
    dl = DataLoader(dataset=img_audio_prep, batch_size=args.batch_size, shuffle=False)

    representations = get_knn_features(dl, fe_model)
    print(f"Embeddings shape: {representations.shape}")

    knn_model = KDTree(data=representations)
    model_fn = os.path.join(args.output_dir, "knn_model.pkl")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    joblib.dump(knn_model, model_fn)

    actions_fn = os.path.join(args.output_dir, "actions.txt")
    np.savetxt(actions_fn, actions)


if __name__ == "__main__":
    main()
