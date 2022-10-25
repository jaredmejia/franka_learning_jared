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
from inference import LivePrep, LivePrepCat, LivePrepVideo, LivePrepAudio, get_feature_extractor
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
parser.add_argument(
    "--num_images_cat",
    default=1,
    type=int,
    help="If greater than 1, combines multiple images/audio to emulate video (as opposed to single frame passed to model)"
)
parser.add_argument("--unimodal", default=False, type=bool)
parser.add_argument("--batch_size", default=512)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def get_teleop_data_cat(teleop_paths, teleop_dir, num_images_cat=8):
    actions = []
    img_path_lists = []
    for path in teleop_paths:
        data_path = os.path.join(teleop_dir, path["traj_id"])
        traj_img_paths = [os.path.join(data_path, img_file) for img_file in path["cam0c"]]
        for i in range(num_images_cat, len(traj_img_paths)):
            # taking chunks of size num_cat_images
            img_path_cat = traj_img_paths[i-num_images_cat:i]  
            img_path_lists.append(img_path_cat)
        
        actions.append(path["actions"][num_images_cat:])
    
    actions = np.vstack(actions)

    return actions, img_path_lists


def collate_fn_joint(batch):
    batched_audio, batched_video = [], []
    for data in batch:
        batched_audio.append(data['audio'])
        batched_video.append(data['video'])
    batched_audio = torch.stack(batched_audio, dim=0).to(device)
    batched_video = torch.stack(batched_video, dim=0).to(device)
    batched_data = {'video': batched_video, 'audio': batched_audio}
    return batched_data

def collate_fn_video(batch):
    batched_video = []
    for data in batch:
        batched_video.append(data['video'])
    batched_video = torch.stack(batched_video, dim=0).to(device)
    batched_data = {'video': batched_video}
    return batched_data

def collate_fn_audio(batch):
    batched_audio = []
    for data in batch:
        batched_audio.append(data['audio'])
    batched_audio = torch.stack(batched_audio, dim=0).to(device)
    batched_data = {'audio': batched_audio}
    return batched_data


def get_knn_features(dl, fe_model, unimodal=False):
    feature_list = []
    fe_model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dl)):
            if unimodal:
                features = fe_model(data)
            else:
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
    fe_model = get_feature_extractor(model.model, unimodal=args.unimodal).to(device)

    with open(args.teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    if args.num_images_cat > 1:
        actions, img_path_lists = get_teleop_data_cat(teleop_paths, args.teleop_dir, args.num_images_cat)
        print(f"Total number of samples: {len(img_path_lists)}")
        print(f"Actions shape: {actions.shape}")

        if args.unimodal:
            if backbone_cfg.model.args.modality == "video":
                prep_data = LivePrepVideo(backbone_cfg.dataset, image_path_lists=img_path_lists, device=device)
                collate_fn = collate_fn_video
            else:
                prep_data = LivePrepAudio(backbone_cfg.dataset, image_path_lists=img_path_lists, device=device)
                collate_fn = collate_fn_audio
        else:
            prep_data = LivePrepCat(backbone_cfg.dataset, image_path_lists=img_path_lists, device=device)
            collate_fn = collate_fn_joint
    else:
        actions, img_paths = get_teleop_data(teleop_paths, args.teleop_dir)
        print(f"Total number of images: {len(img_paths)}")
        print(f"Actions shape: {actions.shape}")

        prep_data = LivePrep(backbone_cfg.dataset, image_paths=img_paths, device=device)
        collate_fn = collate_fn_joint

    dl = DataLoader(
        dataset=prep_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    representations = get_knn_features(dl, fe_model, unimodal=args.unimodal)
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
