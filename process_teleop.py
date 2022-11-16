from sklearn.neighbors import KDTree
import argparse
import joblib
import numpy as np
import os
import pickle
import sys
import torch
import yaml

from PIL import Image
from pretraining import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(1, "/home/vdean/jared_contact_mic/avid-glove")

from utils import misc


parser = argparse.ArgumentParser(
    description="Extract features from expert teleop demonstrations based on model config and fit KNN"
)
parser.add_argument("--cfg", help="path to config file associated with encoder model")
parser.add_argument("--model_name", required=True, help="name of encoder model")
parser.add_argument(
    "--teleop_pickle",
    default="/home/vdean/franka_learning_jared/latest_jared_chopping_exps.pkl",
    help="path to teleop demonstration data pickled object",
)
parser.add_argument(
    "--teleop_dir", default="/home/vdean/franka_demo/logs/jared_chopping_exps/"
)
parser.add_argument(
    "--num_images_cat",
    default=8,
    type=int,
    help="If greater than 1, combines multiple images/audio to emulate video (as opposed to single frame passed to model)",
)
parser.add_argument("--batch_size", default=256)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory to save fitted KNN model and corresponding actions to",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TeleopDataset(Dataset):
    def __init__(self, img_paths, transforms, num_images_cat=8, unimodal=False):
        self.img_paths = img_paths
        self.transforms = transforms
        self.num_images_cat = num_images_cat
        self.unimodal = unimodal

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_list = []
        txt_list = []
        for img_path in self.img_paths[idx]:
            img = Image.open(img_path)
            img_list.append(img)

            if not self.unimodal:
                txt_path = f"{img_path[:-4]}txt"
                txt_arr = np.loadtxt(txt_path)
                txt_arr = txt_arr.T
                txt_list.append(txt_arr)

        if not self.unimodal:
            audio_data = np.concatenate(txt_list, axis=1)
        else:
            audio_data = None

        if len(img_list) == 1:
            img_data = img_list[0]
            video_data = None
        else:
            video_data = img_list
            img_data = None

        data = {"video": video_data, "image": img_data, "audio": audio_data}
        data_t = self.transforms(data)

        return data_t


def multi_collate(batch):
    video_batch, img_batch, audio_batch = [], [], []
    include_video, include_img, include_audio = True, True, True
    for data in batch:
        if "video" not in data.keys() or not include_video:
            include_video = False
        else:
            video_batch.append(data["video"])

        if "image" not in data.keys() or not include_img:
            include_img = False
        else:
            img_batch.append(data["image"])

        if "audio" not in data.keys() or not include_audio:
            include_audio = False
        else:
            audio_batch.append(data["audio"])

    batched_data = {}
    if include_video:
        video_batch = torch.stack(video_batch, dim=0).to(device)
        batched_data["video"] = video_batch
    if include_img:
        img_batch = torch.stack(img_batch, dim=0).to(device)
        batched_data["image"] = img_batch
    if include_audio:
        audio_batch = torch.stack(audio_batch, dim=0).to(device)
        batched_data["audio"] = audio_batch

    return batched_data


def get_teleop_data(teleop_paths, teleop_dir, num_images_cat=8):
    actions = {}
    traj_ids = []
    img_paths = []
    for path in teleop_paths:
        data_path = os.path.join(teleop_dir, path["traj_id"])
        traj_img_paths = [
            os.path.join(data_path, img_file) for img_file in path["cam0c"]
        ]

        for i in range(num_images_cat, len(traj_img_paths)):
            # taking chunks of size num_images_cat
            img_path_cat = traj_img_paths[i - num_images_cat : i]
            img_paths.append(img_path_cat)
            traj_ids.append((path["traj_id"], i - num_images_cat))

        actions[path["traj_id"]] = path["actions"][num_images_cat:]

        assert len(actions[path["traj_id"]]) == len(traj_img_paths) - num_images_cat, (
            f"Number of actions does not match number of images for trajectory {path['traj_id']}:"
            f"\n\tnum actions: {len(actions[path['traj_id']])}"
            f"\n\tnum images: {len(traj_img_paths) - num_images_cat}"
        )

    return actions, traj_ids, img_paths


def get_embeddings(dl, model):
    """computes embeddings from dataloader and pretrained model"""
    emb_list = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dl)):
            emb = model(data)
        emb_list.append(emb.detach())
    embeddings = torch.cat(emb_list, dim=0)
    return embeddings.cpu()


def main():
    args = parser.parse_args()
    cfg = misc.convert2namespace(yaml.safe_load(open(args.cfg)))

    model = load_encoder(args.model_name, cfg).to(device)
    transforms = load_transforms(args.model_name, cfg)

    with open(args.teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    actions, traj_ids, img_paths = get_teleop_data(
        teleop_paths, args.teleop_dir, args.num_images_cat
    )
    print(f"Total number of samples: {len(img_paths)}")
    print(f"Num action trajectories: {len(actions)}")
    print(f"Num images cat: {args.num_images_cat}")

    teleop_data = TeleopDataset(
        img_paths, transforms, num_images_cat=args.num_images_cat
    )

    dl = DataLoader(
        dataset=teleop_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multi_collate,
    )

    print(f"Getting embeddings...")
    embeddings = get_embeddings(dl, model)
    print(f"Fitting KNN...")
    knn_model = KDTree(data=embeddings)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    knn_fn = os.path.join(args.output_dir, "knn_model.pkl")
    print(f"Saving KNN to {knn_fn}")
    joblib.dump(knn_model, knn_fn)

    actions_fn = os.path.join(args.output_dir, "action_trajs.pkl")
    print(f"Saving actions to {actions_fn}")
    with open(actions_fn, "wb") as f:
        pickle.dump(actions, f)

    traj_ids_fn = os.path.join(args.output_dir, "traj_ids.pkl")
    print(f"Saving traj_ids to {traj_ids_fn}")
    with open(traj_ids_fn, "wb") as f:
        pickle.dump(traj_ids, f)


if __name__ == "__main__":
    main()
