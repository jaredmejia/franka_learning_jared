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

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import *

CONTACT_AUDIO_AVG = 2061.0

DEBUG = False


class KNNAudioImage(object):
    def __init__(
        self,
        k,
        backbone_cfg,
        extract_dir,
        encoder_name,
        H=1,
        device="cuda:0",
    ):
        # loading backbone
        cfg = yaml.safe_load(open(backbone_cfg))
        self.encoder = load_encoder(encoder_name, cfg)
        self.transforms = load_transforms(encoder_name, cfg)

        self.KDTree, self.actions, self.traj_ids, _ = load_expert(extract_dir)
        self.k = k
        self.H = H  # horizon length
        self.action_idx = 0  # 0 -> 50
        self.traj_id = None

        self.device = device
        self.num_image_cat = 8
        self.num_audio_cat = 32
        self.image_window = deque([])
        self.audio_window = deque([])
        self.all_audio = deque([])
        self.start_position = np.array(
            [-0.3842, -0.5922, 0.0449, -2.2487, 0.1340, 0.2287, 1.1580]
        )
        self.update_count = 0
        self.traj_count = 0

    def update_window(self, sample):
        print(f"update count: {self.update_count}")
        self.update_count += 1

        img = sample["cam0c"]
        self.image_window.append(img)

        while len(self.image_window) < self.num_image_cat + 1:
            self.image_window.append(img)

        if len(self.image_window) > self.num_image_cat:
            self.image_window.popleft()


        self.audio_window = deque([])
        print(f'num audio: {len(sample["audio"][2:])}')

        for i in range(len(sample["audio"])-1000, len(sample["audio"])):
            audio_tuple, _ = sample["audio"][i]
            audio_arr = np.array(list(audio_tuple), dtype=np.float64).T
            self.audio_window.append(audio_arr)

        if DEBUG:
            if self.update_count % self.num_audio_cat == 0:
                import matplotlib.pyplot as plt

                plt.clf()
                audio_input = np.concatenate(list(self.audio_window), axis=1)
                audio_input = np.mean(audio_input, axis=0)

                assert audio_input.ndim == 1

                plt.plot(list(range(audio_input.shape[0])), audio_input)
                plt.savefig(f"./outputs/avid-ft-2sec/{self.update_count}-audio-v4.png")

        return True

    def get_embedding(self, sample):
        img_input = list(self.image_window)
        audio_input = np.concatenate(list(self.audio_window), axis=1)
        print(f"audio input shape: {audio_input.shape}")

        if self.num_image_cat > 1:
            data = {"video": img_input, "image": None, "audio": audio_input}
        else:
            data = {"video": None, "image": img_input[0], "audio": audio_input}
        data_t = self.transforms(data, inference=True)

        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(data_t)
        embedding = emb.detach()
        return embedding.cpu()

    def predict(self, sample):
        window_full = self.update_window(sample)

        # ensure input is a video for AVID model
        if not window_full:
            print(f"returning to starting position")
            return self.start_position

        if self.H == 1:
            embedding = self.get_embedding(sample)

            knn_dis, knn_idx = self.KDTree.query(embedding, k=self.k)

            if self.k == 1:
                traj_id, traj_sub_idx = self.traj_ids[knn_idx[0][0]]
                return_action = self.actions[traj_id][traj_sub_idx]
                print(f"knn_idx: {knn_idx[0][0]}")

            else:  # weighting top k by distances
                print(self.k)
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
            if self.traj_id is None or self.action_idx >= self.H:
                embedding = self.get_embedding(sample)
                print(f"embedding shape: {embedding.shape}")
                knn_dis, knn_idx = self.KDTree.query(embedding, k=self.k)
                self.traj_id, self.start_action_idx = self.traj_ids[knn_idx[0][0]]
                self.action_idx = 0
                print(f"Beginning trajectory: {self.traj_id}, {self.start_action_idx}")
                print(f"knn dis: {knn_dis}")
                print(f"traj_count: {self.traj_count}")
                self.traj_count += 1

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
                    # print(f"Finished KNN trajectory")
                    self.traj_id = None

        return return_action


def load_expert(extract_dir):
    knn_model_path = os.path.join(extract_dir, "knn_model.pkl")
    actions_path = os.path.join(extract_dir, "action_trajs.pkl")
    traj_ids_path = os.path.join(extract_dir, "traj_ids.pkl")
    img_paths_path = os.path.join(extract_dir, "image_paths.pkl")
    knn_model = joblib.load(knn_model_path)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)
    with open(traj_ids_path, "rb") as f:
        traj_ids = pickle.load(f)
    with open(img_paths_path, "rb") as f:
        image_paths = pickle.load(f)
    return knn_model, actions, traj_ids, image_paths


def identity_transform(img):
    return img


def _init_agent_from_config(config, device="cpu"):
    print("Loading KNNAudioImage................")
    transforms = identity_transform
    knn_agent = KNNAudioImage(
        k=config.knn.k,
        backbone_cfg=config.agent.backbone_cfg,
        extract_dir=config.data.extract_dir,
        encoder_name=config.agent.encoder_name,
        H=config.knn.H,
    )
    return knn_agent, transforms


def load_and_annotate(
    path,
    vision_model,
    height,
    width,
    text=True,
    horizontal_flip=False,
    grayscale=False,
    invert=False,
    affine=False,
):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import torchvision.transforms.functional as F

    # if fig is None:
    im = Image.open(path)
    if horizontal_flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if grayscale:
        im = F.to_grayscale(im, num_output_channels=3)
    if invert:
        im = F.invert(im)
    if affine:
        im = F.affine(im, angle=0, translate=(200, 0), scale=1.0, shear=0)
    # else:
    #     im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    I1 = ImageDraw.Draw(im)

    font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    big_font = ImageFont.truetype(font_path, 40)
    small_font = ImageFont.truetype(font_path, 30)

    if text:
        I1.text(
            (5, height - 80),
            vision_model,
            stroke_width=1,
            fill=(255, 0, 0),
            font=big_font,
        )
        I1.text(
            (5, height - 35),
            path.split("/")[-2],
            stroke_width=1,
            fill=(255, 0, 0),
            font=small_font,
        )
    return im


def main():
    HYDRA_FN = "/home/vdean/franka_learning_jared/outputs/avid-ft-video-2sec/hydra.yaml"
    WIDTH = 640
    HEIGHT = 480
    K = 5
    HORIZONTAL_FLIP = False
    GRAYSCALE = False
    INVERT = False
    AFFINE = False
    NUM_IMAGES_CAT, NUM_AUDIO_CAT = 8, 32

    import io
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    with open(HYDRA_FN, "rb") as f:
        hydra_cfg = yaml.load(f, Loader=yaml.FullLoader)

    backbone_cfg = hydra_cfg["agent"]["backbone_cfg"]
    encoder_name = hydra_cfg["agent"]["encoder_name"]
    extract_dir = hydra_cfg["data"]["extract_dir"]

    # load knn agent, model, and transforms
    cfg = yaml.safe_load(open(backbone_cfg))
    encoder = load_encoder(encoder_name, cfg)
    transforms = load_transforms(encoder_name, cfg)
    KDTree, actions, traj_ids, image_paths = load_expert(extract_dir)

    # set sample inputs to 'image_paths' indices
    sample_inputs = [800, 3776, 7000, 12000]

    # create final image with size large enough for 5 nearest neighbors and audio and image for 4 samples
    new_im = Image.new("RGB", (WIDTH * (K + 1), HEIGHT * len(sample_inputs) * 2))
    x_offset = WIDTH

    # iterate through sample inputs
    y_offset = 0
    for sample_idx in sample_inputs:
        img_list = image_paths[sample_idx]
        x_offset = 0
        label = "sample image" if y_offset == 0 else ""
        test_img = load_and_annotate(
            img_list[0],
            label,
            HEIGHT,
            WIDTH,
            horizontal_flip=HORIZONTAL_FLIP,
            grayscale=GRAYSCALE,
            invert=INVERT,
            affine=AFFINE,
        )
        new_im.paste(test_img, (x_offset, y_offset))

        # prepare input for model
        sample = {"video": None, "audio": None}
        frames = []
        audio_list = []
        for i, img_path in enumerate(img_list):
            if i >= len(img_list) - NUM_IMAGES_CAT:
                img = Image.open(img_path)
                if HORIZONTAL_FLIP:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if GRAYSCALE:
                    img = F.to_grayscale(img, num_output_channels=3)
                if INVERT:
                    img = F.invert(img)
                if AFFINE:
                    img = F.affine(img, angle=0, translate=(200, 0), scale=1.0, shear=0)

                frames.append(img)

            audio_path = f"{img_path[:-len('.jpeg')]}.txt"
            txt_arr = np.loadtxt(audio_path)
            txt_arr = txt_arr.T
            audio_list.append(txt_arr)

        audio_data = np.concatenate(audio_list, axis=1)
        audio_avg = np.mean(audio_data, axis=0)
        fig = plt.figure()
        plt.plot(list(range(audio_avg.shape[0])), audio_avg)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)

        audio_img = load_and_annotate(buf, "", HEIGHT, WIDTH, text=False)
        new_im.paste(audio_img, (x_offset, y_offset + HEIGHT))
        plt.clf()

        # pass data to model
        sample["video"] = frames
        sample["audio"] = audio_data

        t_samples = transforms(sample, inference=True)

        encoder.eval()
        with torch.no_grad():
            emb = encoder(t_samples).detach().numpy()

        # for each sample input obtain 5 nearest neighbors
        knn_dis, knn_idx = KDTree.query(emb, k=100)
        knn_dists = knn_dis[0].tolist()
        knn_idxs = knn_idx[0].tolist()

        curr_traj_id = img_list[0].split("/")[-2]
        count = 0
        for i, idx in enumerate(knn_idxs):
            if i == 0:
                print(idx)

            if count == K:
                break
            elif traj_ids[idx][0].split("/")[-1] == curr_traj_id:
                continue
            else:
                count += 1

            x_offset += WIDTH

            # for each nearest neighbor get the corresponding image and audio file
            knn_match_fn = image_paths[idx][0]
            label = (
                f"dist: {knn_dists[i]:.2f}, neighbor: {count}"
                if y_offset == 0
                else f"dist: {knn_dists[i]:.2f}"
            )

            knn_match_img = load_and_annotate(knn_match_fn, label, HEIGHT, WIDTH)
            new_im.paste(knn_match_img, (x_offset, y_offset))

            knn_match_audio_list = []
            for knn_img in image_paths[idx]:
                knn_audio_fn = f"{knn_img[:-len('.jpeg')]}.txt"
                txt_arr = np.loadtxt(knn_audio_fn)
                txt_arr = txt_arr.T
                knn_match_audio_list.append(txt_arr)

            knn_match_audio = np.concatenate(knn_match_audio_list, axis=1)
            knn_match_audio_avg = np.mean(knn_match_audio, axis=0)
            fig = plt.figure()
            plt.plot(list(range(knn_match_audio_avg.shape[0])), knn_match_audio_avg)
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)

            knn_match_audio_img = load_and_annotate(buf, "", HEIGHT, WIDTH, text=False)
            new_im.paste(knn_match_audio_img, (x_offset, y_offset + HEIGHT))
            plt.clf()

        y_offset += HEIGHT * 2

    new_im.save("./knn_vis/knn_vis_avid-ft-video-2sec.jpeg")


if __name__ == "__main__":
    main()
