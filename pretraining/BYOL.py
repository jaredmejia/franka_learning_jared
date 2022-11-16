import numpy as np
import os
import sys
import torch
import torch.nn as nn

from torchaudio import transforms as AT
from torchvision import models
from torchvision import transforms as VT


BYOL_DIR = "/home/vdean/jared_contact_mic/byol-pytorch/examples/lightning"

sys.path.insert(1, BYOL_DIR)
from train import SelfSupervisedLearner

BYOL_A_DIR = "/home/vdean/jared_contact_mic/byol-a"

sys.path.insert(1, BYOL_A_DIR)
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020

DEVICE = torch.device("cuda")


class MultiModalBYOL(nn.Module):
    def __init__(self, image_encoder, audio_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder

    def forward(self, data):
        assert data["image"] is not None, "Expected image data, but video data is None"
        assert data["audio"] is not None, "Expected audio data, but audio data is None"

        bs = data["image"].shape[0]

        img_emb = self.image_encoder(data["image"]).reshape(bs, -1)
        audio_emb = self.audio_encoder(data["audio"]).reshape(bs, -1)
        cat_emb = torch.cat((img_emb, audio_emb), 1)

        return cat_emb


class BYOLWrapper(nn.Module):
    def __init__(self, byol_model, modality="image"):
        super().__init__()
        self.byol_model = byol_model
        self.modality = modality

    def forward(self, data):
        assert data[self.modality] is not None, f"Expected {self.modality} data, but video data is None"

        return self.byol_model(data[self.modality])


def _load_encoder(model_name, cfg):
    # TODO: make sure every case returns an encoder func that handles the input data dict properly
    # TODO: give option to put model on specified device
    # TODO: give option to put data_t on specified device
    model = _load_model(model_name, cfg)

    if model_name == "byol":
        encoder = BYOLWrapper(model.online_encoder, modality="image")
    elif model_name == "byol-a":
        encoder = BYOLWrapper(model, modality="audio")
    elif model_name == "byol-img-audio":
        encoder = MultiModalBYOL(model["image"].online_encoder, model["audio"])
    else:
        print("Model not implemented")
        raise NotImplementedError

    return encoder


def _load_model(model_name, cfg):
    if model_name == "byol":
        resnet = models.resnet50(pretrained=True)
        model = SelfSupervisedLearner(resnet, **cfg.model.args.__dict__)

        ckp = torch.load(cfg.model.checkpoint)
        net_ckp = {
            k[len("learner.") :]: ckp["state_dict"][k] for k in ckp["state_dict"]
        }
        model.learner.load_state_dict(net_ckp)

    if model_name == "byol-a":
        model = AudioNTT2020(d=cfg.feature_d)
        pretrained_weights_path = os.path.join(
            BYOL_A_DIR, "pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth"
        )
        model.load_weight(pretrained_weights_path, DEVICE)

    else:
        print("Model not implemented")
        raise NotImplementedError

    return model


def _load_transforms(model_name, cfg):
    transforms = {"image": None, "audio": None}

    if model_name == "byol":
        image_transform = VT.Compose(
            [
                VT.Resize(cfg.model.args.image_size),
                VT.CenterCrop(cfg.model.args.image_size),
                VT.ToTensor(),
                VT.Lambda(expand_greyscale),
            ]
        )
        transforms["image"] = image_transform

    elif model_name == "byol-a":
        stats = [9.331433, 1.9019704]
        to_melspec = AT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        normalizer = PrecomputedNorm(stats)

        def audio_transform(wav):
            lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())
            return lms

        transforms["audio"] = audio_transform

    else:
        print("Model not implemented")
        raise NotImplementedError

    def transform_func(data):
        data_t = {}
        if transforms["image"] is not None:
            assert (
                data["image"] is not None
            ), "Expected image data but image data was None"

            data_t["image"] = transforms["image"](data["image"])

        if transforms["audio"] is not None:
            assert (
                data["audio"] is not None
            ), "Expected audio data but audio data was None"

            data_t["audio"] = transforms["audio"](data["audio"])

    return transform_func


def expand_greyscale(t):
    return t.expand(3, -1, -1)


def get_sec(time_str):
    """Get Seconds from time."""
    hour, minute, second, second_decimal = time_str.split(".")
    return (
        int(hour) * 3600 + int(minute) * 60 + int(second) + float("0." + second_decimal)
    )


def read_txt(file_path):
    with open(file_path, "r") as filehandle:
        data = []
        for line in filehandle:
            line = line[:-1]
            line = line.split(" ")
            if len(line) > 1:
                data.append(line)
            else:
                break
    time = [get_sec(line[0]) for line in data]
    time = np.asarray(time) - time[0]  # Start time axis at 0s
    data = [line[1:] for line in data]
    data = np.array(data).astype(float)

    return time, data
