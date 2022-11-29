import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchaudio

AVID_SSL_DIR = "/home/vdean/jared_contact_mic/AVID-CMA"

sys.path.insert(1, AVID_SSL_DIR)
from datasets import preprocessing
import models


CONTACT_AUDIO_FREQ = 32000


class AVIDEncoderWrapper(nn.Module):
    def __init__(self, avid_model):
        super().__init__()
        self.avid_model = avid_model

    def forward(self, data):
        assert data["video"] is not None, "Expected video data, but video data is None"
        assert data["audio"] is not None, "Expected audio data, but audio data is None"

        bs = data["video"].shape[0]
        video_emb, audio_emb = self.avid_model.forward(data["video"], data["audio"])

        video_emb = video_emb.reshape(bs, -1)
        audio_emb = audio_emb.reshape(bs, -1)
        cat_emb = torch.cat((video_emb, audio_emb), 1)

        return cat_emb


def _load_encoder(model_name, cfg):
    if model_name == "avid-ft-d":
        model = _load_model(model_name, cfg)
        encoder = AVIDEncoderWrapper(model)
    else:
        raise NotImplementedError

    return encoder


def _load_model(model_name, cfg):
    if model_name == "avid-ft-d":
        model_cfg = cfg["model"]
        model = models.__dict__[model_cfg["arch"]](**model_cfg["args"])

        ckp_path = os.path.join(
            AVID_SSL_DIR,
            model_cfg["model_dir"],
            model_cfg["name"],
            "checkpoint-best.pth.tar",
        )

        print(f"\nLoading weights from model: {ckp_path}")
        ckp = torch.load(ckp_path, map_location="cpu")
        module_ckp = {k[len("module.") :]: ckp["model"][k] for k in ckp["model"]}
        model.load_state_dict(module_ckp)

        del ckp, module_ckp

    else:
        raise NotImplementedError

    return model


def _load_transforms(model_name, cfg):
    transforms = {"video": None, "audio": None}

    db_cfg = cfg["dataset"]

    num_frames = int(db_cfg["video_clip_duration"] * db_cfg["video_fps"])
    video_transform = preprocessing.VideoPrep_MSC_CJ(
        crop=(db_cfg["crop_size"], db_cfg["crop_size"]),
        augment=False,
        num_frames=num_frames,
        pad_missing=True,
    )

    audio_transform = preprocessing.AudioPrep(
        trim_pad=True,
        duration=db_cfg["audio_clip_duration"],
        augment=False,
        missing_as_zero=True,
    )
    spec_transform = preprocessing.LogSpectrogram(
        db_cfg["audio_fps"],
        n_fft=db_cfg["n_fft"],
        hop_size=1.0 / db_cfg["spectrogram_fps"],
        normalize=True,
    )

    audio_resampler = torchaudio.transforms.Resample(
        orig_freq=CONTACT_AUDIO_FREQ, new_freq=db_cfg["audio_fps"], dtype=torch.float64
    )

    def video_transforms(img_list):
        video = video_transform(img_list)
        return video

    def audio_transforms(audio_arr):
        audio = torch.tensor(audio_arr)
        audio = audio_resampler(audio)
        audio = audio.numpy().astype(np.int16)
        audio = audio / np.iinfo(audio.dtype).max

        # during training used video_clip_duration size audio
        audio, audio_rate = audio_transform(
            audio, db_cfg["audio_fps"], db_cfg["video_clip_duration"]
        )
        stft, stft_rate = spec_transform(
            audio, audio_rate, db_cfg["video_clip_duration"]
        )

        return stft

    transforms["video"] = video_transforms
    transforms["audio"] = audio_transforms

    return transforms
