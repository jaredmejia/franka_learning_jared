import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchaudio

AVID_SL_DIR = "/home/vdean/jared_contact_mic/avid-glove"

sys.path.insert(1, AVID_SL_DIR)
from main import LightningModel
from datasets import preprocessing


CONTACT_AUDIO_FREQ = 32000
INPUT_AUDIO_MULTIPLIER = 2


class MultiModalAVIDEncoder(nn.Module):
    def __init__(self, avid_model):
        super().__init__()
        self.avid_model = avid_model

    def forward(self, data):
        assert data["video"] is not None, "Expected video data, but video data is None"
        assert data["audio"] is not None, "Expected audio data, but audio data is None"

        bs = data["video"].shape[0]

        video_emb = self.avid_model.visual_model(data["video"]).reshape(bs, -1)
        audio_emb = self.avid_model.audio_model(data["audio"]).reshape(bs, -1)
        cat_emb = torch.cat((video_emb, audio_emb), 1)

        return cat_emb


def _load_encoder(model_name, cfg):
    model = _load_model(model_name, cfg)

    if model_name in ["avid-ft-audio", "avid-ft-video"]:
        model.unimodal_classifier = nn.Identity()
        encoder = model
    else:
        model.classifier = nn.Identity()
        encoder = MultiModalAVIDEncoder(model)

    return encoder


def _load_model(model_name, cfg):
    if model_name == "avid-rand":
        model = LightningModel(cfg)
        model_path = os.path.join(AVID_SL_DIR, "models/avid/ckpt/AVID_random.pt")
        model.model.load_state_dict(torch.load(model_path))

    elif model_name == "avid-scratch":
        # TODO: train avid model from scratch on Glove
        print("Model not implemented")
        raise NotImplementedError

    elif model_name == "avid-no-ft":
        model = LightningModel(cfg)
        ckp_path = os.path.join(AVID_SL_DIR, cfg.model.checkpoint)
        ckp = torch.load(ckp_path, map_location="cpu")["model"]

        visual_model_ckp = {
            k[len("module.video_model.") :]: ckp[k]
            for k in ckp
            if k.startswith("module.video_model.")
        }
        model.model.visual_model.load_state_dict(visual_model_ckp)

        audio_model_ckp = {
            k[len("module.audio_model.") :]: ckp[k]
            for k in ckp
            if k.startswith("module.audio_model.")
        }
        model.model.audio_model.load_state_dict(audio_model_ckp)

        del visual_model_ckp, audio_model_ckp, ckp

    # finetuned
    else:
        ckp_path = os.path.join(cfg.save_path, cfg.name, "checkpoints/last.ckpt")
        model = LightningModel.load_from_checkpoint(ckp_path)

    return model.model


def _load_transforms(model_name, cfg):
    transforms = {"video": None, "audio": None}

    video_transform = preprocessing.__dict__[cfg.dataset.video_transform](
        **cfg.dataset.video_transform_args.__dict__, augment=False
    )

    audio_transform = preprocessing.__dict__[cfg.dataset.audio_transform](
        **cfg.dataset.audio_transform_args.__dict__, augment=False
    )

    spec_transform = preprocessing.__dict__[cfg.dataset.spec_transform](
        **cfg.dataset.spec_transform_args.__dict__, augment=False
    )

    audio_resampler = torchaudio.transforms.Resample(
        orig_freq=CONTACT_AUDIO_FREQ,
        new_freq=cfg.dataset.audio_rate * INPUT_AUDIO_MULTIPLIER,
        dtype=torch.float64,
    )

    def audio_transforms(audio_arr):
        audio = torch.tensor(audio_arr)
        audio = audio_resampler(audio)
        audio = audio.numpy().astype(np.int16)
        audio = audio / np.iinfo(audio.dtype).max
        audio, audio_rate = audio_transform(
            audio, cfg.dataset.audio_rate * INPUT_AUDIO_MULTIPLIER
        )
        stft, stft_rate = spec_transform(audio, audio_rate)
        return stft

    def video_transforms(img_list):
        video = video_transform(img_list)
        return video

    # single image AVID version (use video version instead)
    def img_transforms(img):
        video = video_transform([img])
        return video

    if model_name == "avid-ft-audio":
        transforms["audio"] = audio_transforms
    elif model_name == "avid-ft-video":
        transforms["video"] = video_transforms
    else:
        transforms["audio"] = audio_transforms
        transforms["video"] = video_transforms

    return transforms
