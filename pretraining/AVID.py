import os
import sys
import torch
import torch.nn as nn

AVID_DIR = "/home/vdean/jared_contact_mic/avid-glove"

sys.path.insert(1, AVID_DIR)
from main import LightningModel
from datasets import preprocessing


AVID_MODELS = [
    "avid-ft-audio",
    "avid-ft-video",
    "avid-ft",
    "avid-no-ft",
    "avid-scratch",
    "avid-rand",
]


def _load_encoder(model_name, cfg):
    model = _load_model(cfg, model_name)

    if model_name in ["avid-ft-audio", "avid-ft-video"]:
        model.unimodal_classifier = nn.Identity()
    else:
        model.classifier = nn.Identity()

    return model


def _load_model(model_name, cfg):
    if model_name == "avid-rand":
        model = LightningModel(cfg)
        model_path = os.path.join(AVID_DIR, "models/avid/ckpt/AVID_random.pt")
        model.model.load_state_dict(torch.load(model_path))

    elif model_name == "avid-scratch":
        # TODO: train avid model from scratch on Glove
        print("Model not implemented")
        raise NotImplementedError

    elif model_name == "avid-no-ft":
        model = LightningModel(cfg)
        ckp_path = os.path.join(AVID_DIR, cfg.model.checkpoint)
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

    return model


def _load_transforms(model_name, cfg):
    video_transform = preprocessing.__dict__[cfg.video_transform](
        **cfg.video_transform_args.__dict__, augment=False
    )
    audio_transform = preprocessing.__dict__[cfg.audio_transform](
        **cfg.audio_transform_args.__dict__, augment=False
    )

    if model_name == "avid-ft-audio":
        transforms = {"video": None, "audio": audio_transform}
    elif model_name == "avid-ft-video":
        transforms = {"video": video_transform, "audio": None}
    else:
        transforms = {"video": video_transform, "audio": audio_transform}

    return transforms
