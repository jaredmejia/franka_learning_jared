import os
import sys
import torch

from torchvision import models
from torchvision import transforms as T

BYOL_DIR = "/home/vdean/jared_contact_mic/byol-pytorch/examples/lightning"

sys.path.insert(1, BYOL_DIR)
from train import SelfSupervisedLearner


def _load_encoder(model_name, cfg):
    model = _load_model(model_name, cfg)

    if model_name == "byol":
        encoder = model.online_encoder
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

    else:
        print("Model not implemented")
        raise NotImplementedError

    return model


def _load_transforms(model_name, cfg):
    if model_name == "byol":
        transforms = T.Compose(
            [
                T.Resize(cfg.model.args.image_size),
                T.CenterCrop(cfg.model.args.image_size),
                T.ToTensor(),
                T.Lambda(expand_greyscale),
            ]
        )
    else:
        print("Model not implemented")
        raise NotImplementedError

    return transforms


def expand_greyscale(t):
    return t.expand(3, -1, -1)
