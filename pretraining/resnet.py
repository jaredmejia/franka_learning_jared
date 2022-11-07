import torch.nn as nn

from torchvision import models
from torchvision import transforms as T


def _load_encoder(model_name, cfg):
    model = _load_model(model_name, cfg)
    encoder = model.fc = nn.Identity()
    return encoder


def _load_model(model_name, cfg):
    model = models.resnet50(pretrained=True)
    return model


def _load_transforms(model_name, cfg):
    transforms = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms
