from functools import partial

from .AVID import AVID_MODELS

AUDIO_MODEL_LIST = ["avid-ft-audio", "byol-a", "byol-a-ft", "byol-a-scratch"]

VISUAL_MODEL_LIST = [
    "avid-ft-video",
    "byol",
    "moco-in-domain",
    "moco-generic",
    "resnet50",
    "r3m",
]

MODEL_LIST = [
    "avid-ft",
    "avid-no-ft",
    "avid-scratch",
    "avid-rand",
    "byol-audio-vid-cat",
    "gdt",
]

__all__ = ["load_encoder", "load_transforms"]


def load_encoder(encoder_name, cfg):
    assert (
        encoder_name in MODEL_LIST + AUDIO_MODEL_LIST + VISUAL_MODEL_LIST
    ), f"Model {encoder_name} not implemented"

    # ============================================================
    # AVID
    # ============================================================
    if encoder_name in AVID_MODELS:
        from .AVID import _load_encoder
    # ============================================================
    # BYOL
    # ============================================================
    elif encoder_name == "byol":
        from .BYOL import _load_encoder
    # ============================================================
    # ResNet50
    # ============================================================
    elif encoder_name == "resnet50":
        from .resnet import _load_encoder
    else:
        print("Model not implemented")
        raise NotImplementedError

    return _load_encoder(encoder_name, cfg)


def load_transforms(encoder_name, cfg):
    assert (
        encoder_name in MODEL_LIST + AUDIO_MODEL_LIST + VISUAL_MODEL_LIST
    ), f"Model {encoder_name} transforms not implemented"

    # ============================================================
    # AVID
    # ============================================================
    if encoder_name in AVID_MODELS:
        from .AVID import _load_transforms

    # ============================================================
    # BYOL
    # ============================================================
    elif encoder_name == "byol":
        from .BYOL import _load_transforms
    # ============================================================
    # ResNet50
    # ============================================================
    elif encoder_name == "resnet50":
        from .resnet import _load_transforms
    else:
        print("Model not implemented")
        raise NotImplementedError

    transforms = _load_transforms(encoder_name, cfg)
    transforms_f = partial(transform_func, transforms=transforms)

    return transforms_f


def transform_func(data, transforms, inference=False):
    data_t = {}
    for modality in transforms.keys():
        if transforms[modality] is not None:

            assert (
                data[modality] is not None
            ), f"Expected {modality} data but {modality} data was None"

            data_t[modality] = transforms[modality](data[modality])

            if inference:
                data_t[modality] = data_t[modality].unsqueeze(0)

    return data_t
