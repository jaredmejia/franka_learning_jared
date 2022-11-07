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
    "xdc",
]


def load_encoder(encoder_name, cfg):
    assert encoder_name in MODEL_LIST + AUDIO_MODEL_LIST + VISUAL_MODEL_LIST

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
    assert encoder_name in MODEL_LIST + AUDIO_MODEL_LIST + VISUAL_MODEL_LIST

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

    return _load_transforms(encoder_name, cfg)
