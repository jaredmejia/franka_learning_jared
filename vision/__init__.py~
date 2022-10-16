from models.model_loading import MODEL_LIST 


def load_model(config):
    model_type = config.agent.vision_model
    if model_type in MODEL_LIST:
        from .aravind import _load_model
    if model_type == 'byol':
        from .BYOL import _load_model
    if model_type == 'resnet':
        from .Resnet import _load_model
    if model_type == 'rb2':
        from .RB2 import _load_model
    return _load_model(config)


def load_transforms(config):
    model_type = config.agent.vision_model
    if model_type in MODEL_LIST:
        from .aravind import _load_transforms
    if model_type == 'byol':
        from .BYOL import _load_transforms
    if model_type == 'resnet':
        from .Resnet import _load_transforms
    if model_type == 'rb2':
        from .RB2 import _load_transforms
    return _load_transforms(config)
