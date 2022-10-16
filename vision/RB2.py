import baselines
import torch
from torchvision import transforms

def _load_model(config):
    img_encoder  = baselines.get_network('VGGSoftmax')
    img_encoder.load_state_dict(torch.load(config.agent.vision_model_path))
    return img_encoder

def _load_transforms(config):
    return transforms.Compose([
        transforms.RandomResizedCrop((config.data.images.im_h, config.data.images.im_w), (0.8, 1.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225], inplace=True)
    ])
