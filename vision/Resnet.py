import os
import torch
from torch import nn
import torchvision.models as models

class PretrainedResNet(nn.Module):
    def __init__(self, im_h, im_w, out_size, fix_resnet=True):
        super(PretrainedResNet, self).__init__()
        self.net = models.resnet34(pretrained=True)
        self.num_ftrs = self.net.fc.in_features
        self.net.fc = nn.modules.linear.Identity()
        self.out_size = out_size
        for parameters in self.net.parameters():
            parameters.requires_grad = not fix_resnet # Optional fix resnet layers
        self.fc=nn.Linear(self.num_ftrs, self.out_size)

    def forward(self, x):
        x=self.net(x)
        return self.fc(torch.flatten(x, start_dim=1)) # TODO(vdean): do this to rb2 features?

    def load_custom_pretrained(self, vision_model_path, device):
        checkpoint = torch.load(vision_model_path)
        self.net.load_state_dict(checkpoint['convnet'])
        self.net = self.net.to(device)


def _load_model(config):
    device = 'cuda:0'
    img_encoder = PretrainedResNet( 
            config.data.images.im_h, config.data.images.im_w,
            config.data.images.per_img_out, config.agent.fix_resnet)

    if config.agent.vision_model_path:
        img_encoder.load_custom_pretrained(
            config.agent.vision_model_path,
            device)
    return img_encoder


def _load_transforms(config):
    from torchvision import transforms
    return transforms.Compose([
            transforms.Resize((config.data.images.im_h, config.data.images.im_w)),
            transforms.ToTensor(),
            # transforms.Normalize([0,0,0], [255,255,255], inplace=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
            ])

# class PretrainedResNet18(nn.Module):
#     def __init__(self, im_h, im_w, out_size, fix_resnet=True):
#         super(PretrainedResNet18, self).__init__()
#         self.net = models.resnet18(pretrained=False)
#         self.num_ftrs = self.net.fc.in_features
#         self.out_size = out_size
#         for parameters in self.net.parameters():
#             parameters.requires_grad = not fix_resnet # Optional fix resnet layers
#         self.fc=nn.Linear(self.num_ftrs, self.out_size)

#     def forward(self, x):
#         x=self.net(x)
#         return x
#         # return self.fc(torch.flatten(x, start_dim=1)) # TODO(vdean): do this to rb2 features?

#     def load_custom_pretrained(self, vision_model_path, device):
#         checkpoint = torch.load(vision_model_path)
#         self.net.load_state_dict(checkpoint['model_state_dict'])
#         self.net.fc = Identity()
#         self.net = self.net.to(device)