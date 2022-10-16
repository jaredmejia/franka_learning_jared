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
        return self.fc(torch.flatten(x, start_dim=1))

    def load_custom_pretrained(self, custom_resnet_path, device):
        checkpoint = torch.load(custom_resnet_path)
        self.net.load_state_dict(checkpoint['convnet'])
        self.net = self.net.to(device)


if __name__ == "__main__":
    resnet = PretrainedResNet()