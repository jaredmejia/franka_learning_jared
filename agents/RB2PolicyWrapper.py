import numpy as np
import pickle
import torch
import os
import yaml
from utils2 import Namespace
import baselines
from torchvision.utils import save_image

class RB2PolicyWrapper():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    # For 1-batch query only!
    def predict(self, sample):
        # apply policy
        img = torch.unsqueeze(sample['cam0c'], dim=0).to(self.device)
        save_image(torch.Tensor(sample['cam0c']),'outputs/rb2/img.png')
        import pdb;pdb.set_trace()
        state = torch.unsqueeze(torch.Tensor(sample['inputs']), dim=0).to(self.device)
        at = self.policy.forward(img, state)
        return at.to('cpu').detach().numpy()[0] #(1, 7) to (7,)

def _init_agent_from_config(config, device):
    features = baselines.get_network(config.agent.features)
    features.load_state_dict(torch.load(config.agent.vision_model))

    policy = baselines.net.CNNPolicy(features, H=1).to(device)
    policy.load_state_dict(torch.load(config.agent.policy_model))
    policy = policy.eval()
    return RB2PolicyWrapper(policy, device)


if __name__ == "__main__":
    cfg = yaml.load(open('/home/gaoyue/dev/franka_learning/conf/visuomotor_agent.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = Namespace(cfg)
    wrapper, transforms = _init_agent_from_config(cfg, None)
    import pdb; pdb.set_trace()
