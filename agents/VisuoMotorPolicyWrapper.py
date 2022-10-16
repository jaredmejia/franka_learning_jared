import numpy as np
import pickle
import torch
import os
import yaml
from utils2 import Namespace

from models.model_loading import load_pretrained_policy

class VisuoMotorPolicyWrapper():
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        self.history = {'images':[], 'joints':[]}

    # For 1-batch query only!
    def predict(self, sample):
        if len(self.history['joints']) == 0:
            for i in range(self.policy.history_window - 1):
                self.history['joints'].append(torch.unsqueeze(torch.Tensor(sample['inputs']), dim=0))
                self.history['images'].append(torch.unsqueeze(torch.Tensor(sample['cam0c']), dim=0))
            # channel, im_h, im_w = sample['cam0c'].shape
            # self.history['images'] = [torch.zeros([1, channel, im_h, im_w])] * (self.policy.history_window - 1)
            # self.history['joints'] = [torch.zeros(1, self.config.data.out_dim)] * (self.policy.history_window - 1)

        self.history['joints'].append(torch.unsqueeze(torch.Tensor(sample['inputs']), dim=0))
        self.history['images'].append(torch.unsqueeze(torch.Tensor(sample['cam0c']), dim=0))
        cur_images = torch.unsqueeze(torch.cat(self.history['images'][-self.policy.history_window:]), dim=0)
        cur_joints = torch.unsqueeze(torch.cat(self.history['joints'][-self.policy.history_window:]), dim=0)
        inp_dict = {'images': cur_images, 'joints': cur_joints}
        at = self.policy.forward(inp_dict)
        # if False: # NO RANDOM
            # at = at + torch.randn(at.shape).to(policy.device) * torch.exp(policy.log_std)
        return at.to('cpu').detach().numpy()[0] #(1, 7) to (7,)

def _init_agent_from_config(config, device):
    vision_model = config.agent.vision_model
    policy_model = config.agent.policy_model
    base_path = config.agent.policy_base_path
    print(vision_model)
    policy, transforms = load_pretrained_policy(vision_model, \
        os.path.join(base_path, policy_model))
    policy.to(device)
    return VisuoMotorPolicyWrapper(policy, config), transforms


if __name__ == "__main__":
    cfg = yaml.load(open('/home/gaoyue/dev/franka_learning/conf/visuomotor_agent.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = Namespace(cfg)
    wrapper, transforms = _init_agent_from_config(cfg, None)
    import pdb; pdb.set_trace()