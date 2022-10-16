import numpy as np
import pickle

from mjrl.policies.gaussian_mlp import MLP

class MORel():
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        print(f'Obs dim {self.policy.observation_dim} Act dim {self.policy.action_dim}')

    # For 1-batch query only!
    def predict(self, sample):
        at = self.policy.forward(sample['inputs'])
        if False: # NO RANDOM
            at = at + torch.randn(at.shape).to(policy.device) * torch.exp(policy.log_std)
        # clamp states and actions to avoid blowup
        return at.to('cpu').detach().numpy()

def _init_agent_from_config(config, device):

    policy = pickle.load(open(config.agent.policy_pickle, 'rb'))
    policy.set_param_values(policy.get_param_values())
    policy.to(device)

    return MORel(policy, config), None
