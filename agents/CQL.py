import numpy as np
import pickle
import torch

class CQL():
    def __init__(self, policy, config, device):
        self.policy = policy
        self.config = config
        self.device = device
        print(f'Obs dim {self.policy.observation_dim} Act dim {self.policy.action_dim}')

    # For 1-batch query only!
    def predict(self, sample, deterministic=True):
        print("cql predict")
        with torch.no_grad():
            observations = torch.tensor(
                sample['inputs'], dtype=torch.float32, device=self.device
            )
            #print(f'inputs {inp} obs {observations} obs dim ')
            actions, _ = self.policy(observations.unsqueeze(0), deterministic)
            print(f"action: {actions[0].to('cpu').detach().numpy()}")
            return actions[0].to('cpu').detach().numpy()

def _init_agent_from_config(config, device):
    saved_model = pickle.load(open(config.agent.policy_pickle, 'rb'))
    policy = saved_model['sac'].policy # CQL.model.TanhGaussianPolicy
    policy.to(device)

    return CQL(policy, config, device), None
