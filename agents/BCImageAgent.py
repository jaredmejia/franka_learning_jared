import os
import torch
from torch import nn
import torchvision.models as models
from .BCAgent import BCAgent, Policy, init_weights, get_stats
from models.model_loading import MODEL_LIST, load_pvr_model 
from vision import load_model, load_transforms


class BCImageAgent(BCAgent):
    def __init__(self, models, learning_rate, device, cameras):
        super(BCImageAgent, self).__init__(models, learning_rate, device)
        self.cameras = cameras

    def forward(self, sample):
        imgs = [sample[_c] for _c in self.cameras]
        imgs_out = [self.models['img_encoder'](img) for img in imgs]
        concat_inputs = torch.cat([sample['inputs'], sample['goals']] + imgs_out, dim=-1)
        return self.models['decoder'](concat_inputs)

    def save(self, foldername, filename='Agent.pth'):
        state = {'epoch': self.epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        for mname, m in self.models.items():
            if mname != 'img_encoder': # Don't save img encoder net
                state[mname] = m.state_dict()
        torch.save(state, os.path.join(foldername, filename))


def _init_agent_from_config(config, device='cpu', normalization=None):
    assert len(config.data.images.cameras) > 0
    img_encoder = load_model(config)
    transforms = load_transforms(config)
    img_input_size = len(config.data.images.cameras) * config.data.images.per_img_out
    input_dim = config.data.in_dim + \
        (0 if (config.data.relabel is None) else config.data.in_dim)

    models = {
        'img_encoder': img_encoder,
        'decoder': Policy(
            input_dim + img_input_size,
            config.data.out_dim
            )
    }

    if normalization is not None:
        models['decoder'].set_stats(normalization)

    for k,m in models.items():
        m.to(device)
        if k=="img_encoder":
            print("*** Resnet image encoder, init weight only on FC layers")
            if config.agent.vision_model == 'resnet' \
                and config.agent.vision_model_path \
                and 'pt' in config.agent.vision_model_path:
                m.fc.apply(init_weights)
        else:
            m.apply(init_weights)

    agent = BCImageAgent(models, config.training.lr, device,
        config.data.images.cameras)
    
    return agent, transforms
