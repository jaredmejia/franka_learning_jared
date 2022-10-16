import torch
from torch import nn
import torchvision.models as models

from .BCAgent import Policy, init_weights, get_stats
from .BCImageAgent import BCImageAgent, PretrainedResNet


class BCImageGoalAgent(BCImageAgent):
    def __init__(self, models, learning_rate, device, cameras):
        super(BCImageGoalAgent, self).__init__(models, learning_rate, device, cameras)

    def forward(self, sample):
        imgs = [sample[_c] for _c in self.cameras]
        imgs_out = [self.models['img_encoder'](img) for img in imgs]
        goals_out = self.models['img_encoder'](sample['goals'])
        concat_inputs = torch.cat([sample['inputs'], goals_out] + imgs_out, dim=-1)
        return self.models['decoder'](concat_inputs)


class PolicyWithImageGoal(Policy):
    def __init__(self, in_dim, out_dim):
        super(PolicyWithImageGoal, self).__init__(in_dim, out_dim)

    def set_stats(self, dataset):
        # When set stats, ignore the Resnet goal
        inp_dim, inp_mean, inp_std = get_stats(dataset.inputs)
        out_dim, out_mean, out_std = get_stats(dataset.labels)
        self.inp_mean[:inp_dim].copy_(inp_mean)
        self.inp_std[:inp_dim].copy_(inp_std)
        self.out_mean[:out_dim].copy_(out_mean)
        self.out_std[:out_dim].copy_(out_std)


def _init_agent_from_config(config, device='cpu', normalization=None):
    assert len(config.data.images.cameras) > 0
    img_encoder = PretrainedResNet(
        config.data.images.im_h, config.data.images.im_w,
        config.data.images.per_img_out, config.agent.fix_resnet)

    if config.agent.vision_model_path:
        img_encoder.load_custom_pretrained(
            config.agent.vision_model_path,
            device)

    img_input_size = ( 1 + len(config.data.images.cameras)) * config.data.images.per_img_out

    models = {
        'img_encoder': img_encoder,
        'decoder': PolicyWithImageGoal(
            config.data.in_dim + img_input_size,
            config.data.out_dim
            )
    }

    if normalization is not None:
        models['decoder'].set_stats(normalization)

    for k,m in models.items():
        m.to(device)
        if k=="img_encoder":
            print("*** Resnet image encoder, init weight only on FC layers")
            m.fc.apply(init_weights)
        else:
            m.apply(init_weights)

    agent = BCImageGoalAgent(models, config.training.lr, device,
        config.data.images.cameras)
    return agent