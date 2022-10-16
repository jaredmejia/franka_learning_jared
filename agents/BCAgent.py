import torch
from torch import nn

from .BaseAgent import BaseAgent

class BCAgent(BaseAgent):
    def __init__(self, models, learning_rate, device):
        super(BCAgent, self).__init__(models, learning_rate, device)

    def forward(self, sample):
        inputs = torch.cat((sample['inputs'], sample['goals']), dim=-1)
        return self.models['decoder'](inputs)

    def compute_loss(self, sample):
        output = self.forward(sample)
        labels = sample['labels']
        losses = self.loss_fn(output.view(-1, output.size(-1)), labels)
        self.loss = self.loss_reduction(losses)

    def predict(self, sample):
        [m.eval() for m in self.models.values()]
        with torch.no_grad():
            sample = self.pack_one_batch(sample)
            output = self.forward(sample)[0].to('cpu').detach().numpy()
            return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def get_stats(arr):
    arr_std = arr.std(0)
    arr_std[arr_std < 1e-4] = 1
    return len(arr_std), arr.mean(0), arr_std

class DeepMLPBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, dropout_p=0.1):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.drop = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.drop(self.relu(self.fc(x)))

class Policy(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Policy, self).__init__()
        self.fc1 = DeepMLPBlock(inp_dim, 128)
        self.fc2 = DeepMLPBlock(128, 128)
        self.final = nn.Linear(128, out_dim)
        self.register_buffer("inp_mean", torch.zeros(inp_dim))
        self.register_buffer("inp_std", torch.ones(inp_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std", torch.ones(out_dim))

    def set_stats(self, dataset):
        inp_dim, inp_mean, inp_std = get_stats(dataset.inputs)
        goal_dim, goal_mean, goal_std = get_stats(dataset.goals)
        _, out_mean, out_std = get_stats(dataset.labels)

        self.inp_mean[:inp_dim].copy_(inp_mean)
        self.inp_std[:inp_dim].copy_(inp_std)
        self.inp_mean[inp_dim:inp_dim + goal_dim].copy_(goal_mean)
        self.inp_std[inp_dim:inp_dim + goal_dim].copy_(goal_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def forward(self, observations):
        h = (observations - self.inp_mean) / self.inp_std
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.final(h)
        actions = self.out_mean + self.out_std * h
        return actions

def _init_agent_from_config(config, device='cpu', normalization=None):
    tracking_state_dim = (len(config.data.tracking.marker_ids) * 3
        if config.data.tracking is not None
        else 0)
    input_dim = config.data.in_dim + tracking_state_dim
    if config.data.relabel is not None and config.data.relabel.window > 0:
        if config.data.relabel.src == "jointstate":
            input_dim += config.data.in_dim
        elif config.data.relabel.src == "tracking":
            input_dim += tracking_state_dim

    models = {
        'decoder': Policy(
            input_dim,
            config.data.out_dim)
    }

    if normalization is not None:
        models['decoder'].set_stats(normalization)

    for k,m in models.items():
        m.to(device)
        if k=="img_encoder" and config.model.use_resnet:
            print("*** Resnet image encoder, do not init weight")
        else:
            m.apply(init_weights)

    bc_agent = BCAgent(models, config.training.lr, device)
    return bc_agent, None # image transforms