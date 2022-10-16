import logging
import pickle, os, yaml
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from torch.utils.data import DataLoader, random_split
from dataset_lazy import FrankaDataset
from agents import init_agent_from_config
from vision import load_model, load_transforms
from tqdm import tqdm
import baselines

log = logging.getLogger(__name__)

def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def get_opt_sched(cfg, policy): # TODO enable scheduler
    if cfg.policy == "TransformerPolicy":
        opt = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda t: min((t+1) / cfg.warmup_steps, 1))
    elif cfg.policy == "Policy":
        opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        scheduler = None
    else:
        raise ValueError(f"Unrecognized policy {cfg.policy}")
    return opt, scheduler

def global_seeding(seed=0):
    import torch
    torch.manual_seed(seed)
    import numpy
    numpy.random.seed(seed)

@hydra.main(config_path="conf", config_name="train_bc")
def main(cfg : DictConfig) -> None:
    with open_dict(cfg):
        cfg['saved_folder'] = os.getcwd()
        print("Model saved dir: ", cfg['saved_folder'])

        if cfg['data']['relabel']['window'] == 0 or cfg['data']['relabel']['src'] == "":
            cfg['data']['relabel'] = None
    if cfg.agent.type == 'bcimage':
        if cfg['agent']['vision_model'] == 'moco_conv3': # TODO: store these in vision/?
            cfg['data']['images']['per_img_out'] = 2156
        if cfg['agent']['vision_model'] == 'moco_conv5':
            cfg['data']['images']['per_img_out'] = 2048
        if cfg['agent']['vision_model'] == 'r3m':
            cfg['data']['images']['per_img_out'] = 2048
        if cfg['agent']['vision_model'] == 'clip':
            cfg['data']['images']['per_img_out'] = 512
        if cfg['agent']['vision_model'] == 'byol':
            cfg['data']['images']['per_img_out'] = 512
        if cfg['agent']['vision_model'] == 'rb2':
            cfg['data']['images']['per_img_out'] = 256

    print(OmegaConf.to_yaml(cfg, resolve=True))

    with open(os.path.join(os.getcwd(), 'hydra.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    global_seeding(cfg.training.seed)

    with open(os.path.join(hydra.utils.get_original_cwd(), cfg.data.pickle_fn), 'rb') as f:
        data = pickle.load(f)

    dset = FrankaDataset(data,
        logs_folder=cfg.data.logs_folder,
        subsample_period=cfg.data.subsample_period,
        im_h=cfg.data.images.im_h,
        im_w=cfg.data.images.im_w,
        obs_dim=cfg.data.in_dim,
        action_dim=cfg.data.out_dim,
        relabel=cfg.data.relabel,
        device=cfg.training.device,
        cameras=cfg.data.images.cameras,
        img_transform_fn=load_transforms(cfg) if cfg.agent.type != 'bc' else None,
        tracking_info=cfg.data.tracking,
        debug=cfg.debug,
        noise=cfg.data.noise)

    split_sizes = [int(len(dset) * 0.8), len(dset) - int(len(dset) * 0.8)]
    train_set, test_set = random_split(dset, split_sizes)

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size)
    agent, _ = init_agent_from_config(cfg, cfg.training.device, normalization=dset)
    train_metric, test_metric = baselines.Metric(), baselines.Metric()
    print(f"number of epochs: {cfg.training.epochs}, dataloader len: {len(train_loader.dataset)}")
    for epoch in range(cfg.training.epochs):
        acc_loss = 0.
        train_metric.reset(); test_metric.reset()
        batch = 0
        # import pdb; pdb.set_trace()        
        for data in train_loader:
            for key in data:
                data[key] = data[key].to(cfg.training.device)
            agent.train(data)
            acc_loss += agent.loss
            train_metric.add(agent.loss.item())
            print('epoch {} \t batch {} \t train {:.6f} \t\t'.format(epoch, batch, agent.loss.item()), end='\r')
            batch += 1

        test_loss = 0.
        for data in test_loader:
            for key in data:
                data[key] = data[key].to(cfg.training.device)
            test_metric.add(agent.eval(data))
        print('epoch {} \t train {:.6f} \t test {:.6f}'.format(epoch, train_metric.mean, test_metric.mean))

        log.info(f'{acc_loss}')
        if epoch % cfg.training.save_every_x_epoch == 0:
            agent.save(os.getcwd())

    agent.save(os.getcwd())
    print("Saved agent to",os.getcwd())

if __name__ == '__main__':
    main()
