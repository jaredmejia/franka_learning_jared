from errno import EEXIST
import os
import numpy as np
import argparse
from models.model_loading import MODEL_LIST, load_pvr_model, preprocess_image
from models.model_training import precompute_embeddings
import pickle
import torch
from tqdm import tqdm
from tqdm import tqdm
from PIL import Image
from vision import load_model, load_transforms
import yaml, sys, argparse, os, pickle
from utils2 import Namespace


def precompute_embeddings(cfg, paths, data_path):
    print("here 1")
    # data parallel mode to use both GPUs
    device = 'cpu'
    model = load_model(cfg)
    model.to(device)
    transforms = load_transforms(cfg)
    model = model.to(device)
    batch_size = 128
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path['observations'].shape[0]):
            img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))  # Assuming RGB for now
            img = preprocess_image(img, transforms)
            path_images.append(img)
        # compute the embedding
        path_len = len(path_images)
        embeddings = []
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b*batch_size:min(batch_size*(b+1), path_len)])
                    chunk_embed = model(chunk.to(device))
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
        path['observations'] = np.hstack([path['observations'], path['embeddings']])
    return paths


def precompute_embeddings_byol(cfg, paths, data_path):
    # data parallel mode to use both GPUs
    rint("here 2")
    device = 'cuda:0'
    model = load_model(cfg)
    model.to(device)
    byol_transforms = load_transforms(cfg)

    # img_path = '/home/gaoyue/Desktop/cloud-dataset-inserting-v0/data/22-05-19-23-13-32/c0-818312070212-1653016412230.5496-color.jpeg'
    # img = Image.open(img_path)
    # img_tensor = byol_transforms(img).reshape(3, 224, 224)
    # state = model(torch.unsqueeze(img_tensor, dim=0)).detach().numpy()
    # import pdb; pdb.set_trace()

    batch_size = 1
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path['observations'].shape[0]):
            img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))  # Assuming RGB for now
            img = byol_transforms(img).reshape(3, 224, 224)
            path_images.append(img)
        # compute the embedding
        path_len = len(path_images)
        embeddings = []
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b*batch_size:min(batch_size*(b+1), path_len)])
                    chunk_embed = model(chunk)
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
        path['observations'] = np.hstack([path['observations'], path['embeddings']])
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='/home/gaoyue/Desktop/cloud-dataset-inserting-v0/')
    parser.add_argument('-e', '--embedding_name', type=str, default='moco_conv5')
    args = parser.parse_args()

    from utils2 import Namespace
    ### from the config,
    with open(os.path.join('/home/gaoyue/dev/franka_learning/outputs/', 'knn_image', 'hydra.yaml'), 'r') as f: ## Default to use the knn_image config
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Namespace(cfg)

    cfg['agent']['vision_model'] = args.embedding_name
    paths = pickle.load(open(os.path.join(args.path, 'parsed_with_depth.pkl'), 'rb'))
    data_path = os.path.join(args.path, 'data')

    if cfg['agent']['vision_model'] == 'byol':
        paths_with_embeddings = precompute_embeddings_byol(cfg, paths, data_path)
    else:
        paths_with_embeddings = precompute_embeddings(cfg, paths, data_path)

    with open(os.path.join(args.path, f'parsed_with_embeddings_{args.embedding_name}_robocloud.pkl'), 'wb') as f:
        pickle.dump(paths_with_embeddings, f)



