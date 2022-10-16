import numpy as np
from scipy import special
import pickle
from argparse import Action
import pickle
import numpy as np
import pandas as pd
import os

import torch

from glob import glob
from torch import embedding
from torchvision import transforms as T
# from torchvision import models 
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
#from models.model_loading import MODEL_LIST, load_pvr_model, preprocess_image
#from vision import load_model, load_transforms
from PIL import Image, ImageDraw

import sys

sys.path.insert(1, '/home/vdean/jared_contact_mic/avid-glove')

from utils import misc
from main import LightningModel
from inference import LivePrep, get_feature_extractor
import yaml

class KNNImage(object):
    def __init__(self, k, pickle_fn, vision_model, device='cuda:0'):

        cfg = misc.convert2namespace(yaml.safe_load(open('/home/vdean/jared_contact_mic/avid-glove/config/gloveaudio/avcat-avid-ft-jointloss.yaml')))
        print(f'cfg save paths: {cfg.save_path}')
        model = LightningModel.load_from_checkpoint(os.path.join(cfg.save_path, cfg.name, 'checkpoints/last.ckpt'))
        self.fe_model = get_feature_extractor(model.model, unimodal=False).to(device)

        with open(pickle_fn, 'rb') as f:
            paths = pickle.load(f)
        print(paths[0].keys())
        # paths=paths['demos']
        actions = []
        representations = []
        img_paths = []
        print(f'Loading good trajectories...')
        for path in paths:
            # actions.append(path['commands'])
            actions.append(path['actions'])
            print(f'actions: {len(path["actions"])}')
            data_path = os.path.join('/home/vdean/franka_demo/logs/good_chopping_exps/'+str(path['traj_id']))
            
            for img_file in path['cam0c']:
                img_path = os.path.join(data_path, img_file)
                img_paths.append(img_path)

            # for img_file in path['cam0c'][:1]:
            #     print(f'processing sample {len(img_paths)}')
            #     # load image
            #     img_path = os.path.join(data_path, img_file)
            #     img = Image.open(img_path)

            #     # load audio
            #     txt_path = f'{img_path[:-4]}txt'
            #     txt_arr = np.loadtxt(txt_path)

            #     if txt_arr.shape[0] <= 256:  # minimum num frames for preprocessing
            #         continue
            #     txt_arr = txt_arr.T

            #     # preprocess data
            #     data_prepped = self.img_audio_prep(img, txt_arr)

            #     # extract features
            #     features = self.get_features(data_prepped)
            #     representations.append(features.numpy())
                
            #     img_paths.append(img_path)
            # print(f'Num trajectories loaded: {len(representations)}')

        self.img_audio_prep = LivePrep(cfg.dataset, img_paths, device)
        self.batch_size = 1024
        
        dl = DataLoader(
            dataset=self.img_audio_prep,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=6
        )


        self.representations = self.get_knn_features(dl)
        # self.representations = np.vstack(representations)
        self.actions = np.vstack(actions)
        self.img_paths = img_paths
        print(f'Fitting KNN to {len(self.representations)} representations...')
        self.KDTree = KDTree(data=self.representations)
        print(f'KNN fit')
        self.vision_model = vision_model
        self.k = 3
        self.curr_audio_len = 0
        self.device = device
    
    def get_features(self, input_prepped):
        self.fe_model.eval()
        with torch.no_grad():
            features, _, _ = self.fe_model(input_prepped)
        assert features.shape == torch.Size([1, 1024])
        return features.cpu()

    def get_knn_features(self, dl):
        representations = []
        feature_list = []
        self.fe_model.eval()
        with torch.no_grad():
            for i, data in enumerate(dl):
                features, _, _ = self.fe_model(data)
                feature_list.append(features)
                print(f'loaded {len(feature_list) * self.batch_size} features')
        representations = torch.cat(feature_list, dim=0)
        return representations.cpu()


    def predict(self, sample):
        # Sample contains audio and video. Should be a dict so check keys: maybe 'cam0c'=image, 'audio'=audio
        # Save to MP4, load model, etc

        img = sample['cam0c']
        print(len(sample['audio']))

        audio_list = [np.array(list(audio_tuple)) for audio_tuple, _ in sample['audio'][2:]]
        self.curr_audio_len = len(audio_list) - self.curr_audio_len
        audio_list = audio_list[:max(4, self.curr_audio_len)]  # ensure length is greater than 256=4*64 franes
        audio_arr = np.vstack(audio_list).T

        sample_prepped = self.img_audio_prep(img, audio_arr, predict=True)
        sample_features = self.get_features(sample_prepped)

        knn_dis, knn_idx = self.KDTree.query(sample_features, k=self.k)

        return_action = self.actions[knn_idx[0][0]] ### TODO: works for k=1 only so far!!!
        print(f'action_idx: {knn_idx[0][0]}')
        print(knn_idx)
        print("Closest image:",self.img_paths[knn_idx[0][0]])

        actions = [self.actions[i] for i in knn_idx[0]]
        weights = [-1*(knn_dis[0][i]) for i in range(self.k)]
        weights = special.softmax(weights)
        return_action = np.zeros(7)
        for i in range(self.k):
            return_action += weights[i] * actions[i]

        return return_action
        
def crop_and_resize(img):
    #im1 = img.crop((50, 0, 450, 385))
    # return img.resize((50, 50))
    return img

def _init_agent_from_config(config, device='cpu'):
    print("Loading KNNImage................")
    vision_model =  lambda x: x #load_model(config)
    # transforms = load_transforms(config)
    transforms = crop_and_resize
    knn_agent = KNNImage(config.knn.k, config.data.pickle_fn, vision_model)
    return knn_agent, transforms

def img_test(img_path, knn_agent, transforms, k=1):
    img = Image.open(img_path)
    img_tensor = transforms(img)
    state = knn_agent.vision_model(np.array(img_tensor).flatten())

    # device = 'cuda:0'
    # import yaml, sys, argparse, os, pickle
    # from utils import Namespace
    # with open(os.path.join('/home/gaoyue/dev/franka_learning/outputs/', 'knn_image', 'hydra.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    #     cfg = Namespace(cfg)
    # model = load_model(cfg)
    # model.to(device)

    # for p1, p2 in zip(knn_agent.vision_model.parameters(), model.parameters()):
    #     if p1.data.ne(p2.data).sum() > 0:
    #         print("false")
    # print('true')

    # state = np.hstack([np.array([[-0.15769309, -0.1116268,   0.18283216, -2.43667363,  0.01225936,  0.77048624, -0.60743096]]), state])
    knn_dis, knn_idx = knn_agent.KDTree.query(state, k=10)

    return knn_agent.img_paths[knn_idx[0][0]]
    # import pdb; pdb.set_trace()
    # np.mean(np.array([knn_agent.actions[i] for i in knn_idx[0]]), axis=0)
    # import pdb; pdb.set_trace()
    # return knn_idx


def load_and_annotate(path, vision_model, filename):
    im = Image.open(path)
    I1 = ImageDraw.Draw(im)

    big_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 40)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)

    I1.text((5, 5), vision_model, stroke_width=1, fill=(255, 0, 0), font=big_font)
    I1.text((5, HEIGHT-25), filename, fill=(255, 0, 0), font=small_font)
    return im
    
if __name__ == "__main__":
    # img_path = '/home/gaoyue/Desktop/cloud-dataset-inserting-v0/data/22-05-19-23-13-32/c0-818312070212-1653016412230.5496-color.jpeg'
    test_image_folder = "/home/gaoyue/Desktop/cloud-dataset-scooping-v0-part2/data/"
    test_images = ["22-05-05-17-20-20/c0-818312070212-1651785641073.2153-color.jpeg"]
    test_images = [test_image_folder + fname for fname in test_images]

    WIDTH = 640
    HEIGHT = 480

    import yaml, sys, argparse, os, pickle
    from PIL import Image, ImageDraw, ImageFont
    from utils2 import Namespace
    with open(os.path.join('/home/gaoyue/dev/franka_learning/outputs/', 'knn_image', 'hydra.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['data']['pickle_fn'] = "/home/gaoyue/Desktop/cloud-dataset-scooping-v0/parsed_with_embeddings_byol_robocloud.pkl"
        cfg = Namespace(cfg)

    # Load pickle with pre-computed embeddings
    # Edit agent.vision_model, agent.vision_model_path
    knn_agent, transforms = _init_agent_from_config(cfg)
    dataset_folder = os.path.join(os.path.dirname(cfg.data.pickle_fn), 'data')

    #new_im = Image.new('RGB', (WIDTH * len(images), HEIGHT))
    new_im = Image.new('RGB', (WIDTH * 2, HEIGHT))

    for img_path in test_images:
        test_img = load_and_annotate(img_path, "Test image", img_path)
        new_im.paste(test_img, (0, 0))
        x_offset = WIDTH
    
        closest_img = img_test(img_path, knn_agent, transforms)
        path = os.path.join(dataset_folder,closest_img)
        im = load_and_annotate(path, cfg.agent.vision_model, closest_img)
        
        new_im.paste(im, (x_offset,0))
        x_offset += WIDTH

        new_im.save('test.jpg')
        import pdb; pdb.set_trace()
