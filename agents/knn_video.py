import numpy as np
from scipy import special
import pickle
from argparse import Action
import pickle
import numpy as np

import os
import sys
import torch
from  scipy import special
from torch import embedding
from torchvision import transforms as T
from torchvision import models 
from torch import nn
from sklearn.neighbors import KDTree
#from models.model_loading import MODEL_LIST, load_pvr_model, preprocess_image
#from vision import load_model, load_transforms
from PIL import Image, ImageDraw

class KNNVideo(object):
    def __init__(self, k, pickle_fn, fe_model, device='cuda:0'):
        with open(pickle_fn, 'rb') as f:
            paths = pickle.load(f)
        print(paths[0].keys())
        # paths=paths['demos']
        actions = []
        representations = []
        img_paths = []
        for path in paths:
            # actions.append(path['commands'])
            actions.append(path['actions'])
            reps_l =[]
            for img_add in path['cam0c']:
                print(f'image add: {img_add}')
                address= os.path.join('/home/vdean/franka_demo/logs/good_chopping_exps/'+str(path['traj_id']), img_add)
                tmp = Image.open(address)
                print(f'temp size: {np.array(tmp).size}')  # (680, 480)
                reps_l.append(np.array(crop_and_resize(tmp)).flatten())
                tmp.close()

                
            representations.append(reps_l)   
            img_paths.extend([os.path.join(path['traj_id'], fname) for fname in path['cam0c']])
            #representations.append(path['embeddings']) # not using jointstates

        self.representations = np.vstack(representations)
        self.actions = np.vstack(actions)
        self.img_paths = img_paths
        self.KDTree = KDTree(data=self.representations)
        self.fe_model = fe_model
        self.k = 3

    def predict(self, sample):
        # Sample contains audio and video. Should be a dict so check keys: maybe 'cam0c'=image, 'audio'=audio
        # Save to MP4, load model, etc

        state = sample['cam0c']
        # state = crop_and_resize(sample['cam0c'])
        audio_list = [torch.tensor(list(audio_tuple)) for audio_tuple in sample['audio'][0]]
        audio_tensor = torch.stack(audio_list)
        # print(f'audio shape: {audio_tensor.shape}')
        # print(f"sample image shape: {np.array(sample['cam0c']).shape}") ## (50, 50, 3)
        state= np.array(state).flatten()
        print(state.sum())
        state = state.reshape([1, -1])
        knn_dis, knn_idx = self.KDTree.query(state, k=self.k)
        #print(knn_dis[0])
        #print(knn_dis[:5])
        return_action = self.actions[knn_idx[0][0]] ### TODO: works for k=1 only so far!!!
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
    return img.resize((50, 50))

def _init_agent_from_config(config, device='cpu'):
    print("Loading KNNVideo................")
    transforms = crop_and_resize
    fe_model = None
    knn_agent = KNNVideo(config.knn.k, config.data.pickle_fn, fe_model)
    return knn_agent, transforms

def img_test(img_path, knn_agent, transforms, k=1):
    img = Image.open(img_path)
    img_tensor = transforms(img)
    state = knn_agent.vision_model(np.array(img_tensor).flatten())

    knn_dis, knn_idx = knn_agent.KDTree.query(state, k=10)

    return knn_agent.img_paths[knn_idx[0][0]]

    
if __name__ == "__main__":
    pass