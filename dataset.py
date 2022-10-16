from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import itertools
from PIL import Image
from torchvision import transforms

def np_to_tensor(nparr, device):
    return torch.from_numpy(nparr).float().to(device)

def shift_window(arr, window, np_array=True):
    nparr = np.array(arr) if np_array else list(arr)
    nparr[:-window] = nparr[window:]
    nparr[-window:] = nparr[-1] if np_array else [nparr[-1]] * window
    return nparr

class FrankaDataset(Dataset):

    def __init__(self, data,
            logs_folder='./',
            subsample_period=1,
            im_h=480,
            im_w=640,
            obs_dim=7,
            action_dim=7,
            relabel=None,
            device='cpu',
            cameras=None,
            img_transform_fn=None,
            tracking_info=None,
            debug=False,
            noise=None):
        self.data = data
        self.logs_folder = logs_folder
        self.subsample_period = subsample_period
        self.im_h = im_h
        self.im_w = im_w
        self.obs_dim = obs_dim
        self.action_dim = action_dim # not used yet
        self.relabel = relabel
        self.demos = data
        self.device = device
        self.cameras = cameras or []
        self.img_transform_fn = img_transform_fn
        self.tracking_info = tracking_info
        self.use_tracking = (tracking_info and len(tracking_info.marker_ids) > 0)
        print("Use tracking: ", self.use_tracking)
        self.debug = debug
        self.noise = noise
        self.subsample_demos()
        if len(self.cameras) > 0:
            self.load_imgs()
        self.process_demos()
        self.process_goals()

    def subsample_demos(self):
        for traj in self.demos:
            for key in ['cam0c', 'observations', 'actions', 'terminated', 'rewards']:
                if key == 'observations':
                    traj[key] = traj[key][:, :self.obs_dim] ##############
                traj[key] = traj[key][::self.subsample_period]

    def process_demos(self): # each datapoint is a single (st, at)!
        inputs, labels = [], []
        for traj in self.demos:
            inputs.append(traj['observations'])
            labels.append(traj['actions'])
        inputs = np.concatenate(inputs, axis=0).astype(np.float64)
        labels = np.concatenate(labels, axis=0).astype(np.float64)
        if self.cameras:
            images = []
            for traj in self.demos:
                images.append(traj['images'])
            images = np.concatenate(images, axis=0).astype(np.float64)
            self.images = images # (#trajs * horizon, 480, 640, 3)
        if self.use_tracking:
            marker_info = self.process_markers()
            inputs = np.hstack([inputs, marker_info])
        self.inputs = np_to_tensor(inputs, self.device)
        self.labels = np_to_tensor(labels, self.device)

    def cache_imgs(self): # read images into memory
        raise ValueError("cache_imgs is deprecated!")
        self.cached_imgs = {}
        self.img_lkup = {_c: [] for _c in self.cameras}
        self.logs_folder = self.data['logs_folder']

        for traj in self.demos:
            for i in range(len(traj['commands'])):
                for _c in self.cameras:
                    img_fn = f"{traj['traj_id']}/{traj[_c][i]}"
                    self.img_lkup[_c].append(img_fn)
                    self.cached_imgs[img_fn] = None

        if self.debug: # To speed up debug, use one dummy image
            a_img_filen = list(self.cached_imgs.keys())[0]
            img = Image.open(os.path.join(self.logs_folder, a_img_filen))
            for img_fn in self.cached_imgs.keys():
                self.cached_imgs[img_fn] = img.copy()
            img.close()
            return

        print(f"Start caching images")
        for img_fn in self.cached_imgs.keys():
            #self.cached_imgs[img_fn] = os.path.join(self.logs_folder, img_fn) # for debug
            img = Image.open(os.path.join(self.logs_folder, img_fn))  # Assuming RGB for now
            self.cached_imgs[img_fn] = img.copy()
            img.close()

        #if self.img_transform_fn:
        #    self.cached_imgs[img_fn] = self.img_transform_fn(self.cached_imgs[img_fn]).to(self.device)

        print(f"End caching images")

    def load_imgs(self):
        print("Start loading images...")
        cnt = 0
        for path in self.demos:
            print(cnt, "  ", path['traj_id'])
            path['images'] = []
            for i in range(path['observations'].shape[0]):
                img = Image.open(os.path.join(self.logs_folder, os.path.join('data', path['traj_id'], path['cam0c'][i])))  # Assuming RGB for now
                img = transforms.Resize((self.im_h, self.im_w))(img) # resize during loading to reduce memory requirement
                path['images'].append(np.asarray(img))
                img.close()
            path['images'] = np.array(path['images'])  # (horizon, img_h, img_w, 3)
            cnt += 1
        print("Finished loading images...")

    def process_markers(self):
        raise ValueError("process_markers should not be used!")
        markers_positions = {f'marker{idx}': [] for idx in self.tracking_info.marker_ids}
        tracking_pos = []
        for traj in self.demos:
            traj_tracking_pos = []
            for i in range(len(traj['commands'])):
                stacked_pos = np.hstack([traj[f'marker{idx}'][i] for idx in self.tracking_info.marker_ids])
                tracking_pos.append(stacked_pos)
                traj_tracking_pos.append(stacked_pos)
            traj['tracking_pos'] = np.vstack(traj_tracking_pos)

        return np.vstack(tracking_pos)

    def process_goals(self):
        # raise ValueError("need sanity check before proceeding")
        if self.relabel == None:
            self.goals = torch.empty(self.inputs.shape[0], 0).to(self.device)
            print(f"!! Not using any relabeling !!")
        elif self.relabel.src == "jointstate":
            goals = [
                shift_window(traj['observations'], self.relabel.window)
                for traj in self.demos]
            goals = np.concatenate(goals, axis=0).astype(np.float64)
            self.goals = np_to_tensor(goals, self.device)
        elif self.relabel.src == "tracking":
            goals = [
                shift_window(traj['tracking_pos'], self.relabel.window)
                for traj in self.demos]
            goals = np.concatenate(goals, axis=0).astype(np.float64)
            self.goals = np_to_tensor(goals, self.device)
        elif self.relabel.src.startswith("cam"):
            goals = [
                shift_window(traj['images'], self.relabel.window)
                for traj in self.demos]
            goals = np.concatenate(goals, axis=0).astype(np.float64)
            self.goals = np_to_tensor(goals, self.device)
            # cam = self.relabel.src
            # idx = 0
            # goals = []
            # for traj in self.demos:
            #     img_ref = self.img_lkup[cam][idx:idx+len(traj['commands'])]
            #     traj_goal = shift_window(img_ref, self.relabel.window, np_array=False)
            #     idx += len(traj['commands'])
            #     goals.append(traj_goal)
            # self.goals = list(itertools.chain.from_iterable(goals))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        datapoint = {
                'inputs': self.inputs[idx],
                'labels': self.labels[idx],
                }
        if self.noise:
            datapoint['inputs'] += torch.randn_like(datapoint['inputs']) * self.noise
        if self.cameras:
            for _c in self.cameras:
                img = Image.fromarray(self.images[idx].astype(np.uint8))
                datapoint[_c] = (self.img_transform_fn(img) if self.img_transform_fn
                                    else img).float().to(self.device)
                # datapoint[_c] = (
                #         self.cached_imgs[self.img_lkup[_c][idx]] if self.img_transform_fn is None
                #         else self.img_transform_fn(self.cached_imgs[self.img_lkup[_c][idx]])
                #     ).float().to(self.device)
        if self.relabel:
            if self.relabel.src in ["jointstate", "tracking"]:
                datapoint['goals'] = self.goals[idx]
            else:
                img = Image.fromarray(self.goals[idx])
                datapoint['goals'] = (self.img_transform_fn(img) if self.img_transform_fn
                                        else img).float().to(self.device)
        else:
            datapoint['goals'] = self.goals[idx]
        return datapoint

if __name__ == "__main__":
    from utils2 import Namespace

    demos = [
            {
                'jointstates':np.arange(5).reshape(5,1),
                'commands':np.arange(10).reshape(5,2),
                'traj_id': 't0',
                'cam0c': [f'{i}.jpeg' for i in range(5)]
                },
            {
                'jointstates': np.arange(3).reshape(3,1) + 200,
                'commands': np.arange(6).reshape(3,2),
                'traj_id': 't1',
                'cam0c': [f'{i}.jpeg' for i in range(3)]
                }
            ]
    data ={
        'demos': demos,
        'logs_folder': 'logsfolder',
    }
    my_dataset = FrankaDataset(data, cameras=[], relabel=
        Namespace({'window':2,
         'src': "jointstate"}))

    # testing data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(my_dataset,
        batch_size=2, shuffle=False)
    for data in train_loader:
        print(data)

