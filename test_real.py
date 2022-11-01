import yaml, sys, argparse, os, pickle
import numpy as np
from time import time, sleep
from PIL import Image
import torch

from franka_demo_remote.demo_interfaces import run_demo
from franka_demo_remote.demo_interfaces import STATE_UPDATE_FREQ
from franka_demo_remote.addon import add_camera_function, add_logging_function

from dataset_lazy import FrankaDataset
from agents import init_agent_from_config
# from tracking import get_markers_pos
from utils2 import Namespace
#from models.model_loading import MODEL_LIST, load_pvr_model, preprocess_image

def add_imitation_function(state):
    state.handlers['i'] = _press_imitate
    state.modes['imitate'] = __cmd_imitate
    load_imitation_agent(state)
    state.shuffle_goals_idx = -1
    #state.imt_counter = []

def _press_imitate(key_pressed, state):
    state.mode = 'imitate'
    state.shuffle_goals_idx += 1

    state.ngram = state.params['agent_cfg'].data.ngram or 0
    state.ngram_inputs = np.tile(state.robostate, state.ngram)

def __cmd_imitate(state, timestamp):
    query_state = gen_imitation_state(state)
    command = state.agent.predict(query_state)
    #state.imt_counter.append(time())
    #if len(state.imt_counter) > 1:
    #    print(len(state.imt_counter)/(state.imt_counter[-1]  - state.imt_counter[0]))
    command += np.random.standard_normal(size=command.shape) * 0.005 # Was .005
    np.clip(command, state.robostate+state.CMD_DELTA_LOW, state.robostate+state.CMD_DELTA_HIGH, out=command)
    return command

def load_imitation_agent(state):
    agent_cfg = state.params['agent_cfg']
    state.cam_in_state = agent_cfg.data.images.cameras

    if agent_cfg.agent.type == 'knn': ### TODO: sanity check knn
        device = 'cpu'
        with open(cfg.data.pickle_fn, 'rb') as f:
            data = pickle.load(f)
        dset = FrankaDataset(data,
            relabel=agent_cfg.data.relabel,
            device=device,
            cameras=[],
            tracking_info=agent_cfg.data.tracking,
            img_transform_fn=None)
        state.agent, state.img_transform_fn = init_agent_from_config(agent_cfg, device, normalization=dset)
    elif agent_cfg.agent.type in ['bc', 'bcimage', 'bcimagegoal']:
        device = 'cuda:0'
        with open(cfg.data.pickle_fn, 'rb') as f:
            data = pickle.load(f)
        dset = FrankaDataset(data,
            logs_folder=agent_cfg.data.logs_folder,
            subsample_period=agent_cfg.data.subsample_period,
            im_h=agent_cfg.data.images.im_h,
            im_w=agent_cfg.data.images.im_w,
            obs_dim=agent_cfg.data.in_dim,
            action_dim=agent_cfg.data.out_dim,
            relabel=agent_cfg.data.relabel,
            device=agent_cfg.training.device,
            cameras=[],
            img_transform_fn=None,
            tracking_info=agent_cfg.data.tracking)  # load dummy dset so we can set stats
        state.agent, state.img_transform_fn = init_agent_from_config(agent_cfg, device, normalization=dset)
        state.agent.load(agent_cfg['saved_folder'])
    else: ## knn_image, visuomotor, and other custom agents
        device = 'cuda:0'
        state.agent, state.img_transform_fn = init_agent_from_config(agent_cfg, device)

def sliding_window(arr, new_item):
    new_arr = np.zeros_like(arr)
    dim = len(new_item)
    new_arr[0:-dim] = arr[dim:]
    new_arr[-dim:] = new_item
    return new_arr

def gen_imitation_state(state):
    imitation_state = {
        "inputs": np.array(state.robostate),
    }

    if state.ngram > 1:
        state.ngram_inputs = sliding_window(state.ngram_inputs, state.robostate)
        imitation_state['inputs'] = state.ngram_inputs

    cam_state = state.cameras.get_data()
    #for _c in state.cam_in_state:
    for _c in ['cam0c']:
        img = Image.fromarray(cam_state[_c][0][:,:,::-1]) # TODO(kathy) double check the ordering for visuomotor
        imitation_state[_c] = state.img_transform_fn(img)
    for _m in ['audio']:
        ccc= cam_state[_m]
        imitation_state[_m] = cam_state[_m]

    goal_idx = state.shuffle_goals_idx % len(state.params['shuffle_goals'])
    goal_vec = state.params['shuffle_goals'][goal_idx]

    if cfg.data.relabel == None:
        imitation_state["goals"] = torch.empty(0)
    elif state.params['agent_cfg'].data.relabel.src in ["jointstate", "tracking"]:
        imitation_state["goals"] = np.array(goal_vec)
    else:
        imitation_state["goals"] = state.img_transform_fn(goal_vec)

    return imitation_state

def callback_func(state):
    add_logging_function(state)
    add_imitation_function(state)
    add_camera_function(state)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--server_ip",
                        type=str,
                        help="IP address or hostname of the franka server",
                        default="172.16.0.1")
    parser.add_argument("-f", "--agent_folder",
                        type=str,
                        default="/home/franka/dev/franka_learning/outputs")
    parser.add_argument("-a", "--agent_name",
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    with open(os.path.join(args.agent_folder, args.agent_name, 'hydra.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        # Overwriting to use local
        cfg['saved_folder'] = os.path.join(args.agent_folder, args.agent_name)
        cfg = Namespace(cfg)

    if cfg.data.relabel == None:
        shuffle_goals = [0]
    else:
        goal_yaml = os.path.join('goals', cfg.data.pickle_fn.rsplit('.', 1)[0] + '.yaml')
        with open(goal_yaml, 'r') as goal_file:
            goals = yaml.load(goal_file, Loader=yaml.FullLoader)

        if cfg.data.relabel.src == "jointstate":
            shuffle_goals = [np.fromstring(_str, dtype=float, sep=' ') for _str in
                goals['jointpos']]
        elif cfg.data.relabel.src == "tracking":
            shuffle_goals = [np.fromstring(_str, dtype=float, sep=' ') for _str in
                goals['tags']]
        else:
            goal_img_list = goals['img']
            goal_imgs = [Image.open(filename) for filename in goal_img_list]
            shuffle_goals = [img.copy() for img in goal_imgs]
            _ = [img.close() for img in goal_imgs]

    update_freq = STATE_UPDATE_FREQ 
    if cfg.data.subsample_period != {}:
        update_freq = STATE_UPDATE_FREQ / cfg.data.subsample_period
    
    log_folder = os.path.join(cfg['saved_folder'], f"exp_H-{cfg['knn']['H']}_k-{cfg['knn']['k']}")
    run_demo(callback_func, params={
        'ip_address': args.server_ip,
        'log_folder': log_folder,
        'agent_cfg': cfg,
        'shuffle_goals': shuffle_goals,
        'gripper': False,
        'remote': False
    })#, state_update_freq=update_freq, use_keyboard_listener=True)
