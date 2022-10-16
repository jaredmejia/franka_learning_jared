# plot everything in the folder
import numpy as np
import copy
import pickle
import argparse
import os
import json
import matplotlib.pyplot as plt

JOINT_LIMIT_MIN = np.array(
    [-2.8773, -1.7428, -2.8773, -3.0018, -2.8773, 0.0025, -2.8773])
JOINT_LIMIT_MAX = np.array(
    [2.8773, 1.7428, 2.8773, -0.1398, 2.8773, 3.7325, 2.8773])


def plot_actions(paths, keys=None, caption="Actions_comparison"):
    if not keys:
        keys = [i for i in range(len(paths))]
    fig, axs = plt.subplots(2, 4, figsize=(28, 12))
    count = 0
    for i in range(2):
        for j in range(4):
            if count < paths[0]['actions'].shape[1]:
                for p in range(len(paths)):
                    actions = paths[p]['actions']
                    subsample = round(764 / actions.shape[0])
                    xis = [i for i in range(0, 764, subsample)][:actions.shape[0]]
                    print("states.shape[0], subsample: ", actions.shape[0], subsample)
                    axs[i, j].plot(xis, actions[:, count], label=keys[p])
                axs[i, j].set_title("action[{}]".format(count))
                axs[i, j].legend(loc="upper left")
                count += 1
    fig.suptitle(caption)
    handles, labels = axs[i, j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    fig.savefig(os.path.join(output_folder, caption))

def plot_states(paths, keys=None, caption="States_comparison"):
    if not keys:
        keys = [f"{i}" for i in range(len(paths))]

    fig, axs = plt.subplots(2, 4, figsize=(28, 12))
    count = 0
    for i in range(2):
        for j in range(4):
            if count < paths[0]['observations'].shape[1]:
                # axs[i, j].set_ylim(JOINT_LIMIT_MIN[count], JOINT_LIMIT_MAX[count])
                for p in range(len(paths)):
                    states = paths[p]['observations']
                    subsample = round(764 / states.shape[0])
                    xis = [i for i in range(0, 764, subsample)][:states.shape[0]]
                    print("states.shape[0], subsample: ", states.shape[0], subsample, len(xis))
                    axs[i, j].plot(xis, states[:, count], label=keys[p])
                axs[i, j].set_title("state[{}]".format(count))
                axs[i, j].legend(loc="upper left")
                count += 1
            else:
                break
    fig.suptitle(caption)
    print("i, j: ", i, j)
    handles, labels = axs[i, j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    # import pdb; pdb.set_trace()
    fig.savefig(os.path.join(output_folder, caption))

if __name__ == "__main__":
    output_folder = '/home/gaoyue/Desktop'
    train_paths = pickle.load(open('/home/gaoyue/dev/franka_learning/traj_replay_test.pkl', 'rb'))['demos']
    original_path = pickle.load(open('/home/gaoyue/dev/franka_learning/4-28-scooping-red-front-center-mints.pkl', 'rb'))['demos']
    original_path = original_path[0]

    for path in train_paths:
        path['observations'] = path['jointstates']
        path['actions'] = path['commands']
    original_path['observations'] = original_path['jointstates']
    original_path['actions'] = original_path['commands']

    paths = [train_paths[0], train_paths[-1]]
    paths.extend(train_paths[1:4])
    plot_states(paths, keys=['replay_f40_s1', 'f20_s2', 'f10_s4', 'f8_s5', 'f5_s8', 'f20_s2'])
    plot_actions(paths, keys=[ 'replay_f40_s1', 'f20_s2', 'f10_s4', 'f8_s5', 'f5_s8'])

    

    import pdb; pdb.set_trace()


    plot_states(train_paths, ["abs", "deltaaction", "deltastate"], "replay_compare_state")
    plot_actions(train_paths, ["abs", "deltaaction", "deltastate"], "replay_compare_action")
    # plot_states([train_paths[12], train_paths_smoothed[12] ], ["original", "0.25 smoothed"], "smoothingtst")
    # # plot_states(train_paths[10:14], [i for i in range(10, 14)])
    # plot_actions([train_paths[10]], [10], "deltastate-1-traj")
    import pdb; pdb.set_trace()