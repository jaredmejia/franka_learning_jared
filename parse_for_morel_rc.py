import pickle
import csv
import os
from glob import glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image
import shutil

from utils2 import get_reward

# from tracking import get_markers_pos

CAMERA_IDS = {
    "031222070082": 0,  # "818312070212": 0,
}
MARKER_IDS = [4]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--logs_folder",
        type=str,
        help="The folder that contains all logs to parse",
        default="/home/franka/dev/franka_demo/logs/",
    )
    parser.add_argument(
        "-o", "--output_pickle_name", type=str, default="parsed_data.pkl"
    )
    parser.add_argument(
        "-c",
        "--ignore_camera",
        help="When present, expect to NOT parse camera images",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--include_depth",
        help="When expect, include depth camera images",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--ignore_tracking",
        help="When present, expect to NOT get marker positions from images",
        action="store_true",
    )
    parser.add_argument(
        "-rm",
        "--remove_bad_traj",
        help="When present, remove broken trajs",
        action="store_true",
    )
    return parser.parse_args()


args = get_args()

logs_folder = args.logs_folder + ("/" if args.logs_folder[-1] != "/" else "")
logs_folder += "*/[0-9]*[0-9]/"
name_of_demos = glob(logs_folder)
print(f"Found {len(name_of_demos)} recordings in the {logs_folder}")
input(f"Press Enter to continue")

if len(name_of_demos) == 0:
    exit()

if os.path.isfile(args.output_pickle_name):
    print(args.output_pickle_name)
    print(f"Output file existing. Press to confirm overwriting!")
    input()


list_of_demos = []
count_dp = 0
num_valid = 0
for demo in sorted(name_of_demos):
    print(demo)
    traj_reward, valid_data = get_reward(demo)
    # if traj_reward is None:
    #    if args.remove_bad_traj:
    #        print("Removing trajectory",demo)
    #        shutil.rmtree(demo)
    #    continue

    print("Reward for traj: ", traj_reward)
    data_len = len(valid_data)
    print(f"Found {data_len} valid datapoints.")
    num_valid += data_len

    # Extract info
    jointstates = np.array(
        [
            np.fromstring(_data, dtype=np.float64, sep=" ")
            for _data in valid_data["robostate"]
        ]
    )
    commands = np.array(
        [
            np.fromstring(_data, dtype=np.float64, sep=" ")
            for _data in valid_data["robocmd"]
        ]
    )
    print(f"command shape: {commands.shape}")
    if len(commands.shape) != 2:
        import pdb

        pdb.set_trace()
    traj_id = os.path.join(
        os.path.basename(os.path.dirname(demo[:-1])), os.path.basename(demo[:-1])
    )
    info = {
        "traj_id": traj_id,  # removing trailing /
        "jointstates": jointstates,
        "commands": commands,
    }

    if not args.ignore_camera:
        for i in CAMERA_IDS.values():
            info[f"cam{i}c"] = list(valid_data[f"cam{i}c"])
            if args.include_depth:
                info[f"cam{i}d"] = list(valid_data[f"cam{i}d"])

    if not args.ignore_tracking:
        for idx in MARKER_IDS:
            info[f"marker{idx}"] = []

        for i in range(len(info["jointstates"])):
            imgs_ti = []
            for _c in CAMERA_IDS.values():
                img_fn = "{}/{}".format(info["traj_id"], info[f"cam{_c}c"][i])
                img = Image.open(os.path.join(args.logs_folder, img_fn)).copy()
                imgs_ti.append(np.array(img))

            markers_pos_ti = get_markers_pos(imgs_ti, marker_ids=MARKER_IDS)
            for idx in MARKER_IDS:
                info[f"marker{idx}"].append(markers_pos_ti[idx])

    ######################################################
    info["observations"] = info["jointstates"][:, :7]  # TODO: no velocity for now!
    info["actions"] = info["commands"]
    termination = np.zeros(len(info["actions"]))
    termination[-1] = 1
    info["terminated"] = termination
    info["rewards"] = np.zeros(len(info["actions"]))
    info["rewards"][-1] = traj_reward

    info.pop("commands", None)
    info.pop("jointstates", None)
    ######################################################

    list_of_demos.append(info)
    count_dp += len(valid_data)
print(f"Total number of valid datapoints: {num_valid}")
print(f"Num good demos: {len(list_of_demos)}")
with open(args.output_pickle_name, "wb") as f:
    pickle.dump(list_of_demos, f)
