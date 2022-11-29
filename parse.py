import pickle
import csv
import os
from glob import glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image

CAMERA_IDS = {
    "031222070082": 0,
    "815412070907": 1,
    "818312070212": 2,
}
MARKER_IDS = [5]


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
    # parser.add_argument('-m','--marker', nargs='*', help='marker ids', type=int)
    return parser.parse_args()


args = get_args()

logs_folder = args.logs_folder + ("/" if args.logs_folder[-1] != "/" else "")
logs_folder += "[0-9]*[0-9]/"
name_of_demos = glob(logs_folder)
print(f"Found {len(name_of_demos)} recordings in the {logs_folder}")

if len(name_of_demos) == 0:
    exit()

if os.path.isfile(args.output_pickle_name):
    print(f"Output file existing. Press to confirm overwriting!")
    input()


list_of_demos = []
count_dp = 0
for demo in sorted(name_of_demos):
    try:
        csv_log = os.path.join(demo, "log.csv")
        csv_data = pd.read_csv(
            csv_log, names=["timestamp", "robostate", "robocmd", "cam"]
        )
    except Exception as e:
        print(f"Exception: cannot read log.csv at {demo}")
        continue

    initial_len = len(csv_data)

    # Validates Camera Images
    if not args.ignore_camera:
        cam_ts = [_timestamps.split("-") for _timestamps in csv_data["cam"]]
        color_cam_fn = {i: [] for i in CAMERA_IDS.values()}
        depth_cam_fn = {i: [] for i in CAMERA_IDS.values()}
        for dp_idx, datapoint in enumerate(cam_ts):
            my_image_fn = []
            for cam_id, i in CAMERA_IDS.items():
                cimage_fn = f"c{i}-{cam_id}-{datapoint[i*2]}-color.jpeg"
                color_cam_fn[i].append(
                    cimage_fn
                    if os.path.isfile(os.path.join(demo, cimage_fn))
                    else np.nan
                )

                if args.include_depth:
                    dimage_fn = f"c{i}-{cam_id}-{datapoint[i*2+1]}-depth.jpeg"
                    depth_cam_fn[i].append(
                        dimage_fn
                        if os.path.isfile(os.path.join(demo, dimage_fn))
                        else np.nan
                    )
        for i in CAMERA_IDS.values():
            csv_data[f"cam{i}c"] = color_cam_fn[i]
            if args.include_depth:
                csv_data[f"cam{i}d"] = depth_cam_fn[i]

        # Remove datapoints missing any image
        csv_data.dropna(inplace=True)

    valid_data = csv_data[csv_data["robocmd"] != "None"]
    print(
        f"Found {initial_len} datapoints. "
        + (f"{len(csv_data)} has all images. " if not args.ignore_camera else "")
        + f"{len(valid_data)} usable. "
    )

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
    info = {
        "traj_id": os.path.basename(demo[:-1]),  # removing trailing /
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

    list_of_demos.append(info)
    count_dp += len(valid_data)

with open(args.output_pickle_name, "wb") as f:
    pickle.dump(
        {
            "logs_folder": args.logs_folder,
            "ignore_camera": args.ignore_camera,
            "include_depth": args.include_depth,
            "ignore_tracking": args.ignore_tracking,
            "demos": list_of_demos,
        },
        f,
    )

print(f"Saved parsed data with {count_dp} data points.")

with open(args.output_pickle_name, "rb") as f:
    x = pickle.load(f)["demos"]
    print(f"Loaded {len(x)} trajectories.")
    print(f"First 5 jointstates from the first trajectory {x[0]['jointstates'][0:5]}")
    print(f"First 5 commands from the first trajectory {x[0]['commands'][0:5]}")

    marker_san_check = []

    if not args.ignore_camera:
        first_cam_id = list(CAMERA_IDS.values())[0]
        first_cam = f"cam{first_cam_id}c"
        print(f"Last color image from the first trajectory cam0 {x[0][first_cam][-1]}")

    if not args.ignore_tracking:
        from tracking import get_markers_pos

        first_tracker = f"marker{MARKER_IDS[0]}"
        print(
            f"Last marker{first_tracker}-pose from the first trajectory {x[0][first_tracker][-1]}"
        )

        for traj in x:
            marker_san_check.append(traj[first_tracker][0])

        marker_san_check = np.array(marker_san_check).T
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(marker_san_check[0], marker_san_check[1], marker_san_check[2])
        plt.show()
