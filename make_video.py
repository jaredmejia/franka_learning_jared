import cv2
import os
import argparse
import glob
from utils2 import get_reward

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/gaoyue/dev/franka_demo/logs/4-1-scooping-plate-1')
parser.add_argument('--novideo', action='store_true')
args = parser.parse_args()

#demo_folder = args.path
if args.path[-1] == '/':
    args.path = args.path[:-1]
height = 480
width = 640
frame_rate = 30 * 10



def write_to_video(trial_path, video, reward, trial_number):
    # print("Writing trial", trial_number, trial_path)
    images = sorted([img for img in os.listdir(trial_path) if img.endswith("color.jpeg")])

    for image_name in images:
        image = cv2.imread(os.path.join(trial_path, image_name))
        text = "Trial " + str(trial_number) + ": " + str(reward) + "g"
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, text, (5,465), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
        video.write(image)


folders = glob.glob(args.path)
for demo_folder in sorted(folders):
    if not os.path.isdir(demo_folder):
        continue
    video_save_folder = "/home/gaoyue/dev/franka_demo/logs/videos"
    video_name = os.path.join(video_save_folder, demo_folder.split('/')[-1] + '_video.mp4')
    if not args.novideo:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width,height))
    print("Saving",demo_folder,"to",video_name)
    good_trial_counter = 0
    bad_trial_counter = 0
    for trial in sorted(os.listdir(demo_folder)):
        trial_path = os.path.join(demo_folder, trial)
        if not os.path.isdir(trial_path):
            continue

        reward, _ = get_reward(trial_path)
        if reward is None:
            bad_trial_counter += 1
            continue

        if not args.novideo:
            write_to_video(trial_path, video, reward, good_trial_counter)
        good_trial_counter += 1

    print ("  ",good_trial_counter,"good,",bad_trial_counter,"bad trials")
    if not args.novideo:
        video.release()
    # os.system("xdg-open " + video_name)

cv2.destroyAllWindows()
