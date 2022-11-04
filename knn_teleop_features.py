from sklearn.neighbors import KDTree
import argparse
import joblib
import numpy as np
import os
import pickle
import sys
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(1, "/home/vdean/jared_contact_mic/avid-glove")

# avid-glove imports
from inference import (
    LivePrep,
    LivePrepCat,
    LivePrepVideo,
    LivePrepAudio,
    get_feature_extractor,
)
from main import LightningModel
from utils import misc


parser = argparse.ArgumentParser(
    description="Extract features from expert teleop demonstrations and fit to KNN"
)
parser.add_argument(
    "--backbone_cfg",
    default="/home/vdean/jared_contact_mic/avid-glove/config/gloveaudio/avcat-avid-ft-jointloss.yaml",
    help="path to config file associated with backbone model for feature extracting",
)
parser.add_argument(
    "--teleop_pickle",
    default="/home/vdean/franka_learning_jared/jared_chopping_exps.pkl",
    help="path to teleop demonstration data pickled object",
)
parser.add_argument(
    "--teleop_dir", default="/home/vdean/franka_demo/logs/jared_chopping_exps/"
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory to save fitted KNN model and corresponding actions to",
)
parser.add_argument(
    "--num_images_cat",
    default=8,
    type=int,
    help="If greater than 1, combines multiple images/audio to emulate video (as opposed to single frame passed to model)",
)
parser.add_argument(
    "--raw_input",
    default=None,
    help="If None, KNN is fit on features extracted from NN model, otherwise KNN is fit on type of raw input (audio, image, audio-image)",
)
parser.add_argument(
    "--pretraining",
    default="finetuned",
    choices=["finetuned", "non-finetuned", "random"],
    type=str,
)
parser.add_argument("--save_knn", default=False, type=bool)
parser.add_argument("--unimodal", default=False, type=bool)
parser.add_argument("--batch_size", default=512)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AVID_DIR = "/home/vdean/jared_contact_mic/avid-glove/"


def get_teleop_data(teleop_paths, teleop_dir, num_images_cat=8):
    actions = {}
    traj_ids = []
    img_paths = []
    for path in teleop_paths:
        data_path = os.path.join(teleop_dir, path["traj_id"])
        traj_img_paths = [
            os.path.join(data_path, img_file) for img_file in path["cam0c"]
        ]
        for i in range(num_images_cat, len(traj_img_paths)):
            # taking chunks of size num_cat_images
            img_path_cat = traj_img_paths[i - num_images_cat : i]
            img_paths.append(img_path_cat)
            traj_ids.append((path["traj_id"], i - num_images_cat))

        actions[path["traj_id"]] = path["actions"][num_images_cat:]

    return actions, traj_ids, img_paths


def collate_fn_joint(batch):
    batched_audio, batched_video = [], []
    for data in batch:
        batched_audio.append(data["audio"])
        batched_video.append(data["video"])
    batched_audio = torch.stack(batched_audio, dim=0).to(device)
    batched_video = torch.stack(batched_video, dim=0).to(device)
    batched_data = {"video": batched_video, "audio": batched_audio}
    return batched_data


def collate_fn_video(batch):
    batched_video = []
    for data in batch:
        batched_video.append(data["video"])
    batched_video = torch.stack(batched_video, dim=0).to(device)
    batched_data = {"video": batched_video}
    return batched_data


def collate_fn_audio(batch):
    batched_audio = []
    for data in batch:
        batched_audio.append(data["audio"])
    batched_audio = torch.stack(batched_audio, dim=0).to(device)
    batched_data = {"audio": batched_audio}
    return batched_data


# TODO: implement
def get_knn_features_raw(img_paths, raw_modality):
    raise NotImplementedError

    if raw_modality == "image":
        pass
    elif raw_modality == "audio":
        pass
    else:  # raw_modality == 'audio-image'
        pass
    return


def get_knn_features_model(dl, fe_model, unimodal=False):
    feature_list = []
    fe_model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dl)):
            if unimodal:
                features = fe_model(data)
            else:
                features, _, _ = fe_model(data)
            feature_list.append(features.detach())
    representations = torch.cat(feature_list, dim=0)
    return representations.cpu()


def load_model(cfg, pretraining="finetuned"):
    if pretraining == "finetuned":
        checkpoint_path = os.path.join(cfg.save_path, cfg.name, "checkpoints/last.ckpt")
        print(f"Loading: {checkpoint_path}")
        model = LightningModel.load_from_checkpoint(checkpoint_path)

    elif pretraining == "random":
        model = LightningModel(cfg)
        random_pt = os.path.join(AVID_DIR, "models/avid/ckpt/AVID_random.pt")
        print(f"Loading: {random_pt}")
        model.model.load_state_dict(torch.load(random_pt))
    else:
        model = LightningModel(cfg)
        pretrained_checkpoint = os.path.join(AVID_DIR, cfg.model.checkpoint)

        print(f"loading: {pretrained_checkpoint}")
        ckp = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

        if cfg.criterion.mode == "unimodal":
            if cfg.model.args.modality == "video":
                ckp_prefix = "module.video_model."
            elif cfg.model.args.modality == "audio":
                ckp_prefix = "module.audio_model."
            else:
                raise ValueError("Unknown modality.")
            unimodal_model_ckp = {
                k[len(ckp_prefix) :]: ckp[k] for k in ckp if k.startswith(ckp_prefix)
            }
            model.model.unimodal_model.load_state_dict(unimodal_model_ckp)
            del unimodal_model_ckp
        else:
            visual_model_ckp = {
                k[len("module.video_model.") :]: ckp[k]
                for k in ckp
                if k.startswith("module.video_model.")
            }
            model.model.visual_model.load_state_dict(visual_model_ckp)
            audio_model_ckp = {
                k[len("module.audio_model.") :]: ckp[k]
                for k in ckp
                if k.startswith("module.audio_model.")
            }
            model.model.audio_model.load_state_dict(audio_model_ckp)
            del visual_model_ckp, audio_model_ckp, ckp
    return model


def main():
    args = parser.parse_args()

    backbone_cfg = misc.convert2namespace(yaml.safe_load(open(args.backbone_cfg)))
    model = load_model(backbone_cfg, args.pretraining)

    fe_model = get_feature_extractor(model.model, unimodal=args.unimodal).to(device)

    with open(args.teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    actions, traj_ids, img_paths = get_teleop_data(
        teleop_paths, args.teleop_dir, args.num_images_cat
    )
    print(f"Total number of samples: {len(img_paths)}")
    print(f"Num action trajectories: {len(actions)}")
    print(f"Num images cat: {args.num_images_cat}")

    if args.save_knn:
        if args.raw_input:
            representations = get_knn_features_raw(img_paths, args.raw_input)
        else:
            if args.num_images_cat > 1:
                if args.unimodal:
                    if backbone_cfg.model.args.modality == "video":
                        prep_data = LivePrepVideo(
                            backbone_cfg.dataset,
                            image_path_lists=img_paths,
                            device=device,
                        )
                        collate_fn = collate_fn_video
                    else:
                        prep_data = LivePrepAudio(
                            backbone_cfg.dataset,
                            image_path_lists=img_paths,
                            device=device,
                        )
                        collate_fn = collate_fn_audio
                else:
                    prep_data = LivePrepCat(
                        backbone_cfg.dataset, image_path_lists=img_paths, device=device
                    )
                    collate_fn = collate_fn_joint
            else:
                prep_data = LivePrep(
                    backbone_cfg.dataset, image_paths=img_paths, device=device
                )
                collate_fn = collate_fn_joint

            dl = DataLoader(
                dataset=prep_data,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            representations = get_knn_features_model(
                dl, fe_model, unimodal=args.unimodal
            )
            print(f"Embeddings shape: {representations.shape}")

        knn_model = KDTree(data=representations)
        model_fn = os.path.join(args.output_dir, "knn_model.pkl")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        joblib.dump(knn_model, model_fn)

    actions_fn = os.path.join(args.output_dir, "action_trajs.pkl")
    with open(actions_fn, "wb") as f:
        pickle.dump(actions, f)

    traj_ids_fn = os.path.join(args.output_dir, "traj_ids.pkl")
    with open(traj_ids_fn, "wb") as f:
        pickle.dump(traj_ids, f)


if __name__ == "__main__":
    main()
