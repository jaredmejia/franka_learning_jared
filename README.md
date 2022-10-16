# Franka Learning

- Use `parse.py` to parse logs and generate relabel. It will generate `parsed_data.pkl`.
    `python parse.py`
- Use `train.py` to train a behavior cloning agent. The default hydra config is `conf/train.yaml`.
    `python train.py training.batch_size=64`
- Use `test_real.py` to test the agent on real robot.
    `python test_real.py -f FOLDER_TO_HYDRA_OUTPUT`