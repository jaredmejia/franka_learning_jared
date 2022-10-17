import collections
import numpy as np
import os
import pandas as pd
#from models.model_loading import MODEL_LIST, load_pvr_transforms

CAMERA_IDS = {
    "031222070082": 0,
    #"815412070907": 1,
    #"818312070212": 2,
}

def validate_images(path, csv_data):
    color_cam_fn = {i:[] for i in CAMERA_IDS.values()}
    depth_cam_fn = {i:[] for i in CAMERA_IDS.values()}

    for dp_idx, timestamps in enumerate(csv_data['cam']):
        timestamps = timestamps[16:]
        if pd.isna(timestamps) or len(timestamps.split('-')) != 2:
            for i in CAMERA_IDS.values():
                color_cam_fn[i].append(np.nan)
                depth_cam_fn[i].append(np.nan)
            continue
        datapoint = timestamps.split('-')
        # print(path,CAMERA_IDS,datapoint)
        for cam_id, i in CAMERA_IDS.items():
            cimage_fn = f"c{i}-{cam_id}-{datapoint[i*2]}-color.jpeg"
            txt_audio_fn = f"{cimage_fn[:-4]}txt"
            color_cam_fn[i].append(cimage_fn
                if os.path.isfile(os.path.join(path, cimage_fn)) and os.path.isfile(os.path.join(path, txt_audio_fn))
                else np.nan
            )
            dimage_fn = f"c{i}-{cam_id}-{datapoint[i*2+1]}-depth.png"
            depth_cam_fn[i].append(dimage_fn
                if os.path.isfile(os.path.join(path, dimage_fn))
                else np.nan
            )


    for i in CAMERA_IDS.values():
        csv_data[f"cam{i}c"] = color_cam_fn[i]
        csv_data[f"cam{i}d"] = depth_cam_fn[i]

    old_data = csv_data['cam'].copy()
    # Remove datapoints missing any image
    csv_data.dropna(inplace=True)
    valid_data = csv_data[csv_data['robocmd'] != 'None']

    print(len(valid_data))
    if len(valid_data) < 10: # Constraint on trajectory length
        print('Too short after image validation',path)
        return None
    return valid_data


def get_reward(path):
    try:
        # print(path)
        csv_log = os.path.join(path, 'log.csv')
        csv_data = pd.read_csv(csv_log, names=['timestamp','robostate','robocmd', 'reward', 'cam'])

        reward = csv_data.iloc[-1].reward
        reward = 1
        #if reward == "None" or int(reward) == -1:
        #    print('Bad reward',path)
        #    return None, None
    except Exception as e:
        # File log.csv did not exist or could not be read
        print('Issue reading log',path)
        return None, None

    valid_data = validate_images(path, csv_data)
    if valid_data is None:
        return None, None
    return int(reward), valid_data


class Namespace(collections.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self): 
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)
