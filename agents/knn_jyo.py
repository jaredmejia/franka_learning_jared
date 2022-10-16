import numpy as np
from scipy import special
import pickle

class KNNJyo(object):
    def __init__(self, k, pickle_fn):
        paths = pickle.load(open(pickle_fn, 'rb'))
        self.s = np.concatenate([p['observations'][:-1] for p in paths if p['rewards'][-1] > 0])
        self.a = np.concatenate([p['actions'][:-1] for p in paths])
        self.k = k

    def predict(self, sample):
        print("knn jyo predict")
        state = sample['inputs']
        return_action = np.zeros(state.shape[0])
        dist_action = [] #list of tuples (distance between state and observation_i, action_i)
        for i in range(self.s.shape[0]):
            dist = np.linalg.norm(self.s[i] - state)
            dist_action.append((dist, self.a[i]))
        dist_action.sort(key=lambda x: x[0])

        weights = [-1*(dist_action[i][0]) for i in range(self.k)]
        weights = special.softmax(weights)
        
        for i in range(self.k):
            return_action += weights[i] * dist_action[i][1]
        return return_action

def _init_agent_from_config(config, dataset):
    knn_agent = KNNJyo(config.knn.k, config.data.pickle_fn)
    return knn_agent