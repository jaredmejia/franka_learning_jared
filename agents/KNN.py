import numpy as np
from sklearn.neighbors import KDTree
from scipy.linalg import sqrtm

from .BCAgent import get_stats

def e_to_negative_x(x):
    w = np.exp(-x)
    return w / np.sum(w)

def uniform_weights(x):
    return None

def generate_norm_weights(normalization, ngram, weights):
    if normalization == 'with_target':
        w = np.ones(8)
        w[0:3] = weights['xyz']
        w[3:7] = weights['quat']
        ngram_w = np.tile(w, ngram)
        w_target = weights['target']
        return np.hstack((ngram_w, np.tile(w_target, 3)))
    if normalization == 'with_target_first':
        w = np.ones(ngram * 8 + 3)
        w[0:3] = weights['xyz']
        w[3:7] = weights['quat']
        w[-3:] = weights['target']
        return w

class KNN(object):
    def __init__(self, k, in_dim, out_dim, dataset,
                 weight_fn=e_to_negative_x,
                 normalization='std', norm_weights=[], ngram=1):

        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.X = np.concatenate((dataset.inputs, dataset.goals), axis=1)
        self.Y = dataset.labels

        self.X_mean = np.zeros(in_dim)
        self.X_std = np.ones(in_dim)
        self.Y_mean = np.zeros(out_dim)
        self.Y_std = np.ones(out_dim)

        self.set_stats(dataset)

        self.weight_fn = weight_fn
        # self.norm_weights = generate_norm_weights(normalization, ngram, norm_weights)
        # # Prepare normalization
        # self.normalization = normalization
        # if self.normalization == 'std':
        #     self.X_mean = np.mean(X, axis=0)
        #     self.X_std = np.std(X, axis=0)
        # elif self.normalization == 'closed-form':
        #     self.scaling = sqrtm(np.linalg.inv(np.matmul(X.T, X)))
        # elif self.normalization == 'with_target':
        #     print('Data shape = ', X.shape)
        #     print('Normalization weights = ', self.norm_weights)
        #     assert(X.shape[1] == len(self.norm_weights))
        self.X = self._normalize(self.X)
        self.KDTree = KDTree(data=self.X)
        print(f"KNN, k= {k} stores {len(self.X)} datapoints")

    def set_stats(self, dataset):

        inp_dim, inp_mean, inp_std = get_stats(dataset.inputs)
        goal_dim, goal_mean, goal_std = get_stats(dataset.goals)

        assert(inp_dim + goal_dim == self.in_dim)

        self.X_mean[:inp_dim] = inp_mean
        self.X_std[:inp_dim] = inp_std
        self.X_mean[inp_dim:inp_dim + goal_dim] = goal_mean
        self.X_std[inp_dim:inp_dim + goal_dim] = goal_std

        out_dim, self.out_mean, self.out_std = get_stats(dataset.labels)
        assert(out_dim == self.out_dim)

    # For 1-batch query only!
    def predict(self, sample):
        one_input = np.concatenate((sample['inputs'], sample['goals']))
        x = one_input.reshape(1,-1)
        x = self._normalize(x)
        knn_dis, knn_idx = self.KDTree.query(x, k=self.k)
        weights = self.weight_fn(knn_dis[0]) # we only queried one point
        values = self.Y[knn_idx[0]]
        return np.average(values, axis=0, weights=weights)

    def _normalize(self, x):
        x = (x - self.X_mean) / self.X_std
        return x
        # TODO: add normalization methods? Optional


def _init_agent_from_config(config, dataset):
    tracking_state_dim = (len(config.data.tracking.marker_ids) * 3
        if config.data.tracking is not None
        else 0)
    input_dim = config.data.in_dim + tracking_state_dim
    if config.data.relabel is not None and config.data.relabel.window > 0:
        if config.data.relabel.src == "jointstate":
            input_dim += config.data.in_dim
        elif config.data.relabel.src == "tracking":
            input_dim += tracking_state_dim

    knn_agent = KNN(config.knn.k, input_dim, config.data.out_dim, dataset,
        norm_weights=config.knn.weights)
    return knn_agent, None # image transforms