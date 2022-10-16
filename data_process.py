import pickle
import numpy as np

train_paths1 = pickle.load(open('/home/gaoyue/dev/franka_learning/parsed_datasets/inserting-yellow.pkl', 'rb'))
train_paths2 = pickle.load(open('/home/gaoyue/dev/franka_learning/parsed_datasets/inserting-blue.pkl', 'rb'))

for path1 in train_paths1:
    path1['observations'] = path1['observations'][:, :7]
    path1['observations'] = np.hstack([path1['observations'], np.ones([path1['observations'].shape[0], 1]), np.zeros([path1['observations'].shape[0], 1])])
for path2 in train_paths2:
    path2['observations'] = path2['observations'][:, :7]
    path2['observations'] = np.hstack([path2['observations'], np.zeros([path2['observations'].shape[0], 1]), np.ones([path2['observations'].shape[0], 1])])

train_paths1.extend(train_paths2)
with open('/home/gaoyue/dev/franka_learning/parsed_datasets/inserting-yellow-blue-dim8.pkl', 'wb') as f:
    pickle.dump(train_paths1, f)
import pdb; pdb.set_trace()