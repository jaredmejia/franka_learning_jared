import numpy as np

#robo_xyz = np.array( [0.59967156, 0.34842243, 0.24311814])
#tag_xyz = np.array( [-0.07842285,  0.20901184, -0.22535597])
offset = np.array([0.6, 0.07, 0.46])

def coor_transform(tag_frame_pos):
    xt, yt, zt = tag_frame_pos
    x, y, z = -zt, -xt, yt
    franka_frame_pos = np.array([x, y, z]) + offset
    return franka_frame_pos

def metrics_lift_obj(initial_tag_pos, cur_tag_pos, ee_pos):
    assert np.linalg.norm(initial_tag_pos) != 0
    lift_threshold = 0.0
    box_height = 0.2
    initial_tag_pos_robo = coor_transform(initial_tag_pos)
    cur_tag_pos_robo = coor_transform(cur_tag_pos)

    rest_tag_height = initial_tag_pos_robo[2]
    amount_lift = cur_tag_pos_robo[2] - rest_tag_height
    if amount_lift > 0.02 and np.linalg.norm(cur_tag_pos) != 0: # 0.02 for tracking error tolerance, ideally 0 would suffice;
        return max(0, lift_threshold - amount_lift)             # if tag not in sight, assume didn't lift up

    initial_handle_pos = initial_tag_pos_robo + np.array([0, 0, box_height])
    l2_err = np.linalg.norm(ee_pos - initial_handle_pos)
    print("ee pos: ", ee_pos, "      initial_tag_pos_robo: ", initial_tag_pos_robo, "     initial_handle_pos: ", initial_handle_pos)
    return l2_err


def metrics_reach_tag(initial_tag_pos, cur_tag_pos, ee_pos):
    # print(initial_tag_pos, cur_tag_pos, ee_pos)
    assert np.linalg.norm(initial_tag_pos) != 0
    initial_tag_pos_robo = coor_transform(initial_tag_pos)
    cur_tag_pos_robo = coor_transform(cur_tag_pos)
    if np.linalg.norm(cur_tag_pos) == 0: # tag not in sight
        return np.linalg.norm(ee_pos - initial_tag_pos_robo)
    else:
        return np.linalg.norm(ee_pos - cur_tag_pos_robo)


def metrics_reach_tag_init_only(initial_tag_pos, ee_pos):
    assert np.linalg.norm(initial_tag_pos) != 0
    initial_tag_pos_robo = coor_transform(initial_tag_pos)
    return np.linalg.norm(ee_pos - initial_tag_pos_robo)

def get_rewards(info):
    from franka_wrapper import FrankaWrapper
    f = FrankaWrapper()
    n = len(info['commands'])
    tag_xyz = info['goal']
    #print(tag_xyz, coor_transform(tag_xyz))
    #print(info['jointstates'][0], f.get_fk(info['jointstates'][0]))
    #print(f.get_fk(info['jointstates'][-1]))
    dis = coor_transform(tag_xyz) - f.get_fk(info['jointstates'][-1])
    #print(dis)

    rew = np.zeros(n)
    for i in range(n):
        rew[i] = 0.5 - metrics_reach_tag_init_only(tag_xyz, f.get_fk(info['jointstates'][i]))
    #import pdb;pdb.set_trace()
    #print(rew[::30])
    return rew