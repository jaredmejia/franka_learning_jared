### Please ignore all tracking related parts for RoboCloud
import arucoX as ax
# from realsense import RealSense
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import sophus
# Initialize camera modules

def get_markers_pos(imgs_ti, intrinsics=None, extrinsics=None, marker_ids=[1], marker_size=0.076):
    c1 = ax.CameraModule()
    c2 = ax.CameraModule()
    # c3 = ax.CameraModule()
    # cameras = [c1, c2, c3]
    cameras = [c1, c2]
    scene = ax.Scene(cameras=cameras)

    for i in range(len(marker_ids)):
        scene.register_marker_size(marker_ids[i], marker_size)

    # TODO(during data collection): need to save the intrinsic matrices and pose for all cameras
    if extrinsics is None:
        # intrinsics = [(np.array([[609.14013672,   0.        , 317.13171387],
        #     [  0.        , 609.48712158, 242.33857727],
        #     [  0.        ,   0.        ,   1.        ]]), np.array([0., 0., 0., 0., 0.])), (np.array([[608.03778076,   0.        , 310.80316162],
        #     [  0.        , 608.10168457, 230.40539551],
        #     [  0.        ,   0.        ,   1.        ]]), np.array([0., 0., 0., 0., 0.])), (np.array([[604.33380127,   0.        , 318.17504883],
        #     [  0.        , 604.43933105, 236.77078247],
        #     [  0.        ,   0.        ,   1.        ]]), np.array([0., 0., 0., 0., 0.]))]

        # pose0 = sophus.sophuspy.SE3([[  0.199831422995461,  -0.697761551652838,   0.687892592937759,  -0.469065082957429],
        #     [ -0.979718350894256,  -0.131676778980289,   0.151040321763476,  -0.254678342293666],
        #     [-0.0148106483533835,  -0.704123599173043,  -0.709922954821825,   0.484499852337606],
        #     [                  0,                   0,                   0,                   1]])
        # pose1 = sophus.sophuspy.SE3([[ 0.678207011839507,  0.301150595675989, -0.670330939026183,   0.50898459605032],
        #     [ 0.733688043639065,  -0.22574785893923,  0.640889817991701, -0.562057208349386],
        #     [0.0416785761849928, -0.926469763623374, -0.374054372222597,  0.320794266987536],
        #     [                 0,                  0,                  0,                  1]])
        # pose2 = sophus.sophuspy.SE3([[ -0.630371499310977,   0.565429557757062,  -0.531903363470362,   0.420464780924452],
        #     [   0.77475983335861,   0.415186761686536,  -0.476830319436995,   0.132501510645652],
        #     [-0.0487747216349831,  -0.712677604625612,  -0.699794011402305,   0.504744933970798],
        #     [                  0,                   0,                   0,                   1]])

        intrinsics = [
            (np.array([[616.96191406,   0.        , 318.91717529],
                       [  0.        , 617.26654053, 243.1308136 ],
                       [  0.        ,   0.        ,   1.        ]]),
             np.array([0., 0., 0., 0., 0.])),
            (np.array([[614.80383301,   0.        , 321.73431396],
                       [  0.        , 615.15679932, 242.98834229],
                       [  0.        ,   0.        ,   1.        ]]),
             np.array([0., 0., 0., 0., 0.]))]
        # pose0 = sophus.sophuspy.SE3([[  0.885267818141362, -0.0167997686968407,  -0.464778073853496,   0.689327208031845],
        # [ 0.0411782291479955,  -0.992591274395227,   0.114310609475653,  -0.140593996436941],
        # [ -0.463255052435986,  -0.120334241869024,  -0.878017327064984,    1.70812702535832],
        # [                  0,                   0,                   0,                   1]])
        # pose1 = sophus.sophuspy.SE3([[  0.918972931906929,  0.0189001028738159,   0.393867409839582,  -0.480567800273784],
        # [-0.0315078758974673,  -0.992137331550996,    0.12112295034096,  -0.136542268366384],
        # [   0.39305979720497,  -0.123718638265332,  -0.911151850333938,    1.76530054574554],
        # [                  0,                   0,                   0,                   1]])


        pose0 = sophus.sophuspy.SE3([[  0.986714834423933, -0.0171308808680262,  -0.161556084529294,    0.56948910605487],
            [ 0.0125455179494032,   -0.98342122036788,   0.180901391121048,  -0.113595089039536],
            [  -0.16197668198581,  -0.180524890945361,  -0.970141390850856,    1.47913493176081],
            [                  0,                   0,                   0,                   1]])
        pose1 = sophus.sophuspy.SE3([[  0.949794139515992,  0.0356570910626084,  -0.310837038330422,   0.548160344785772],
            [ 0.0549250903255342,  -0.997058654855102,   0.053453467906828,  0.0192671952660912],
            [ -0.308016764144109, -0.0678425429615314,  -0.948958936082434,    1.62354430916394],
            [                  0,                   0,                   0,                   1]])

        # extrinsics = [pose0, pose1, pose2]
        extrinsics = [pose0, pose1]

    # imgs_ti = []
    # for i in range(3):
    #     filename = "/home/franka/dev/kathyz/cam_track/cam{}_t0.jpeg".format(i)
    #     img = cv2.imread(filename)
    #     imgs_ti.append(img)

    for i in range(len(cameras)):
        matrix, dist_coeffs = intrinsics[i]
        cameras[i].set_intrinsics(matrix=matrix, dist_coeffs=dist_coeffs)
        scene.cameras[i] = scene.cameras[i]._replace(pose=extrinsics[i])

    markers_pos = {}
    for m in marker_ids:
        markers_pos[m] = np.array([0, 0, 0])
    markers = scene.detect_markers(imgs_ti)
    for marker in markers:
        if marker.id in marker_ids:
            pos, quat = ax.utils.se3_to_xyz_quat(marker.pose)
            markers_pos[marker.id] = pos

    # img = scene.cameras[0].module.render_markers(imgs_ti[0])
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', img)
    # cv2.waitKey(1)
    # # import pdb; pdb.set_trace()

    # print("marker pos: ", markers_pos[4])

    # robo_xyz = np.array([0.46149745, 0.10199627, 0.66039947])
    # tag_xyz = np.array([-0.0431918 ,  0.02292705, -0.04471399])
    # xt, yt, zt = tag_xyz
    # rotated_coor = np.array([-zt, -xt, yt])
    # offset = robo_xyz - rotated_coor


    # def coor_transform(tag_frame_pos):
    #     xt, yt, zt = tag_frame_pos
    #     x, y, z = -zt, -xt, yt
    #     franka_frame_pos = np.array([x, y, z]) + offset
    #     return franka_frame_pos

    # # print("original pos: ", markers_pos[5], "  marker_pos_robo_frame: ", coor_transform(markers_pos[5]))

    # time.sleep(0.1)


    return markers_pos
