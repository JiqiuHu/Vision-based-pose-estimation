import numpy as np
from scipy import spatial
import math
import torch
from torchvision import transforms
from config import config
from torch.utils.data import Dataset,DataLoader
from pose_estimator import Pose
from keypoint import pnp
from pose2 import Pose2
from ultilities import extract_points_from_stl, quaternion_to_rotation_matrix
from dataset import PoseEstimationDataset, Pose_keypointDataset2, Pose_keypointDataset
from key_model import ViTKeypointModel
from keypoint import farthest_point_sampling, select_top_points


def isRotationMatrix(R):
    # TODO check if it is a valid rotation matrix
    """
    :param R: 3x3 ndarray rotation matrix
    :return: True or False
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 10000 * 1e-6


# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    # TODO convert the rotation matrix to euler angles
    '''
    :param R: 3x3 ndarray rotation matrix
    :return: Corresponding angles in degree.
    '''
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # return np.array([x, y, z])
    return np.rad2deg([x, y, z]) #convert angles from radians to degrees


def re(R_est, R_gt):
    # TODO calculate rotational Error
    """
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error (in degree).
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))  #convert angles from radians to degrees

    return rd_deg


def project(xyz, K, RT):
    # TODO project 3D points to 2D image pixel
    """
    xyz: [N, 3], coordinates of 3D points
    K: [3, 3], intrinsic matrix of camera
    RT: [3, 4], pose
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_model_diameter(model):
    # TODO Get the diameter of 3d model from .npy/.stl files
    '''
    :param model:
    :return: diameter of the model
    '''
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])

    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    diameter = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return diameter


def model_diameter(instrument_type):
    model_diameter = {'Grasper': 1.6167998441428093, 'Scissor': 1.946739544331231}
    return model_diameter[instrument_type]



class Evaluator:

    def __init__(self, model, instrument_type):
        self.model = model
        self.diameter = model_diameter(instrument_type)
        self.K = np.array(config['intrinsic_mat']).reshape(3, 3)

        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.rotation = []
        self.translation = []
        self.add_dist = []
        self.trans_error = []
        self.rot_error = []
        self.adds = []
        self.adds_dist = []
        self.proj2d5 = []
        self.proj2d10 = []
        self.proj2d15 = []
        self.proj2d20 = []

    def trans_rot_error(self, pose_pred, pose_targets):
        gt_pose_rot = pose_targets[:3, :3]
        gt_pose_trans = pose_targets[:3, -1]
        pred_pose_rot = pose_pred[:3, :3]
        pred_pose_trans = pose_pred[:3, -1]

        trans_error = np.linalg.norm(gt_pose_trans - pred_pose_trans)
        rot_error = re(pred_pose_rot, gt_pose_rot)
        self.trans_error.append(trans_error)
        self.rot_error.append(rot_error)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        print('2d projections error: {}'.format(proj_mean_diff))

        self.proj2d.append(proj_mean_diff)
        self.proj2d5.append(proj_mean_diff < 5)
        self.proj2d10.append(proj_mean_diff < 10)
        self.proj2d15.append(proj_mean_diff < 15)
        self.proj2d20.append(proj_mean_diff < 20)

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        add_error = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        adds_error_index = spatial.cKDTree(model_pred)
        adds_error, _ = adds_error_index.query(model_targets, k=1)
        adds_error = np.mean(adds_error)

        self.add_dist.append(add_error)
        self.add.append(add_error < diameter)
        self.adds_dist.append(adds_error)
        self.adds.append(adds_error < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3])* 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.rotation.append(angular_distance < 5)
        self.translation.append(translation_distance < 5)
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def evaluate(self, pose_gt, pose_pred):
        self.projection_2d(pose_pred, pose_gt, self.K)
        self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.trans_rot_error(pose_pred, pose_gt)

    def summarize(self, save_path=None):
        proj2d = np.mean(self.proj2d)
        proj5 = np.mean(self.proj2d5)
        proj10 = np.mean(self.proj2d10)
        proj15 = np.mean(self.proj2d15)
        proj20 = np.mean(self.proj2d20)
        add = np.mean(self.add)
        add_dists = self.add_dist
        add_dist = np.mean(self.add_dist)
        adds = np.mean(self.adds)
        adds_dist = np.mean(self.adds_dist)
        cmd5 = np.mean(self.cmd5)
        cm5 = np.mean(self.translation)
        degree5 = np.mean(self.rotation)
        trans_error = np.mean(self.trans_error)
        rot_error = np.mean(self.rot_error)

        print('2d projections: {}'.format(proj2d))
        print('2d projections 5 metric: {}'.format(proj5))
        print('2d projections 10 metric: {}'.format(proj10))
        print('2d projections 15 metric: {}'.format(proj15))
        print('2d projections 20 metric: {}'.format(proj20))
        print('ADD metric: {}'.format(add))
        print('ADD mean distance', add_dist)
        print('ADD-S metric: {}'.format(adds))
        print('ADD-S mean distance', adds_dist)
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('5 cm metric: {}'.format(cm5))
        print('5 degree metric: {}'.format(degree5))
        print('trans_error: {}'.format(trans_error))
        print('rot_error: {}'.format(rot_error))

        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.icp_add = []
        self.trans_error = []
        self.rot_error = []
        results = {'proj2d': proj2d, 'proj5':proj5, 'proj10':proj10, 'proj15':proj15, 'proj20':proj20, 'add': add, 'add-s': adds, 'ADD-distance': add_dist,
                   'ADD-distance list': add_dists, 'ADDS-distance': adds_dist, 'cmd5': cmd5, '5cm': cm5, '5 degree': degree5,
                   'trans_error': trans_error, 'rot_error': rot_error}
        if save_path is not None:
            np.save(save_path, results)
        return {'proj2d': proj2d, 'proj5':proj5, 'proj10':proj10, 'proj15':proj15, 'proj20':proj20, 'add': add, 'add-s': adds, 'ADD-distance': add_dist,
                   'ADD-distance list': add_dists, 'ADDS-distance': adds_dist, 'cmd5': cmd5, '5cm': cm5, '5 degree': degree5,
                   'trans_error': trans_error, 'rot_error': rot_error}



def test_pose_vit(test_data, model, kp_3d, evaluator):
    model.eval()
    intrinsic = np.array(config['intrinsic_mat']).reshape(3,3)
    dist_coeffs = np.array(config['dist_coeffs'])
    # # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (image,  inputs, score, centroid, true_keypoints, r, t) in enumerate(test_data):
            # Initial Prediction for grasper
            print('{}/{}'.format(i, len(test_data)))
            pre_keypoints, pre_conf = model(inputs, centroid)
            pre_r, pre_t = pnp(kp_3d, pre_keypoints, pre_conf, score)
            r_pre_vit = np.squeeze(pre_r, 0)
            t_pre_vit = np.squeeze(pre_t, 0).reshape(-1, 1)

            r = torch.squeeze(r, 0).cpu().numpy()
            t = torch.squeeze(t, 0).cpu().numpy().reshape(-1, 1)
            pre = np.concatenate([r_pre_vit, t_pre_vit], axis=-1)
            gro = np.concatenate([r, t], axis=-1)

            evaluator.evaluate(pre, gro)
        evaluator.summarize('./vit_result1.npy')


def test_pose_dic(test_data, model, evaliation):
    iteration = 8
    model.eval()
    with torch.no_grad():
        for i, (inputs, label, r, t) in enumerate(test_data):
            print('{}/{}'.format(i, len(test_data)))
            r_pre, t_pre = model(inputs, iteration)
            # torch.Size([batch, 3, 3])
            # vr_pre = quaternion_to_rotation_matrix(rq_pre)
            vr_pre = torch.squeeze(r_pre, 0).cpu().numpy()
            t_pre = torch.squeeze(t_pre, 0).cpu().numpy().reshape(-1, 1)
            r = torch.squeeze(r, 0).cpu().numpy()
            t = torch.squeeze(t, 0).cpu().numpy().reshape(-1, 1)
            pre = np.concatenate([vr_pre, t_pre], axis=-1)
            gro = np.concatenate([r, t], axis=-1)
            evaliation.evaluate(gro, pre)
        evaliation.summarize('./direct_pose2_result.npy')


def test_dic_vit(test_data,  model1, model2, kp_3d, evl1, evl2):
    iteration = 8
    model1.eval()
    model2.eval()
    intrinsic = np.array(config['intrinsic_mat']).reshape(3,3)
    dist_coeffs = np.array(config['dist_coeffs'])
    with torch.no_grad():
        for i, (image, img_crop, inputs, score, centroid, true_keypoints, r, t) in enumerate(test_data):
            # Initial Prediction for grasper
            print('{}/{}'.format(i, len(test_data)))
            pre_keypoints, pre_conf = model1(inputs, centroid)
            pre_r, pre_t = pnp(kp_3d, pre_keypoints, pre_conf, score)
            r_pre_vit = np.squeeze(pre_r, 0)
            t_pre_vit = np.squeeze(pre_t, 0).reshape(-1,1)

            r_pre, t_pre = model2(inputs, iteration)
            # torch.Size([batch, 3, 3])
            # vr_pre = quaternion_to_rotation_matrix(rq_pre)
            r_pre_dic = torch.squeeze(r_pre, 0).cpu().numpy()
            t_pre_dic = torch.squeeze(t_pre, 0).cpu().numpy().reshape(-1,1)

            r = torch.squeeze(r, 0).cpu().numpy()
            t = torch.squeeze(t, 0).cpu().numpy().reshape(-1,1)

            pre_vit = np.concatenate([r_pre_vit, t_pre_vit], axis=-1)
            pre_dic = np.concatenate([r_pre_dic, t_pre_dic], axis=-1)
            gro = np.concatenate([r, t], axis=-1)

            evl1.evaluate(gro, pre_vit)
            evl2.evaluate(gro, pre_dic)
        evl1.summarize('./vit_pose2_result.npy')
        evl2.summarize('./dic_pose2_result.npy')




if __name__=='__main__':
    points_g = extract_points_from_stl('/vol/bitbucket/jh523/dataset/Training data/grasper.stl')
    num_points = 50
    points_3d = select_top_points(points_g, num_points, vertical_axis=0)

    k_gra = np.array(np.load('grasperCUT_FPS8.npy'), dtype=np.float64)
    keypoints_3d = torch.tensor(k_gra, dtype=torch.float32)
    
    # points_s = extract_points_from_stl('/vol/bitbucket/jh523/dataset/Training data/scissor.stl')
    # num_points = 1000
    # keypoints_3d = farthest_point_sampling(points_g, num_points)

    evaliation1 = Evaluator(points_3d,'Grasper')
    evaliation2 = Evaluator(points_3d,'Grasper')

    root_dir = '/vol/bitbucket/jh523/dataset/Training data/Pose estimation/Grasper'
    extrinsic_mat = np.array(config['extrinsic_mat'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    generator = torch.Generator()
    generator.manual_seed(0)
    full_data = Pose_keypointDataset2(root_dir, k_gra, transform)
    train_size = len(full_data)
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size], generator=generator)

    test_data = DataLoader(train_data, batch_size=1, shuffle=True)
    

    # model_dic = Pose2()#
    # checkpoints = torch.load('/vol/bitbucket/jh523/checkpoints/cnn/120_epoch_', weights_only=True)
    # model_dic.load_state_dict(checkpoints['model'])

    model_vit = ViTKeypointModel()
    checkpoints = torch.load('/vol/bitbucket/jh523/checkpoints/vit_frozen6/300_epoch_vit4', weights_only=True)
    model_vit.load_state_dict(checkpoints['model'])


    # test_dic_vit(test_data, model_vit, model_dic, keypoints_3d, evaliation1, evaliation2)
    test_pose_vit(test_data, model_vit, keypoints_3d, evaliation1)