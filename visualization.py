import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import cv2
import numpy as np
from config import config
import seaborn as sns
from ultilities import camera_to_object, zoomin, extract_points_from_stl
from PIL import Image
from key_model import ViTKeypointModel
from keypoint import *
import torch
from torchvision import transforms
from evaluation import re
from torch.utils.data import Dataset,DataLoader
from dataset import PoseEstimationDataset, Pose_keypointDataset, Pose_keypointDataset2


#TODO visualize semantic segmentation result
def visualize_segmentation(image, segmentation, ground_truth):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation.detach().cpu().numpy().argmax(0))
    plt.axis('off')
    plt.title('Segmentation Mask')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth)
    plt.axis('off')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation.detach().cpu().numpy().argmax(0))
    plt.axis('off')
    plt.title('Segmentation Mask')

    plt.tight_layout()
    plt.show()


def show_points(stl_path):
    # Create a new plot
    figure = pyplot.figure()
    axes = figure.add_subplot(projection='3d')

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(stl_path)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()


def show_pose(image, keypoints_3d,  r, tvec,  pre_r, pre_t, camera_matrix, dist_coeffs = None):
    # Rotation matrix and translatoon vector
    rvec, _ = cv2.Rodrigues(r)
    r_pre,_ = cv2.Rodrigues(pre_r)

    # project 3D points to 2D
    keypoints_2d, _ = cv2.projectPoints(keypoints_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pre_keypoints, _ = cv2.projectPoints(keypoints_3d, r_pre, pre_t, camera_matrix, dist_coeffs)
    print('trans error:', np.linalg.norm((t-pre_t)))
    print('rotation error:', re(pre_r, r))
    # print('ground truth:', keypoints_2d)
    # print('prediction:', pre_keypoints)
    print('projection error:',np.mean(np.linalg.norm(keypoints_2d - pre_keypoints, axis=-1)))


    # Plot 2D points
    for i, point in enumerate(keypoints_2d):
        point = tuple(point.ravel().astype(int))  # 转换为整型坐标
        cv2.circle(image, point, 5, (0, 0, 255), -1)  # 在图像上绘制一个小圆点
        cv2.putText(image, f'({keypoints_3d[i][0]}, {keypoints_3d[i][1]}, {keypoints_3d[i][2]})',
                    (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, point in enumerate(pre_keypoints):
        point = tuple(point.ravel().astype(int))  # 转换为整型坐标
        cv2.circle(image, point, 5, (0, 255, 0), -1)

    # Plot axis
    axis_length = 0.05  # length of axis
    axis_points_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1,3)
    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    axis_points_2d_pre, _ = cv2.projectPoints(axis_points_3d, r_pre, pre_t, camera_matrix, dist_coeffs)


    origin = tuple(axis_points_2d[0].ravel().astype(int))
    origin2 = tuple(axis_points_2d_pre[0].ravel().astype(int))

    cv2.circle(image, origin, 5, (255, 255, 0), -1)
    image = cv2.line(image, origin, tuple(axis_points_2d[1].ravel().astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    image = cv2.line(image, origin, tuple(axis_points_2d[2].ravel().astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    image = cv2.line(image, origin, tuple(axis_points_2d[3].ravel().astype(int)), (0, 0, 255), 2, cv2.LINE_AA)

    image = cv2.line(image, origin2, tuple(axis_points_2d_pre[1].ravel().astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    image = cv2.line(image, origin2, tuple(axis_points_2d_pre[2].ravel().astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    image = cv2.line(image, origin2, tuple(axis_points_2d_pre[3].ravel().astype(int)), (0, 255, 0), 2, cv2.LINE_AA)

    # show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Projected Keypoints and Coordinate Axes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def cosine_similarity(x,y, norm=False):
    assert len(x) == len(y), 'len(x) != len(y)'
    xy = x.dot(y)
    x2y2 = np.linalg.norm(x, ord=2) * np.linalg.norm(x, ord = 2)
    sim = xy / x2y2
    return sim


def show_pos(pos):
    num_patch = 16 * 16
    cos = np.zeros((num_patch, num_patch))
    for i in range(num_patch):
        for j in range(num_patch):
            cos[i,j] = cosine_similarity(pos[i, :], pos[j, :])
    fig, axs = plt.subplots(nrows=16, ncols=16, figsize = (16, 16))
    i = 0
    cos = cos.reshape(num_patch, 16, 16)
    for ax in axs.flat:
        ax.imshow(cos[i, :, :], cmap = 'viridis')
        i +=1
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('position.png')
    # plt.show()


def show_attention(a):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize = (12, 9))
    print(a.shape)
    for i in range(3):
        for j in range(4):
            a1 = a[3*i + j, :, :]
            b = (a1 - a1.min())/ (a1.max() - a1.min())
            ax[i][j].imshow(b, cmap='viridis')
    plt.show()

def attention(a, image, i):
    a = torch.squeeze(a, 0)
    att1 = torch.mean(a,dim=0)
    att1 = att1[0, 1:].view(16, 16)
    att1 = att1.unsqueeze(0).unsqueeze(0)
    att1 = torch.nn.functional.interpolate(att1, size=(608, 608), mode='bilinear')

    # im = image.permute(0, 2, 3, 1).mul(255).clamp(0, 255)
    # im = np.squeeze(im.cpu().detach().numpy().astype('uint8'),0)
    plt.imshow(image)
    plt.imshow(att1.squeeze().cpu().numpy(), alpha=0.4, cmap='rainbow')
    plt.axis('off')
    plt.savefig('attentions/attention_layer{}.png'.format(i))
    # plt.show()


def all_prediction(predictions, image):
    color = [(0, 0, 255), (128, 0, 0), (0, 128, 0), (128, 128, 128), (255, 192, 203), (238, 130, 238), (106, 90, 205), (230, 230, 250)]
    for j in range(predictions.shape[1]):
        keypoint = predictions[:, j, :]
        c = color[j]
        for i, point in enumerate(keypoint):
            point = tuple(point.ravel().astype(int))
            cv2.circle(image, point, 1, c, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('All Predictions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_prediction(predictions, image):
    color = [(0, 0, 255), (128, 0, 0), (0, 128, 0), (128, 128, 128), (255, 192, 203), (238, 130, 238), (106, 90, 205), (230, 230, 250)]
    for j in range(predictions.shape[0]):
        keypoint = predictions[j, :, :]
        c = color[j]
        for i, point in enumerate(keypoint):
            point = tuple(point.ravel().astype(int))
            cv2.circle(image, point, 5, c, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Selected Predictions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    intrinsic_mat = np.array(config['intrinsic_mat'])
    extrinsic_mat = np.array(config['extrinsic_mat'])
    dist_coeffs = np.array(config['dist_coeffs'])
    k_gra = np.load('grasperCUT_FPS8.npy').reshape(-1, 3)
    key3d = torch.tensor(k_gra, dtype=torch.float32)
    key3d = key3d

    points_g = extract_points_from_stl('/vol/bitbucket/jh523/dataset/Training data/grasper.stl')
    num_points = 200
    points_3d = select_top_points(points_g, num_points, vertical_axis=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_vit = ViTKeypointModel()
    checkpoints = torch.load('/vol/bitbucket/jh523/checkpoints/vit_frozen6/300_epoch_vit4', weights_only=False, map_location='cpu')
    model_vit.load_state_dict(checkpoints['model'])
    model_vit.eval()

    # root_dir = '/vol/bitbucket/jh523/dataset/Training data/Pose estimation/Grasper'
    root_dir = '/homes/jh523/msc_project/Training data/Scissor'
    full_data = Pose_keypointDataset(root_dir, k_gra, transform)
    # train_size = int(0.125 * len(full_data))
    train_size = 5
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

    test_data = DataLoader(train_data, batch_size=1, shuffle=True)
    patch_size = 38

    with torch.no_grad():
        for i, (image, image_crop, inputs, score, centroid, true_keypoints, r, t) in enumerate(test_data):
            pre_keypoints, pre_conf, pos, att = model_vit(inputs, centroid)
            pre_r, pre_t = pnp(key3d, pre_keypoints, pre_conf, score)
            # pos = np.squeeze(pos.cpu().detach().numpy(),0)
            # select_p, s = select(pre_keypoints, pre_conf, score)
            # image = np.squeeze(image.cpu().detach().numpy(), 0)
            # all_prediction(np.squeeze(pre_keypoints.cpu().detach().numpy(),0), image)
            # select_prediction(np.squeeze(s.cpu().detach().numpy(),0), image)
            # attention(att[11], np.squeeze(image_crop.cpu().detach().numpy(),0))
            # out = np.squeeze(out.cpu().detach().numpy(), 0)
            # show_pos(pos)
            # for i in range(12):
            #     attention(att[i], np.squeeze(image_crop.cpu().detach().numpy(),0), i)
            r_pre_vit = np.squeeze(pre_r, 0)
            t_pre_vit = np.squeeze(pre_t, 0).reshape(-1,1)

            key3d = key3d.cpu().detach().numpy()
            r = np.squeeze(r.cpu().detach().numpy(),0)
            t = np.squeeze(t.cpu().detach().numpy(),0).reshape(-1,1)
            image = np.squeeze(image.cpu().detach().numpy(),0)

            show_pose(image, key3d, r, t, r_pre_vit, t_pre_vit, intrinsic_mat, dist_coeffs=dist_coeffs)
            break