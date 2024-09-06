from keypoint import pnp
from visualization import visualize_segmentation
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from pose2 import Pose2
from config import config
import os
from key_model import ViTKeypointModel
from keypoint import farthest_point_sampling, select_top_points
import cv2
import matplotlib.pyplot as plt
from ultilities import *

def test_seg(model_path, image_path, ground_truth):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=10, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    model = load()
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints['model'])
    model.eval()
    with torch.no_grad():
        output = segment_image(image_path, model, preprocess)
        visualize_segmentation(image, output, ground_truth)

def seg(model_path, image_path):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0, contrast=5, saturation=0.1, hue=(-0.4,0)),
        transforms.GaussianBlur(kernel_size=5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    model = load()
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints['model'])
    model.eval()
    with torch.no_grad():
        output = segment_image(image_path, model, preprocess)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(output.detach().cpu().numpy().argmax(0))
        plt.axis('off')
        plt.title('Segmentation Mask')

        plt.tight_layout()
        plt.show()


def test_pose_vit(test_data, model1, model2, kp_3d, evaluator):
    model1.eval()
    model2.eval()

    dist_coeffs = np.array(config['dist_coeffs'])
    iteration = 25
    # # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (inputs, score, centroid, true_keypoints, r, t) in enumerate(test_data):
            # Initial Prediction for grasper
            print('{}/{}'.format(i, len(test_data)))
            pre_keypoints, pre_conf = model1(inputs, centroid)
            pre_r, pre_t = pnp(kp_3d, pre_keypoints, pre_conf)
            r_pre_vit = np.squeeze(pre_r, 0)
            t_pre_vit = np.squeeze(pre_t, 0).reshape(-1, 1)

            r_pre, t_pre = model2(inputs, iteration)
            vr_pre_dic = torch.squeeze(r_pre, 0).cpu().numpy()
            t_pre_dic = torch.squeeze(t_pre, 0).cpu().numpy().reshape(-1, 1)

            r = torch.squeeze(r, 0).cpu().numpy()
            t = torch.squeeze(t, 0).cpu().numpy().reshape(-1, 1)
            pre = np.concatenate([r_pre_vit, t_pre_dic], axis=-1)
            gro = np.concatenate([r, t], axis=-1)

            evaluator.evaluate(pre, gro)
        evaluator.summarize('./vit_result1.npy')

def sample_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('Error: Could not open video.')
        exit()

    frame_count = 0
    max_frames = 240
    frames = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

    for idx, frame in enumerate(frames):
        up_sample = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'frames/frame_{idx}.png', up_sample)

    print(f'Sample Completed, totol number is {len(frames)}')


def track(path):
    trajectory = []
    camera_matrix = np.array(config['intrinsic_mat'])
    extrinsic_mat = np.array(config['extrinsic_mat'])
    dist_coeffs = np.array(config['dist_coeffs'])
    image_files = sorted([os.path.join(path, img) for img in os.listdir(path) if img.endswith('.png')])
    pose_files = sorted([os.path.join(path, img) for img in os.listdir(path) if img.endswith('.npz')])
    for i in range(len(image_files)):
        img_file = image_files[i]
        data = np.load(pose_files[i])
        print(img_file, pose_files[i])
        object_poses = data['object_poses']
        label, pose = object_poses[1]
        r, tvec = camera_to_object(extrinsic_mat, pose)  # Size([3, 3]) Size([3])
        rvec, _ = cv2.Rodrigues(r)
        #
        # image = Image.open(img_file).convert('RGB')
        # image = np.array(image)
        #
        # axis_length = 0.05  # 坐标轴长度
        # axis_points_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(
        #     -1, 3)
        # axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        #
        # origin = tuple(axis_points_2d[0].ravel().astype(int))
        # cv2.circle(image, origin, 5, (255, 255, 0), -1)
        # image = cv2.line(image, origin, tuple(axis_points_2d[1].ravel().astype(int)), (0, 0, 255), 2,
        #                  cv2.LINE_AA)  # X轴 蓝色
        # image = cv2.line(image, origin, tuple(axis_points_2d[2].ravel().astype(int)), (255, 0, 0), 2,
        #                  cv2.LINE_AA)  # Y轴 红色
        # image = cv2.line(image, origin, tuple(axis_points_2d[3].ravel().astype(int)), (0, 255, 0), 2,
        #                  cv2.LINE_AA)  # Z轴 绿色
        #
        # # 显示图像
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'track/frame_{i}.png', image)
        trajectory.append(tvec)


    trajectory = np.array(trajectory).reshape(-1, 3)
    if len(trajectory) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label = 'Object Trajectory', color = 'r', marker = 'o')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Object trajectory')
        ax.legend()
        plt.show()


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

if __name__=='__main__':
    # imagepath = '/vol/bitbucket/jh523/dataset/Training data/Segmentation/002/0150.png'
    img_path = '/homes/jh523/PycharmProjects/pythonProject/frames/frame_120.png'
    image = Image.open(img_path).convert("RGB")
    image = np.array(image)
    plt.imshow(image)
    plt.show()
    result = image[500:1108, 450:1058]
    center = np.array(500 + (1108-500) //2 , 450 +(1058-450)//2)

    plt.imshow(result)
    plt.show()
    # ima = cv2.imread(imagepath)
    # result = apply_brightness_contrast(ima, -3, -8)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # mask = np.load('/vol/bitbucket/jh523/dataset/Training data/Segmentation/002/0150.npz')['segmentation_masks']
    # model_path = '/vol/bitbucket/jh523/checkpoints/seg/100_epoch_seg'
    # # test_seg(model_path, imagepath, mask)
    # seg(model_path, imagepath)
    # track('/vol/bitbucket/jh523/dataset/Training data/Pose estimation/Grasper/test')






