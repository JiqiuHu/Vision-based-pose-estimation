import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from ultilities import camera_to_object, rotation_matrix_to_quaternion, zoomin
import numpy as np
from keypoint import extract_patches_from_mask,patch_centers
from config import config
import cv2




# TODO build segmentation dataset
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, device = 'cpu'):
        """
        :param roor_dir: root path of folder including images and masks
        :param transform: pre-process of images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []
        self.mask_path = []
        self.device = device

        # Iterate all folders to read images adn masks
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted(os.listdir(folder_path))
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.png'):
                        self.image_path.append(file_path)
                    elif file_name.endswith('.npz'):
                        self.mask_path.append(file_path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        image = Image.open(img_path).convert("RGB")
        # mask = Image.open(mask_path).convert("L")
        data = np.load(mask_path)
        mask = data['segmentation_masks']

        if self.transform:
            image = self.transform(image)

        # Ensure mask is in the same format as expected by the model
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image.to(self.device), mask.to(self.device)


#TODO build pose estimation dataset
class PoseEstimationDataset(Dataset):
    def __init__(self, root_dir, extrinsic_mat, transform=None, device = 'cpu'):
        '''
        :param roor_dir: root path of folder including images and masks
        :param extrinsic_mat: camera extrinsic matrix
        :param transform: pre-process of images
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []
        self.mask_path = []
        self.pose_path = []
        self.extrinsic_mat = np.array(config['extrinsic_mat'])
        self.device = device
        # Iterate all folders to read images adn masks
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted(os.listdir(folder_path))
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.png'):
                        self.image_path.append(file_path)
                    elif file_name.endswith('.npz'):
                        self.mask_path.append(file_path)
                        self.pose_path.append(file_path)

        if len(self.mask_path) != len(self.pose_path) != len(self.image_path):
            raise ValueError("Sorry, the length of dataset dose not be matched")


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        pose_path = self.pose_path[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        data = np.load(pose_path)
        object_poses = data['object_poses']  #'Camera', 'grasper', 'scissor', 'Sun', 'Sun.001'
        mask = data['segmentation_masks']
        image_cropped, cropped_mask, abs_center = zoomin(image, mask)

        label, pose = object_poses[1]
        #Transfer label string to number, so that we can transfer label to tensor
        if label == 'grasper':
            label_ = 1
        elif label == 'scissor':
            label_ = 2
        else:
            raise ValueError("Sorry, the label of dataset is wrong. It should be either 'grasper' or 'scissor' ")

        if self.transform:
            image_cropped = self.transform(image_cropped)

        # Ensure mask is in the same format as expected by the model
        label_= torch.tensor(np.array(label_), dtype=torch.int8).to(self.device)
        pose = np.array(pose)
        # pose = torch.tensor(np.array(pose), dtype=torch.float32).to(self.device) #torch.Size([4,4])

        # Convert pose wrt world coordinate to camera coordinate
        r, t = camera_to_object(self.extrinsic_mat, pose)  # Size([3, 3]) Size([3])
        r = torch.tensor(r, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        # Convert rotation matrix to a quaternion
        rq = rotation_matrix_to_quaternion(r)  # torch.Size([4])

        return (image_cropped.to(self.device), label_.to(self.device), r.to(self.device),
                rq.to(self.device), t.to(self.device))


#TODO build pose estimation dataset
class Pose_keypointDataset(Dataset):
    def __init__(self, root_dir, points, transform=None,  device = 'cpu'):
        '''
        :param roor_dir: root path of folder including images and masks
        :param extrinsic_mat: camera extrinsic matrix
        :param transform: pre-process of images
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.points_3d = points
        self.image_path = []
        self.mask_path = []
        self.pose_path = []
        self.patch_size = config['patch_size']
        self.intrinsic_mat = np.array(config['intrinsic_mat'])
        self.extrinsic_mat = np.array(config['extrinsic_mat'])
        self.dist_coeffs = np.array(config['dist_coeffs']).T
        self.device = device
        # Iterate all folders to read images adn masks
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted(os.listdir(folder_path))
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.png'):
                        self.image_path.append(file_path)
                    elif file_name.endswith('.npz'):
                        self.mask_path.append(file_path)
                        self.pose_path.append(file_path)

        if len(self.mask_path) != len(self.pose_path) != len(self.image_path):
            raise ValueError("Sorry, the length of dataset dose not be matched")


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        pose_path = self.pose_path[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        data = np.load(pose_path)
        object_poses = data['object_poses']  # 'Camera', 'grasper', 'scissor', 'Sun', 'Sun.001'
        mask = data['segmentation_masks']

        # Crop image and mask
        image_cropped1, cropped_mask, abs_center = zoomin(image, mask)
        # fig, ax = plt.subplots()
        # ax.imshow(image_cropped)
        # for y in range(0,608, 38):
        #     ax.axhline(y, color = 'green', linewidth = 1)
        # for x in range(0,608, 38):
        #     ax.axvline(x, color = 'green', linewidth = 1)
        # plt.show()

        # Calculate scores for each maks patch
        _, score = extract_patches_from_mask(cropped_mask, self.patch_size)
        score = torch.tensor(score, dtype = torch.float32)

        # Calculate patch's centroid wrt original image coordinate
        patch_center = patch_centers(abs_center, self.patch_size).clone().detach()
        # patch_center = torch.tensor(patch_center, dtype = torch.float32).clone().detach().requires_grad(True)

        label, pose = object_poses[1]
        # Transfer label string to number, so that we can transfer label to tensor
        if label == 'grasper':
            label_ = 1
        elif label == 'scissor':
            label_ = 2
        else:
            raise ValueError("Sorry, the label of dataset is wrong. It should be either 'grasper' or 'scissor' ")

        if self.transform:
            image_cropped = self.transform(image_cropped1)

        # Ensure mask is in the same format as expected by the model
        # label_ = torch.tensor(np.array(label_), dtype=torch.int32).to(self.device)
        pose = np.array(pose) # torch.Size([4,4])

        # Convert pose wrt world coordinate to camera coordinate
        r, t = camera_to_object(self.extrinsic_mat, pose)  # Size([3, 3]) Size([3])
        rvec, _ = cv2.Rodrigues(r)
        true_keypoints, _ = cv2.projectPoints(self.points_3d, rvec, t, self.intrinsic_mat, self.dist_coeffs)
        true_keypoints = torch.tensor(true_keypoints, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        # print(img_path, pose_path)
        return (image, image_cropped1, image_cropped.to(self.device), score.to(self.device), patch_center.to(self.device),
                true_keypoints.to(self.device), r.to(self.device), t.to(self.device))


class Pose_keypointDataset2(Dataset):
    def __init__(self, root_dir, points, transform=None,  device = 'cpu'):
        '''
        :param roor_dir: root path of folder including images and masks
        :param extrinsic_mat: camera extrinsic matrix
        :param transform: pre-process of images
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.points_3d = points
        self.image_path = []
        self.mask_path = []
        self.pose_path = []
        self.patch_size = 38
        self.intrinsic_mat = np.array(config['intrinsic_mat'])
        self.extrinsic_mat = np.array(config['extrinsic_mat'])
        self.dist_coeffs = np.array(config['dist_coeffs']).T
        self.device = device
        # Iterate all folders to read images adn masks
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted(os.listdir(folder_path))
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.png'):
                        self.image_path.append(file_path)
                    elif file_name.endswith('.npz'):
                        self.mask_path.append(file_path)
                        self.pose_path.append(file_path)

        if len(self.mask_path) != len(self.pose_path) != len(self.image_path):
            raise ValueError("Sorry, the length of dataset dose not be matched")


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        pose_path = self.pose_path[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        data = np.load(pose_path)
        object_poses = data['object_poses']  # 'Camera', 'grasper', 'scissor', 'Sun', 'Sun.001'
        mask = data['segmentation_masks']

        # Crop image and mask
        image_cropped, cropped_mask, abs_center = zoomin(image, mask)
        # fig, ax = plt.subplots()
        # ax.imshow(image_cropped)
        # for y in range(0,608, 38):
        #     ax.axhline(y, color = 'green', linewidth = 1)
        # for x in range(0,608, 38):
        #     ax.axvline(x, color = 'green', linewidth = 1)
        # plt.show()

        # Calculate scores for each maks patch
        _, score = extract_patches_from_mask(cropped_mask, self.patch_size)
        score = torch.tensor(score, dtype = torch.float32)

        # Calculate patch's centroid wrt original image coordinate
        patch_center = patch_centers(abs_center, self.patch_size).clone().detach()
        # patch_center = torch.tensor(patch_center, dtype = torch.float32).clone().detach().requires_grad(True)

        label, pose = object_poses[1]
        # Transfer label string to number, so that we can transfer label to tensor
        if label == 'grasper':
            label_ = 1
        elif label == 'scissor':
            label_ = 2
        else:
            raise ValueError("Sorry, the label of dataset is wrong. It should be either 'grasper' or 'scissor' ")

        if self.transform:
            image_cropped = self.transform(image_cropped)

        # Ensure mask is in the same format as expected by the model
        # label_ = torch.tensor(np.array(label_), dtype=torch.int32).to(self.device)
        pose = np.array(pose) # torch.Size([4,4])

        # Convert pose wrt world coordinate to camera coordinate
        r, t = camera_to_object(self.extrinsic_mat, pose)  # Size([3, 3]) Size([3])
        rvec, _ = cv2.Rodrigues(r)
        true_keypoints, _ = cv2.projectPoints(self.points_3d, rvec, t, self.intrinsic_mat, self.dist_coeffs)
        true_keypoints = torch.tensor(true_keypoints, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        # print(img_path, pose_path)
        return (image, image_cropped.to(self.device), score.to(self.device), patch_center.to(self.device),
                true_keypoints.to(self.device), r.to(self.device), t.to(self.device))



if __name__=='__main__':
    #multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # root_dir1 = '/vol/bitbucket/jh523/dataset/Training data/Training data/Segmentation'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # data_seg = SegmentationDataset(root_dir1, transform, device)
    # train_loader = DataLoader(data_seg, batch_size=5, shuffle=True)
    # for images, masks in train_loader:
    #     print(images.shape, masks.shape) #torch.Size([5, 3, 1080, 1920]) torch.Size([5, 1080, 1920])
    #     break

    root_dir2 = '/vol/bitbucket/jh523/dataset/Training data/Pose estimation/Grasper'
    # data = np.load('Training data/Pose estimation/Grasper/001/0001.npz')
    # intrinsic_mat = torch.from_numpy(data['intrinsic_mat'])
    # extrinsic_mat = torch.tensor(data['extrinsic_mat'],dtype=torch.float32).to(device)
    # print(intrinsic_mat, extrinsic_mat)
    points = np.load('grasper_FPS8.npy')
    data_pose = Pose_keypointDataset(root_dir2, points, transform, device)
    train_loader_pose = DataLoader(data_pose, batch_size=12, shuffle=True)
    for image, score, centroid, keypoints, r, t in train_loader_pose:
        print(score.shape)
        #torch.Size([12, 256])
        break
        # print(image.shape, score.shape, keypoints.shape, centroid.shape, r.shape, t.shape)
        # #-Size([1, 3, 608, 608]), Size([1, 256],) Size([1, 8, 1, 2]), Size([1, 3, 3]), Size([1, 3]), Size([1])
        # print(centroid)
        # break

