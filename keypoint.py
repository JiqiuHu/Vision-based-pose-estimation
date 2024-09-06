import numpy as np
from stl import mesh
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from ultilities import zoomin, camera_to_object
from PIL import Image
from config import config
import cv2


def read_stl(file_path):
    '''
    TODO: Load and read 3D model

    :param file_path: .stl file of 3D model
    :return: 3D points (numpy array) and mesh data
    '''
    mesh_data = mesh.Mesh.from_file(file_path)
    points = np.concatenate((mesh_data.v0, mesh_data.v1, mesh_data.v2))
    return points, mesh_data



def farthest_point_sampling(points, npoint):
    """
    TODO: Perform farthest point sampling to select a subset of points.

    :param points: Input point cloud data, shape (N, 3)
    :param npoint: Number of points to sample
    :return: Sampled points (numpy array), shape (npoint, 3)
    """
    N, D = points.shape
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return points[centroids]


def select_top_points(points, npoint, vertical_axis=0):
    """
    TODO:Select the top points from the point cloud based on the farthest point sampling method.

    :param points: Input point cloud data, shape (N, D)
    :param npoint: Number of points to sample
    :param vertical_axis: The axis index representing vertical direction (default is 0 for x-axis)
    :return: Sampled points based on top (x) direction
    """
    # Compute the centroid of the point cloud
    centroid = np.mean(points, axis=0)

    # Calculate the vertical distance (x-axis) of each point
    vertical_distances = points[:, vertical_axis]

    # Filter points based on the vertical distance, keeping only the top points
    # You may adjust the percentile or threshold to define "top" more precisely
    top_points_mask = vertical_distances >= np.percentile(vertical_distances, 80)  # Keeping top 5% based on x-axis
    top_points = points[top_points_mask]

    if len(top_points) == 0:
        raise ValueError("No points found in the top percentile. Adjust percentile or check data.")

    # Perform farthest point sampling on the top points
    sampled_top_points = farthest_point_sampling(top_points, npoint)

    return sampled_top_points


def plot_points_and_mesh(original_points, sampled_points, mesh_data):
    '''
    TODO: Visulize 3D model and sampled keypoints
    :param original_points: 3D points of model (numpy array)
    :param sampled_points:  sampled keypoints (numpy array)
    :param mesh_data: mesh data of model
    :return: None
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original mesh
    ax.add_collection3d(art3d.Poly3DCollection(mesh_data.vectors, facecolors='cyan', linewidths=0.1, edgecolors='b', alpha=.25))

    # Plot the original points
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], s=3, c='blue', label='Original Points')

    # Plot the sampled points
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=10, c='red', label='Sampled Points')
    # Setting axis limits to zoom in
    ax.set_xlim([-0.5, 0.1])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([-0.3, 0.3])
    # Adjusting the view angle
    ax.view_init(elev=20., azim=30)
    ax.legend()
    plt.show()


def extract_patches_from_mask(mask, patch_size):
    """
    TODO: Extract patches of the given size from the mask and compute the sum of pixel values for each patch.

    :param mask: The mask from which patches will be extracted.
    :param patch_size: The size of each patch.
    :return: A tuple containing two numpy arrays:
             - An array of extracted patches.
             - An array of scores (sums of pixel values) for each patch.
    """
    patches = []
    scores = []
    height, width = mask.shape

    # Ensure that the mask dimensions are divisible by patch_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Mask dimensions must be divisible by patch_size.")

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = mask[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            score = patch.sum() / (patch_size ** 2)
            if score > 0.0:
                score = 1
            else:
                score = 0
            scores.append(score)

    # Convert lists to numpy arrays
    patches = np.array(patches)
    scores = np.array(scores)

    return patches, scores


def patch_centers(abs_center, patch_size):
    '''
    TODO: Calculate coordinates of patch centers

    :param abs_center: Absolute center point coordinate w.r.t original image, shape (1, 2)
    :return: Absolute point coordinate of each patch center w.r.t original image, shape (num_patches, 2)
    '''
    image_size = config['resize_imagesize']
    patch_grid = image_size //patch_size
    num_patches = patch_grid ** 2

    # Create relative coordinates
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, patch_grid - 1, patch_grid),
                                    torch.linspace(0, patch_grid - 1, patch_grid), indexing='ij')
    relative_coords = torch.stack((grid_y, grid_x), dim=-1).reshape(num_patches, -1)  # shape ([1444, 2])

    # Convert to absolute coordinates
    patch_centers = (relative_coords + 0.5) * patch_size + (abs_center - (patch_grid * patch_size) // 2)

    return patch_centers



def select(pre_keypoints2d, conf, score):
    '''
    TODO: Select top N confident keypoint positions predictions
    :param pre_keypoints2d: All predictions for keypoint positions, Tensor [batch_size, patches_number, number_keypoints, 2]
    :param conf: Confidence scores of predictions. Tensor [batch_size, patches_number, number_keypoints]
    :param score: Mask score. Tensor [batch_size, patches_number]
    :return: Selected predictions. Tensor [batch_size, number_keypoints, 2]
    '''
    batchsize, patches_num, num_keypoints, _ = pre_keypoints2d.shape
    top_n = 12
    top_cof_index = torch.zeros((batchsize, num_keypoints, top_n), dtype=torch.long, device=conf.device)
    for i in range(batchsize):
        for j in range(patches_num):
            conf[i,j,:] = conf[i,j,:] * score[i, j]


    for j in range(batchsize):
        for i in range(num_keypoints):
            top_cof_index[j, i, :] = torch.topk(conf[j, :, i], top_n).indices
    selected_keypoints = torch.zeros((batchsize, num_keypoints, 2), device=conf.device)
    s = torch.zeros((batchsize,  num_keypoints, top_n, 2))

    # Calculate mean of top N predictions as the final posision predictionn for each keypoint
    for j in range(batchsize):
        for i in range(num_keypoints):
            s[j, i, :, : ] = pre_keypoints2d[j, top_cof_index[j, i, :], i, :]
            selected_keypoints[j, i, :] = torch.mean(pre_keypoints2d[j, top_cof_index[j, i, :], i, :], dim=0)

    return selected_keypoints, s


def pnp(keypoints_3d, pre_keypoints2d, conf, score):
    ''' TODO: Estimate pose by RANSAC_PnP
    :param keypoints_3d: tensor [num_keypoints, 3]
    :param pre_keypoints2d: tensor [batch, patch, num_keypoints, 2]
    :param conf: tensor [batch, patch, num_keypoints]
    :param camera_matrix: numpy,array [3,3]
    :param dist_coeffs: numpy,array [1,4]
    :return: R numpy.array [batch, 3, 3]
             t numpy.array [batcn, 3]
    '''

    # Filter keypoints by confidence
    batchsize, patches_num, num_keypoints, _ = pre_keypoints2d.shape
    rvec = np.zeros((batchsize, 3, 3))
    tvec = np.zeros((batchsize, 3))
    selected_keypoints, _ = select(pre_keypoints2d, conf, score)
    selected_keypoints = selected_keypoints.cpu().detach().numpy()
    keypoints_3d = keypoints_3d.cpu().detach().numpy()
    camera_matrix = np.array(config['intrinsic_mat']).reshape(3,3)
    dist_coeffs = np.array(config['dist_coeffs'])

    for j in range(batchsize):
        assert selected_keypoints.shape[1] == keypoints_3d.shape[
            0], 'Points 3D and points 2D must have the same number of vertices'
        if selected_keypoints.shape[1] >= 4:  # Minimum 4 points required for PnP
            _, r, t, _ = cv2.solvePnPRansac(keypoints_3d, selected_keypoints[j], camera_matrix, dist_coeffs,
                                            reprojectionError=12.0, flags=cv2.SOLVEPNP_EPNP, iterationsCount=200)
            R, jac = cv2.Rodrigues(r)
            rvec[j, :, :] = R
            tvec[j, ] = t.reshape(-1)
        else:
            print("Not enough keypoints with high confidence for PnP")
            return None, None
    return rvec, tvec



if __name__ == "__main__":
    stl_file_path = 'grasper_cut.stl'
    points, your_mesh = read_stl(stl_file_path)
    n_samples = 8
    npoint = 8
    keypoints_3d = farthest_point_sampling(points, n_samples)
    # keypoints_3d = select_top_points(points, npoint, vertical_axis=0)
    np.save('grasperCUT_FPS8.npy', keypoints_3d)
    print(keypoints_3d)
