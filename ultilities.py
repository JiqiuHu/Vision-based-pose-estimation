import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import models
from stl import mesh
from PIL import Image
from tqdm import tqdm


def segment_image(image_path, model, preprocess):
    """
    TODO: Read an image and give the segmentation output

    :param image_path: path of the image
    :param model: trained or initial segmentation model
    :param preprocess: pre-process of image before fed into model
    :return: mask of segmentation mask
    """
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output


def get_parameter_number(model):
    """
    TODO: Print the number of total parameters and trainable parameters of a model

    :param model: the model that you want to know about parameters
    :return: None
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: {}'.format(total_num), 'Trainable parameters: {}'.format(trainable_num))


def load():
    """
    TODO: Load the segmentation model after training

    :return: loaded segmentation model
    """
    # Loading pre-trained DeepLabV3+ model
    model = models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
    # Froze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Only train parameters in classification layer
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def maskd(im):
    '''
    TODO: Apply an ellipse mask on the semantic mask
    :param im: Orginal mask, numpy.array (shape [1080, 1920])
    :return: maksed semantic mask: numpy.array (shape [1080, 1920])
    '''
    # Image size
    height, width = im.shape[0], im.shape[1]

    # Center of ellipse
    center_x, center_y = 0.5, 0.5

    # Height and width of the ellipse
    ellipse_width, ellipse_height = 0.73, 0.73

    # Convert the pixel unit
    center_x_px = int(center_x * width)
    center_y_px = int(center_y * height)
    ellipse_width_px = ellipse_height_px=int(ellipse_width * width / 2)
    # ellipse_height_px = int(ellipse_height * height / 2)

    # Create all black background
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create ellipse mask
    y, x = np.ogrid[:height, :width]
    ellipse_mask = ((x - center_x_px) ** 2) / (ellipse_width_px ** 2) + ((y - center_y_px) ** 2) / (
                ellipse_height_px ** 2) <= 1

    # Apply ellipse mask
    mask[ellipse_mask] = 255
    mask = mask / 255.0
    im = np.array(im * mask).astype(np.uint8)
    return im


def zoomin(image, mask):
    """
    TODO: Crop images according to the segmentation mask and keep the absolute center point coordinate.

    :param image: The original image that includes two tools.
    :param mask: The mask for this image.
    :return: A tuple containing cropped images, masks, and the absolute center point coordinate.
    """
    fixed_size = (608, 608)  # Fixed size for cropping
    labels = np.unique(mask)
    # cropped_img, cropped_mask = None, None
    abs_center = np.zeros([1, 2])
    mask = maskd(mask)


    for label in labels:
        if label == 0:
            continue

        # Create binary mask for the current label
        label_mask = np.where(mask == label, 255, 0).astype(np.uint8)

        # Find contours for the current label
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue  # Skip if no contours are found

        # Get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)

        # Ca;culate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        x_start = max(center_x - fixed_size[1] // 2, 0)
        y_start = max(center_y - fixed_size[0] // 2, 0)


        # # Adjust bounding box to ensure it fits within the image
        x_start = min(x_start, image.shape[1] - fixed_size[1])
        y_start = min(y_start, image.shape[0] - fixed_size[0])
        x_end = x_start + fixed_size[1]
        y_end = y_start + fixed_size[0]

        # Crop the image and mask
        cropped_img = image[y_start:y_end, x_start:x_end]
        cropped_mask = mask[y_start:y_end, x_start:x_end]

        # Calculate the absolute center point of the cropped region in the original image
        abs_center[0] = (x_start + (x_end - x_start) // 2, y_start + (y_end - y_start) // 2)
        if cropped_img.shape[0] != fixed_size[1] or cropped_img.shape[1] != fixed_size[1]:
            raise ValueError('Cropped image size is not 608*608')
    if cropped_img is None or cropped_mask is None:
        raise ValueError('No valid crop found in the imag')
    return cropped_img, cropped_mask, abs_center


def quaternion_to_rotation_matrix(q):
    # TODO Convert quaternion to 3*3 rotation matrix
    """
    :param q: quaternion, [batch, w, x, y, z]
    :return: 3*3 rotation matrix
    """
    N = q.shape[0]

    R_batch = torch.zeros((N, 3, 3))
    for i in range(N):
        w, x, y, z = q[i]
        R_batch[i] = torch.tensor([
            [x**2 + w**2 - y**2 - z**2, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), w**2 - x**2 + y**2 - z**2, 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), w**2 - x**2 - y**2 + z**2]
        ])
    return R_batch


def rotation_matrix_to_quaternion(matrix):
    # TODO Convert a 3x3 rotation matrix to a quaternion.
    """
    Parameters:
    matrix (tensor): 3x3 rotation matrix.

    Returns:
    tensor: Quaternion (w, x, y, z).
    """
    m = matrix
    trace = torch.trace(m)
    if trace > 0:
        s = 2.0 * torch.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s

        else:
            s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = torch.tensor([w, x, y, z])
    if w < 0:
        q =-q
    return q



def get_label(masks):
    # TODO get labels in a mask
    """
    :param masks: batch of masks [b,c,h,w]
    :return: list of labels appearing in each image [label in image1, label in image2, ...]
    """
    # label_map = {0: 'background', 1: 'grasper', 2: 'scissor'}
    masks_soft = torch.nn.Softmax(masks, dim = 1)
    masks_label = torch.argmax(masks_soft, dim = 1)
    labels = []
    for i in range(masks_label.shape[0]):
        label = torch.unique(masks_label[i])
        labels.append(label)
    return labels


def camera_to_object(extrinsic_matrix, pose_world_to_object):
    # TODO convert pose wrt world coordinates to pose wrt camera coordinate
    """
    :param extrinsic_matrix: camera extrinsic matrix [3,3]
    :param pose_world_to_object: object pose gotten from blender [b,4,4]
    :return: objects pose wrt camera coordinate [b,3,3], [b,3]
    """
    rotation_otw = pose_world_to_object[:3, :3]
    trans_otw = pose_world_to_object[:3, 3]
    rotation_wtc = extrinsic_matrix[:3, :3]
    trans_wtc = extrinsic_matrix[:3, 3]

    rotation_otc = np.matmul(rotation_wtc, rotation_otw)
    trans_otc = np.matmul(rotation_wtc, trans_otw) + trans_wtc
    return rotation_otc, trans_otc


def rotation_vector_to_matrix(rot_vecs):
    """
    TODO Convert rotation vectors to rotation matrices.

    :param rot_vecs: Tensor of shape [batch_size, 3] where each row is a rotation vector.
    :return: Tensor of shape [batch_size, 3, 3] where each slice is a rotation matrix.
    """
    batch_size = rot_vecs.shape[0]

    # Compute rotation angles (theta) and unit rotation axes (u)
    angles = torch.norm(rot_vecs, dim=1, keepdim=True)
    axes = rot_vecs / (angles + 1e-10)  # Add a small constant to avoid division by zero

    # Compute the skew-symmetric matrix K
    K = torch.zeros(batch_size, 3, 3, device=rot_vecs.device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    # Compute R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    K2 = torch.bmm(K, K)

    I = torch.eye(3, device=rot_vecs.device).unsqueeze(0).expand(batch_size, 3, 3)

    R = I + sin_angles.unsqueeze(1) * K + (1 - cos_angles).unsqueeze(1) * K2

    return R.to(device = rot_vecs.device)

def project(xyz, K, R, T):
    # TODO project 3D points of object into 2D images pixel
    """
    xyz: [N, 3]
    K: [3, 3]
    R: [batch, 3, 3]
    T: [batch, 3]
    return xy: [batch, N, 2]
    """
    batch_size = R.shape[0]
    N = xyz.shape[0]
    xyz = torch.matmul(R, torch.t(xyz))
    xyz = xyz + T[:, :, None]
    xyz = xyz.reshape(batch_size, N, -1)
    xyz = torch.matmul(xyz, torch.t(K))
    xy = xyz[:, :, :2] / xyz[:, :, 2:]
    return xy


def R_to_q(r):
    # TODO convert the output to a quaternion
    """
    :param r: the output of network
    :return: quaternion
    """
    q = r.clone().detach()
    theta = r[1]
    q[1] = torch.cos(r[1])
    q[2] = torch.sin(r[1]) * r[2]
    q[3] = torch.sin(r[1]) * r[3]
    q[4] = torch.sin(r[1]) * r[4]
    q = torch.nn.functional.normalize(q, dim=0)
    print(q.shape)
    return q


def rq_to_rq(q):
    q_ = torch.zeros(q.shape)
    theta = q[:, 0]
    theta_cos = torch.cos(theta)
    theta_sin = torch.sin(theta)
    q_[:, 0] = theta_cos
    q_[:, 1] = q[:, 1] * theta_sin
    q_[:, 2] = q[:, 2] * theta_sin
    q_[:, 3] = q[:, 3] * theta_sin
    q_ = torch.nn.functional.normalize(q_, dim=1)
    return q_



def extract_points_from_stl(file_path):
    # loading an existing stl file:
    your_mesh = mesh.Mesh.from_file(file_path)

    # Get all vertices
    points = your_mesh.vectors.reshape(-1, 3)

    # Remove the repeated vertics
    unique_points = np.unique(points, axis=0)

    return unique_points


def loss_projection(r_pre, t_pre, r_g, t_g, intrinsic_mat, points):

    points_3d_pre = project(points, intrinsic_mat, r_pre, t_pre)
    points_3d_g = project(points, intrinsic_mat, r_g, t_g)
    loss_func = torch.nn.MSELoss(reduction='mean')
    n = torch.linalg.vector_norm((points_3d_pre - points_3d_g), dim = -1)
    return torch.mean(n)


def masked_image(image_path, mask_path, label, background = None):
    # TODO Add the segmentation mask to the image in order to ignore another tool
    '''
    :param image_path: path of images
    :param mask_path: path of mask file
    :param label: label of one tool that you want to ignore ('Grasper': 1, 'Scissor': 2)
    :param background: path of background used to paint mask, if you don't have the original or similar image of
    background, this algorithm will choose cv2.inpaint method.
    :return:
    '''

    # Loading the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB form

    # Loading the mask
    data = np.load(mask_path)
    mask = data['segmentation_masks']

    # convert mask to a binary image，foreground is 255，background is 0
    label_mask = np.where(mask == label, 255, 0).astype(np.uint8)

    # Inverse mask
    inverted_mask = cv2.bitwise_not(label_mask)

    # Convert mask to a 3-channel image
    mask_3ch = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2RGB)

    # normalize mask so that value's range is [0,1]
    mask_normalized = mask_3ch / 255.0


    if background == None:
        # Apply the mask
        masked_image = image * mask_normalized
        masked_image = masked_image.astype(np.uint8)
        inpainted_image = cv2.inpaint(masked_image, label_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Show the results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Semantic Mask')
        plt.imshow(inverted_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Masked Image')
        plt.imshow(inpainted_image)
        plt.axis('off')

        plt.show()
        return inpainted_image

    else:
        # Loading the background
        background_image = cv2.imread(background)
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # Make sure that the sizes of foreground image and background image are same
        if image.shape != background_image.shape:
            background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))


        # Use Gaussian filter to get smooth edges
        blurred_mask = cv2.GaussianBlur(inverted_mask, (21, 21), 0)

        mask_3ch = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2RGB)
        mask_normalized = mask_3ch / 255.0

        # Apply the mask on foreground image
        foreground = image * mask_normalized

        # Apply the inversed mask on background image
        inverse_mask_normalized = 1.0 - mask_normalized
        background = background_image * inverse_mask_normalized

        # combine foreground image and background image
        combined_image = cv2.addWeighted(foreground, 1, background, 1, 0)

        combined_image = combined_image.astype(np.uint8)

        # Show the results
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 4, 1)
        plt.title('Foreground Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Semantic Mask')
        plt.imshow(label_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Background Image')
        plt.imshow(background_image)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Combined Image')
        plt.imshow(combined_image)
        plt.axis('off')

        plt.show()

        return combined_image








