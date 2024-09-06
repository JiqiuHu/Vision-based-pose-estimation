from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from ultilities import get_parameter_number
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from config import config
from dataset import Pose_keypointDataset
from vit_training import *
from keypoint import pnp


class ViTKeypointModel(nn.Module):
    def __init__(self,  num_keypoints=8, image_size=608,
                 patch_size=38):
        super(ViTKeypointModel, self).__init__()
        # Load pre-trained ViT model
        configuration = ViTConfig(attn_implementation="eager")
        configuration.patch_size = patch_size
        configuration.num_hidden_layers = 12
        configuration.image_size = image_size
        self.vit = ViTModel(configuration)
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        self.configuration = self.vit.config
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_width = 1080
        self.image_height = 1920
        self.num_patches = (image_size // patch_size) ** 2
        # Freeze the specified number of encoder layers
        # self.freeze_vit_layers(6)

        # Keypoint regression head
        self.keypoint_head = nn.Sequential(
            nn.Linear(self.configuration.hidden_size * 2, 1024), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 516),  
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(516, 516), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(516, 218),  
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(218, 218),  
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(218 ,num_keypoints * 2) # 2 for x and y offsets
        )
        # Confidence score head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.configuration.hidden_size * 2, 1024), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 516), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(516, 218), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(218, num_keypoints),
            nn.Sigmoid()
        )

    def freeze_vit_layers(self, num_layers_to_freeze):
        """Freeze the specified number of ViT encoder layers."""
        layers = self.vit.encoder.layer
        for i, layer in enumerate(layers):
            # print(layer)
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x, patch_centers):
        # Get the features from ViT
        batch_size = x.shape[0]
        outputs = self.vit(x, output_attentions = True)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
        attntion = outputs.attentions

        # Extract positional encoding
        pos_encoding= self.vit.embeddings.position_embeddings  # Shape: (num_patches + 1, hidden_size)
        pos_encoding = pos_encoding.expand(batch_size, self.num_patches + 1, -1)  # Shape: (batch_size, num_patches + 1, hidden_size)

        # Add positional encoding
        hidden_states_with_pos = torch.cat((hidden_states, pos_encoding), dim=-1) # Shape: (batch_size, num_patches + 1, hidden_size)

        # Flatten the features and pass through the keypoint head
        keypoints_offsets = self.keypoint_head(hidden_states_with_pos)[:, 1:self.num_patches+1, :]
        keypoints_offsets = keypoints_offsets.view(-1, self.num_patches, self.num_keypoints, 2)

        # Pass through the confidence head
        confidence_scores = self.confidence_head(hidden_states_with_pos)[:, 1:self.num_patches+1, :]  # Shape: (batch_size, num_patches, num_keypoints)

        # Calculate the predicted keypoints' 2D coordinates
        # patch_centers = self.patch_centers.to(x.device).unsqueeze(0).unsqueeze(2)  # Shape: (1, num_patches, 1, 2)
        predicted_keypoints = torch.unsqueeze(patch_centers, 2) + keypoints_offsets  # Shape: (batch_size, num_patches, num_keypoints, 2)

        # return predicted_keypoints, confidence_scores, pos_encoding, attntion

        return predicted_keypoints, confidence_scores



def get_data(root_dir, tool_type, points, preprocess=None, device='cpu'):
    '''
    TODO: split dataset into training data and validation data
    '''

    full_data = Pose_keypointDataset(root_dir, points, preprocess, device)
    extrinsic_mat = full_data.extrinsic_mat
    intrinsic_mat = full_data.intrinsic_mat
    dist_coeffs = full_data.dist_coeffs

    train_size = int(0.9 * len(full_data))
    val_size = len(full_data) - train_size

    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    validation_loader = DataLoader(val_data, batch_size=8, shuffle=True)
    print('{}:'
          'The number of full dataset: {}, '
          'The number of training dataset: {}, '
          'The number of validation dataset: {}'.format(tool_type, len(full_data), train_size, val_size))


    return train_loader, validation_loader, intrinsic_mat, extrinsic_mat, dist_coeffs



if __name__ == "__main__":
    # Choosing GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    print('We are using {}'.format(device))

    # Loading the segmentation dataset
    root_dir_gra = config['root_dir_grasper']
    root_dir_sci = config['root_dir_scissor']
    # grasper_path = '/vol/bitbucket/jh523/dataset/Training data/grasper.stl'
    # scissor_path = '/vol/bitbucket/jh523/dataset/Training data/scissor.stl'
    points_gra = np.array(np.load('grasperCUT_FPS8.npy'), dtype=np.float64)


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_gra, val_gra, intrinsic_mat, extrinsic_mat, dist_coeffs = get_data(root_dir_gra, 'Grasper',
                                                                             points_gra,preprocess,device)
    # train_sci, val_sci, intrinsic_mat, extrinsic_mat, dist_coeffs = get_data(root_dir_sci, 'Scissor',preprocess,
    #                                           device)

    # #Define optimizer and loss function
    train_model(train_gra, val_gra,  'Grasper', device=device)
    #
    # train_pose(train_sci, val_sci, criterion, points_sci, intrinsic_mat, 'Scissor', device = device)



