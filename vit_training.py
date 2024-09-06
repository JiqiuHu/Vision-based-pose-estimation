import torch
from tqdm import tqdm
from ultilities import get_parameter_number
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from key_model import *
import numpy as np


def compute_loss(pred_keypoints, true_keypoints, score, pred_confidences):
    '''
    TODO
    :param pred_keypoints: tensor ([batch, num_patches, num_points, 2])
    :param true_keypoints: tensor ([[batch, N, 1, 2]])
    :param score:  tensor ([batch, num_patches])
    :param pred_confidences: tensor ([batch, num_patches, num_points])
    :return:
    '''
    batch_size, num_patches, num_keypoints, _ = pred_keypoints.shape
    true_keypoints = true_keypoints.view(batch_size, 1, num_keypoints, 2)  # shape (batch, 1, N, 2)
    # Compute L1 loss for keypoints where masks are non-zero
    diff_points = torch.sum(torch.abs(pred_keypoints - true_keypoints), dim=3) #shape (batch, num_patch, N)
    diff_patches = torch.sum(diff_points, dim = 2) #shape (batch, num_patch)
    # mask_patch1 = score * diff_patches
    # non_zero_values1 = diff_patches[score.nonzero(as_tuple=True)]
    #
    # l1_pose = torch.mean(non_zero_values1.float())

    l1_pose = (score * diff_patches).mean()

    # Compute L2 loss for keypoints where masks are non-zero
    diff_l2 = torch.sqrt(torch.sum((pred_keypoints - true_keypoints) ** 2, dim=3))  #shape (batch, num_patch, N)
    diff_points_conf = torch.sum(torch.abs(pred_confidences - torch.exp(-diff_l2)), dim=2) #shape (batch, num_patch)
    # mask_patch2 = score * diff_points_conf
    l1_conf = (score * diff_points_conf).mean()
    # non_zero_values2 =  diff_points_conf[score.nonzero(as_tuple=True)]
    # l1_conf = torch.mean(non_zero_values2.float())
    Loss = l1_pose + 1000 * l1_conf

    return Loss, l1_conf, l1_pose


def train_one_epoch(training_loader, optimizer, model, epoch_index, tb_writer, device):
    running_loss1 = 0.
    running_loss2 = 0.
    running_loss3 = 0.
    running_loss4 = 0.
    running_loss5 = 0.
    last_loss1 = 0.
    last_loss2 = 0.
    last_loss3 = 0.
    last_loss4 = 0.
    last_loss5 = 0.

    loop = tqdm(training_loader, leave=False)
    k_gra = np.load('grasperCUT_FPS8.npy').reshape(-1, 3)
    key3d = torch.tensor(k_gra, dtype=torch.float32)


    for i, (img, imf_c, images, score, centroid, true_keypoints, r, t) in enumerate(loop):
        #Shape (B, 3, 1080, 1920), (B, 3, 608, 608), (B, 256), (B, 256, 2), (B, 8, 1, 2), (B, 3, 3), (B, 3)

        # Predicting 2 D keypoint coordinates and confidence
        # Shape: (batch_size, num_patches, num_keypoints, 2)
        pre_keypoints, confidence = model(images, centroid)
        pre = pre_keypoints.clone()
        conf = confidence.clone()
        s = score.clone()
        pre_r, pre_t = pnp(key3d, pre, conf, s)
        pre_r = torch.tensor(pre_r, device=device)
        pre_t = torch.tensor(pre_t, device=device)
        loss_r = ((pre_r - r) ** 2).sum(-1).sum(-1).mean()
        loss_t = torch.linalg.norm((pre_t - t), dim = -1).mean()

        # Compute the loss and its gradients
        loss, l1_conf, l1_pose = compute_loss(pre_keypoints, true_keypoints, score, confidence)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, requires_grad=True)

        # Zero pose initial prediction network's gradients for every batch!
        optimizer.zero_grad()
        # loss_2d.backward(retain_graph=True)
        loss.backward()
        # loss_2d.backward()
        # Adjust learning weights
        optimizer.step()
        # torch.nn.utils.clip_grad_norm(pose_model.parameters(), max_norm=1)

        running_loss1 += loss.item()
        running_loss2 += l1_conf.item()
        running_loss3 += l1_pose.item()
        running_loss4 += loss_r.item()
        running_loss5 += loss_t.item()


        if i % 8 == 7:
            last_loss1 = running_loss1 / 8  # reports on the loss for every 4 batches.
            last_loss2 = running_loss2 / 8
            last_loss3 = running_loss3 / 8
            last_loss4 = running_loss4 / 8
            last_loss5 = running_loss5 / 8

            # print('  batch {} loss: {}'.format(batch_number, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train(total)', last_loss1, tb_x)
            tb_writer.add_scalar('Loss/train(confidence)', last_loss2, tb_x)
            tb_writer.add_scalar('Loss/train(keypoint)', last_loss3, tb_x)
            tb_writer.add_scalar('Loss/train(rotation)', last_loss4, tb_x)
            tb_writer.add_scalar('Loss/train(translation)', last_loss5, tb_x)
            running_loss1 = running_loss2 = running_loss3 =  running_loss4 =  running_loss5 = 0.

            loop.set_postfix(
                batch_loss= last_loss1,
                batch_conf= last_loss2,
                batch_keypoints = last_loss3,
                batch_r = last_loss4,
                batch_t=last_loss5
            )
            loop.set_description(f"Epoch {epoch_index + 1}")
            torch.cuda.empty_cache()

    return last_loss1, last_loss2, last_loss3


def train_model(train_loader, validation_loader, tool_type, device):
    model = ViTKeypointModel1()
    model.train()
    print('The number of parameters of Pose prediction Network: ')
    get_parameter_number(model)
    model.to(device)

    # Initializing hyperparameters

    epoch_number = 0
    EPOCHS = 300
    best_vloss = 0.01
    Lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)

    # Initializing in a separate cell
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if tool_type == 'Grasper':
        writer = SummaryWriter('VIT6layer/Grasper/pose_trainer_{}'.format(timestamp))
    else:
        writer = SummaryWriter('VIT6layer/Scissor/pose_trainer_{}'.format(timestamp))

    # Training
    print('-' * 25, 'TRAINING START', '-' * 25)
    for epoch in range(EPOCHS):
        print('optimizer Learning rate: {}'.format(scheduler.get_last_lr()))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss1, avg_loss2, avg_loss3 = train_one_epoch(train_loader, optimizer, model, epoch, writer, device)

        running_vloss1 = 0.0
        running_vloss2 = 0.0
        running_vloss3 = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, (img, vimgc, vinputs, vscore, vcentroid, vtrue_keypoints, vr, vt) in enumerate(validation_loader):
                # Initial Prediction for grasper
                v_keypoints, v_conf = model(vinputs, vcentroid)

                # Compute the loss
                vloss, vl1_conf, vl1_pose = compute_loss(v_keypoints, vtrue_keypoints, vscore, v_conf)

                # Gather data and report
                running_vloss1 += vloss
                running_vloss2 += vl1_conf
                running_vloss3 += vl1_pose
                torch.cuda.empty_cache()

        avg_vloss1 = running_vloss1 / (i + 1)
        avg_vloss2 = running_vloss2 / (i + 1)
        avg_vloss3 = running_vloss3 / (i + 1)

        print('LOSS train_total {} train_conf {} train_keypoint{} '
              'val_total {} val_conf {} val_keypoint{}'.format(
            avg_loss1, avg_loss2, avg_loss3, avg_vloss1, avg_vloss2, avg_vloss3))

        # scheduler.step(avg_loss1)

        torch.cuda.empty_cache()

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss (total)',
                           {'Training': avg_loss1, 'Validation': avg_vloss1},
                           epoch_number)
        writer.flush()

        writer.add_scalars('Training vs. Validation Loss (Confidence)',
                           {'Training': avg_loss2, 'Validation': avg_vloss2},
                           epoch_number)
        writer.flush()

        writer.add_scalars('Training vs. Validation Loss (Keypoints)',
                           {'Training': avg_loss3, 'Validation': avg_vloss3},
                           epoch_number)
        writer.flush()

        # Track best performance, and save the model's state
        checkpoint_path = '/vol/bitbucket/jh523/checkpoints/{}_epoch_6layer'.format(epoch_number)
        if epoch_number % 20 == 0:
            print('Save the model')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_number
            }, checkpoint_path)

        epoch_number += 1
        if epoch_number < 220:
            scheduler.step()

    print('-' * 25, 'TRAINING Complete', '-' * 25)
