import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import HandPoseDataset
from model import ResNetTransformerModel
from loss import HandParamsLoss
import smplx
from torch.utils.data import ConcatDataset
import os
import numpy as np
import cv2
import math
from geometry.geometry import batch_rodrigues
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hand_model = smplx.MANOLayer(data_dir='data',
                             model_path=os.path.join('data', 'mano/'),
                             gender='neutral',
                             create_body_pose=False).to(device)
def cosine_learning_rate(configs, epoch, batch_iter, optimizer, train_batch):
    total_epochs = configs['epochs']
    warm_epochs = configs['warmup_epochs']
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1/2 * (1 + math.cos(batch_iter * math.pi /
                                     ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = configs['lr'] * lr_adj

    # print('Learning rate:', configs['lr'] * lr_adj)

    return configs['lr'] * lr_adj

def train(model, train_loader, criterion, optimizer, curr_epoch, configs, device):
    model.train()
    total_loss = 0
    train_mpjpe = 0
    for idx, data in enumerate(train_loader):
        lr = cosine_learning_rate(
            configs, curr_epoch, idx + 1 + (curr_epoch - 1) * len(train_loader), optimizer, configs['train_batch_count']
        )

        pressure_data = data['pressure'].to(device)
        hand_label = data['hand_label']
        translation = data['translation'].to(device)
        metadata = data['metadata']

        hand_label = {
            'betas': hand_label['betas'].to(device),
            'hand_pose': hand_label['hand_pose'].to(device),
            'global_orient': hand_label['global_orient'].to(device)
        }

        # optimizer.zero_grad()
        model_outputs = model(pressure_data)

        outputs = process_model_output(model_outputs, translation)
        gt = process_model_output(hand_label, translation)

        loss, loss_dict = criterion(outputs, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # twist_angle = model_outputs['hand_pose'].clone().detach().cpu().numpy()
        # print('train:', loss_dict['mpjpe'].mean(), loss_dict['twist'], twist_angle[:, [21, 24]].mean(0))
        # train_mpjpe = train_mpjpe + loss_dict['mpjpe']
    # print('train_mpjpe:', train_mpjpe / len(train_loader))
        train_mpjpe = train_mpjpe + loss_dict['mpjpe'].mean().item()
    print('train_mpjpe:', train_mpjpe / len(train_loader))
    return total_loss / len(train_loader)


def mano_joints_with_fingertips(mano_joints, mano_vertices):
    """
      get joints from MANO model
      MANO model does not come with fingertip joints, so we have selected vertices
      that correspond to fingertips
      AND tranform to openpose format
        fingertip_idxs = [333, 444, 672, 555, 745]
      """
    tip1 = (mano_joints[:, [15]] + mano_vertices[:, [745]] * 3) / 4
    tip2 = (mano_joints[:, [3]] + mano_vertices[:, [333]] * 3) / 4
    tip3 = (mano_joints[:, [6]] + mano_vertices[:, [444]] * 3) / 4
    tip4 = (mano_joints[:, [12]] + mano_vertices[:, [555]] * 3) / 4
    tip5 = (mano_joints[:, [9]] + mano_vertices[:, [672]] * 3) / 4
    format_joint = torch.cat([mano_joints[:, [0, 13, 14, 15]],
                              tip1,
                              mano_joints[:, [1, 2, 3]],
                              tip2,
                              mano_joints[:, [4, 5, 6]],
                              tip3,
                              mano_joints[:, [10, 11, 12]],
                              tip4,
                              mano_joints[:, [7, 8, 9]],
                              tip5],
                             dim=1)
    return format_joint

def process_model_output(model_outputs, translation):
    outputs = {}
    outputs['betas'] = model_outputs['betas'].reshape(-1, 10)
    outputs['hand_pose'] = batch_rodrigues(model_outputs['hand_pose'].reshape(-1, 3)).reshape(-1, 15, 3, 3)
    outputs['global_orient'] = batch_rodrigues(model_outputs['global_orient'].reshape(-1, 3)).reshape(-1, 1, 3, 3)
    outputs['translation'] = translation.reshape(-1, 3).detach()


    with torch.no_grad():
        mano_output = hand_model(global_orient=outputs['global_orient'],
                                 hand_pose=outputs['hand_pose'],
                                 betas=outputs['betas'],
                                 transl=outputs['translation'],
                                 pose2rot=False)
        joints = mano_output['joints']
        vertices = mano_output['vertices']
        outputs['kp_3d'] = mano_joints_with_fingertips(joints, vertices)
        # outputs['kp_3d'] = joints
    return outputs


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    mpjpe = 0
    with torch.no_grad():
        for data in val_loader:
            pressure_data = data['pressure'].to(device)
            hand_label = data['hand_label']
            translation = data['translation'].to(device)
            metadata = data['metadata']

            hand_label = {
                'betas': hand_label['betas'].to(device),
                'hand_pose': hand_label['hand_pose'].to(device),
                'global_orient': hand_label['global_orient'].to(device)
            }

            model_outputs = model(pressure_data)
            outputs = process_model_output(model_outputs, translation)
            gt = process_model_output(hand_label, translation)

            loss, loss_dict = criterion(outputs, gt)
            total_loss += loss.item()
            # print('val:', loss_dict)
            mpjpe = mpjpe + loss_dict['mpjpe'].mean().item()
        print('val_mpjpe:', mpjpe/len(val_loader))
    return total_loss / len(val_loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_betas = 10
    num_hand_pose = 45
    num_global_orient = 3

    dataset_split = {
        'qy': {'train': list(range(1, 7)), 'val': [7]},
        'lby': {'train': list(range(1, 7)), 'val': [7]},
        'xyf': {'train': list(range(1, 7)), 'val': [7]},
        'yqj': {'train': list(range(1, 7)), 'val': [7]},
        'wq': {'train': list(range(1, 7)), 'val': [7]},
        'dhy': {'train': list(range(1, 5)), 'val': [7]},
        'fcf': {'train': [1, 2, 3, 4, 6], 'val': [7]},
        'zyk': {'train': list(range(1, 7)), 'val': [7]},
        'ky': {'train': list(range(1, 7)), 'val': [7]},
        'oygw': {'train': list(range(1, 7)), 'val': [7]},
        'wzj': {'train': [1, 2, 3, 4, 5, 7], 'val': [8]},
        'zch': {'train': [1, 2, 3, 4, 6], 'val': [7]},
        'wyc': {'train': list(range(1, 6)), 'val': [7]},
        'jwq': {'train': list(range(1, 7)), 'val': [7]},
        'dyk': {'train': list(range(1, 7)), 'val': [7]},
        'wzy': {'train': list(range(1, 7)), 'val': [7]},
        'xft': {'train': list(range(1, 7)), 'val': [7]},
        'nmt': {'train': [1, 3, 4, 5, 6, 12, 14], 'val': [7]},
    }


    train_datasets = []
    for user, splits in dataset_split.items():
        train_datasets.append(
            HandPoseDataset(
                pressure_dir=f'/workspace/nmt/Pressure_DATA/filtered_pressure/{user}',
                label_dir=f'/workspace/nmt/camera0_GT/{user}',
                file_indices=splits['train'],
                user=user,
                sequence_length=16,
                step=1
            )
        )
    train_dataset = ConcatDataset(train_datasets)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_datasets = []
    for user, splits in dataset_split.items():
        val_datasets.append(
            HandPoseDataset(
                pressure_dir=f'/workspace/nmt/Pressure_DATA/filtered_pressure/{user}',
                label_dir=f'/workspace/nmt/camera0_GT/{user}',
                file_indices=splits['val'],
                user=user,
                sequence_length=16,
                step=1
            )
        )
    val_dataset = ConcatDataset(val_datasets)
    print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_epochs = 50
    min_val_loss = float('inf')
    best_epoch = 0
    lr = 1e-5

    model = ResNetTransformerModel(num_betas=num_betas, num_hand_pose=num_hand_pose,
                                   num_global_orient=num_global_orient).to(device)

    criterion = HandParamsLoss(
        betas_loss_weight=50.0,
        hand_pose_loss_weight=800,
        global_orient_loss_weight=50.0,
        kp_3d_loss_weight=500.0,
        smooth_betas_loss_weight=0.01,
        smooth_hand_pose_loss_weight=0.01,
        smooth_global_orient_loss_weight=0.01,
        smooth_kp_3d_loss_weight=0.01,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

    configs = {
        'epochs': num_epochs,
        'lr': lr,
        'warmup_epochs': 5,
        'train_batch_count': len(train_dataset),
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, configs, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),'/workspace/nmt/Pressure_HPE/test_result/models/Cross_validation_5_8.pth')
        #


if __name__ == "__main__":
    main()