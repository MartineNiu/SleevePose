import os
import torch
import smplx
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from geometry.geometry import *
from data_loader import HandPoseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hand_model = smplx.MANOLayer(data_dir='data',
                             model_path=os.path.join('data', 'mano/'),
                             gender='neutral',
                             create_body_pose=False).to(device)

class ResNetTransformerModel(nn.Module):
    def __init__(self, num_betas, num_hand_pose, num_global_orient, sequence_length=16, d_model=512, nhead=8, num_encoder_layers=2,
                 dim_feedforward=2048):
        super(ResNetTransformerModel, self).__init__()
        self.sequence_length = sequence_length

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Identity()

        # Transformer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.regressor = Regressor(d_model)

        self.apply(self.init_weights)

    def forward(self, x):
        batch_size, seq_len, h, w = x.shape
        total_frames = batch_size * seq_len

        x = x.view(total_frames, 1, h, w)
        x = self.resnet(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.transformer_encoder(x)
        x = x.view(total_frames, -1)
        params = self.regressor(x)

        return params

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

class Regressor(nn.Module):
    def __init__(self, feature_len):
        super(Regressor, self).__init__()

        npose = 16 * 6

        self.fc1 = nn.Linear(feature_len + npose + 10, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        init_pose = torch.zeros((96)).type(torch.float32)
        init_shape = torch.zeros((10)).type(torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 16, 3, 3)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 48)

        return {
            'betas': pred_shape,
            'hand_pose': pose[:, 3:],
            'global_orient': pose[:, :3],
            'rotmat': pred_rotmat
        }

