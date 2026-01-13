import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_loader import HandPoseDataset
from model import ResNetTransformerModel
from geometry.geometry import batch_rodrigues

class HandParamsLoss(nn.Module):
    def __init__(self,
                 betas_loss_weight=10.0,
                 hand_pose_loss_weight=10.0,
                 global_orient_loss_weight=10.0,
                 kp_3d_loss_weight=100.0,
                 smooth_betas_loss_weight=10.0,
                 smooth_hand_pose_loss_weight=10.0,
                 smooth_global_orient_loss_weight=10.0,
                 smooth_kp_3d_loss_weight=10.0,
                 twist_loss_weight=1.0,
                 ):
        super(HandParamsLoss, self).__init__()

        self.betas_loss_weight = betas_loss_weight
        self.hand_pose_loss_weight = hand_pose_loss_weight
        self.global_orient_loss_weight = global_orient_loss_weight
        self.kp_3d_loss_weight = kp_3d_loss_weight

        self.smooth_betas_loss_weight = smooth_betas_loss_weight
        self.smooth_hand_pose_loss_weight = smooth_hand_pose_loss_weight
        self.smooth_global_orient_loss_weight = smooth_global_orient_loss_weight
        self.smooth_kp_3d_loss_weight = smooth_kp_3d_loss_weight

        self.twist_loss_weight = twist_loss_weight

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion_shape = nn.L1Loss(reduction='mean').to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='sum').to(self.device)

    def forward(self, outputs, gt):
        betas_loss = self.betas_loss(outputs['betas'], gt['betas'])
        hand_pose_loss = self.hand_pose_loss(outputs['hand_pose'], gt['hand_pose'])
        global_orient_loss = self.global_orient_loss(outputs['global_orient'], gt['global_orient'])
        kp_3d_loss = self.kp_3d_loss(outputs['kp_3d'], gt['kp_3d'])

        smooth_betas_loss = self.smooth_betas_loss(outputs['betas'])
        smooth_hand_pose_loss = self.smooth_hand_pose_loss(outputs['hand_pose'])
        smooth_global_orient_loss = self.smooth_global_orient_loss(outputs['global_orient'])
        smooth_kp_3d_loss = self.smooth_kp_3d_loss(outputs['kp_3d'])

        twist_loss = torch.tensor(0).to(self.device)


        total_loss = (self.betas_loss_weight * betas_loss +
                      self.hand_pose_loss_weight * hand_pose_loss +
                      self.global_orient_loss_weight * global_orient_loss +
                      self.kp_3d_loss_weight * kp_3d_loss +
                      self.smooth_betas_loss_weight * smooth_betas_loss +
                      self.smooth_hand_pose_loss_weight * smooth_hand_pose_loss +
                      self.smooth_global_orient_loss_weight * smooth_global_orient_loss +
                      self.smooth_kp_3d_loss_weight * smooth_kp_3d_loss
                      )

        mpjpe_loss = self.mpjpe(outputs['kp_3d'], gt['kp_3d']).mean(0)

        loss_dict = {
            'mpjpe': mpjpe_loss,
            'betas': betas_loss.item(),
            'hand_pose': hand_pose_loss.item(),
            'global_orient': global_orient_loss.item(),
            'kp_3d': kp_3d_loss.item(),
            'smooth_betas': smooth_betas_loss.item(),
            'smooth_hand_pose': smooth_hand_pose_loss.item(),
            'smooth_global_orient': smooth_global_orient_loss.item(),
            'smooth_kp_3d': smooth_kp_3d_loss.item(),
            'twist': twist_loss.item()
        }

        return total_loss, loss_dict

    def betas_loss(self, pred_betas, gt_betas):
        return self.criterion_shape(pred_betas, gt_betas)

    def hand_pose_loss(self, pred_hand_pose, gt_hand_pose):
        return self.criterion_shape(pred_hand_pose, gt_hand_pose)

    def global_orient_loss(self, pred_global_orient, gt_global_orient):
        return self.criterion_shape(pred_global_orient, gt_global_orient)


    def kp_3d_loss(self, pred_kp_3d, gt_kp_3d):
        return self.criterion_keypoints(pred_kp_3d, gt_kp_3d)

    def mpjpe(self, pre_kp_3d, gt_kp_3d):
        return ((pre_kp_3d - gt_kp_3d)**2).sum(-1).sqrt()

    def smooth_betas_loss(self, pred_betas):
        pred_betas_diff = (pred_betas.reshape(-1, 16, 10))[:, 1:] - (pred_betas.reshape(-1, 16, 10)[:, :-1])
        return torch.mean(pred_betas_diff.abs())

    def smooth_hand_pose_loss(self, pred_hand_pose):
        pred_hand_pose_diff = (pred_hand_pose.reshape(-1, 16, 15, 3, 3))[:, 1:] - (pred_hand_pose.reshape(-1, 16, 15, 3, 3))[:, :-1]
        return torch.mean(torch.sum(pred_hand_pose_diff.abs()), dim=0)

    def smooth_global_orient_loss(self, pred_global_orient):
        pred_global_orient_diff = (pred_global_orient.reshape(-1, 16, 1, 3, 3))[:, 1:] - (pred_global_orient.reshape(-1, 16, 1, 3, 3)[:, :-1])
        return torch.mean(torch.sum(pred_global_orient_diff.abs()), dim=0)

    def smooth_kp_3d_loss(self, pred_kp_3d):
        pred_kp_3d_diff =  (pred_kp_3d.reshape(-1, 16, 21, 3))[:, 1:] - (pred_kp_3d.reshape(-1, 16, 21, 3)[:, :-1])
        return torch.mean(torch.sum(pred_kp_3d_diff.abs()), dim=0)

    # def twist_loss(self, pred_hand_pose):
    #
    #     return pred_hand_pose.reshape(-1, 15, 3)[:, [8, 7, 11, 10, 5, 4, 2, 1, 14, 13], 0].abs().sum(-1).mean()


# if __name__ == '__main__':
# #
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 实例化数据集
#     train_dataset = HandPoseDataset(
#         pressure_dir=r'D:\data_collect\filtered_pressure\czh',
#         label_dir=r'D:\data_collect\czh_camera0_GT',
#         file_indices=range(1, 8),
#         sequence_length=16,
#         step=8,
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
#
#     # 实例化模型
#     num_betas = 10
#     num_hand_pose = 45
#     num_global_orient = 3
#     model = ResNetTransformerModel(num_betas, num_hand_pose, num_global_orient)
#     model.to(device)
#
#     # 实例化损失函数
#     criterion = HandParamsLoss()
#
#     # 进行一次前向传播和损失计算
#     for data in train_loader:
#         pressure_data = data['pressure'].to(device)
#         hand_labels = data['hand_label']
#
#         # 正向传播
#         outputs = model(pressure_data)
#
#         # 计算损失
#         loss = criterion(outputs, hand_labels)  # 确保 hand_labels 的格式与 outputs 匹配
#
#         # 打印损失值
#         print(f'Loss: {loss.item():.4f}')
#         break