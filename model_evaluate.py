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
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_set = {
        'qy': [8, 9, 10],
        'lby': [8, 9, 10],
        'xyf': [8, 9, 10],
        'yqj': [8, 9, 10],
        'wq': [8, 9, 10],
        'dhy': [8, 9, 10],
        'fcf': [8, 9, 10, 12],# 12组是站立
        'zyk': [8, 9, 10],
        'ky': [8, 9, 10],
        'oygw': [8, 9, 10],
        'wzj': [9, 10, 11],
        'zch': [8, 9, 10],
        'wyc': [8, 9, 10],
        'jwq': [8, 9, 10],
        'dyk': [8, 9, 10],
        'wzy': [8, 9, 10],
        'xft': [8, 9, 10, 13],# 8.9分别是常规和自由（已经拆开，不用处理）
        'nmt': [8, 11, 13, 15], #15组是站立
    }

valid_frames = {
            "qy": {
                1: (581, 4460), 2: (621, 4360), 3: (311, 4140), 4: (201, 3960), 5: (141, 3810),
                6: (301, 4090), 7: (191, 3990), 8: (161, 4570), 9: (181, 3980), 10: (1911, 5680),
                11: (221, 4020), 12: (211, 3980)
            },
            'twj':{
                1: (431, 4240), 2: (271, 4010), 3: (211, 3970), 4: (191, 4020), 5: (191, 4120),
                6: (281, 4010), 7: (211, 3960), 8: (301, 4710), 9: (201, 4660), 10: (211, 3980),
                11: (281, 4110)
            },
            'lby':{
                1: (571, 4330), 2: (221, 4010), 3: (321, 4110), 4: (211, 4040), 5: (171, 4080),
                6: (161, 3970), 7: (831, 4600), 8: (131, 4470), 9: (151, 4020), 10: (171, 3950)
            },
            'xyf':{
                1: (171, 3990), 2: (971, 4730), 3: (231, 4010), 4: (111, 3960), 5: (131, 3940),
                6: (311, 4080), 7: (171, 3940), 8: (221, 4330), 9: (171, 3910), 10: (301, 4110)
            },
            'yqj':{
                1: (361, 4140), 2: (171, 4000), 3: (231, 3980), 4: (151, 3970), 5: (211, 3980),
                6: (141, 3920), 7: (201, 3950), 8: (161, 4390), 9: (321, 4070), 10: (191, 3870)
            },
            'wq':{
                1: (611, 4330), 2: (301, 4050), 3: (371, 4100), 4: (391, 4090), 5: (491, 4230),
                6: (191, 3960), 7: (301, 4020), 8: (281, 5000), 9: (531, 4340), 10: (521, 4240)
            },
            'dhy':{
                1: (231, 4030), 2: (241, 4000), 3: (151, 3920), 4: (151, 3930), 5: (271, 4020),
                6: (201, 3960), 7: (171, 3910), 8: (171, 4440), 9: (211, 4000), 10: (231, 3980)
            },
            'fcf':{
                1: (181, 3840), 2: (181, 3920), 3: (141, 3960), 4: (171, 3930), 5: (141, 3970),
                6: (251, 4030), 7: (141, 4020), 8: (171, 4440), 9: (191, 3910), 10: (341, 4090),
                11: (191, 3930), 12: (481, 4390)
            },
            'zyk':{
                1: (141, 4030), 2: (171, 4020), 3: (181, 3980), 4: (201, 4000), 5: (241, 4040),
                6: (171, 3900), 7: (191, 3880), 8: (191, 4430), 9: (311, 4001), 10: (221, 3950)
            },
            'ky':{
                1: (411, 4100), 2: (161, 3940), 3: (141, 3940), 4: (131, 3950), 5: (181, 3910),
                6: (221, 3940), 7: (181, 3960), 8: (381, 4720), 9: (321, 4000), 10: (551, 4270)
            },
            'oygw':{
                1: (221, 4010), 2: (231, 4020), 3: (231, 3950), 4: (181, 3960), 5: (161, 3930),
                6: (181, 3880), 7: (341, 4080), 8: (201, 2510), 9: (171, 3950), 10: (171, 3930)
            },
            'wzj':{
                1: (391, 4130), 2: (181, 3970), 3: (341, 4020), 4: (281, 4080), 5: (161, 4150),
                6: (291, 4180), 7: (281, 4050), 8: (181, 3910), 9: (181, 4610), 10: (221, 4650),
                11: (201, 3940)
            },
            'mch':{
                1: (461, 4260), 2: (171, 3950), 3: (351, 4090), 4: (151, 3940), 5: (151, 3910),
                6: (131, 3880), 7: (141, 3900), 8: (171, 4680), 9: (311, 4070), 10: (231, 3910)
            },
            'zch':{
                1: (231, 3960), 2: (191, 4010), 3: (301, 4130), 4: (201, 4020), 6: (151, 3900),
                7: (211, 4000), 8: (161, 4330), 9: (341, 4550), 10: (281, 4040)
            },
            'czh':{
                1: (631, 4560), 2: (251, 4000), 3: (331, 4050), 4: (211, 3990), 5: (291, 4040),
                6: (201, 4010), 7: (251, 3980), 8: (331, 3800), 9: (701, 4810), 10: (121, 1830),
                11: (221, 3950), 12: (271, 3950)
            },
            'wyc':{
                1: (481, 4330), 2: (431, 4220), 3: (251, 4060), 4: (141, 4000), 5: (331, 4130),
                6: (221, 4000), 7: (171, 3930), 8: (221, 4390), 9: (351, 4020), 10: (211, 3950)
            },
            'jwq':{
                1: (491, 4120), 2: (141, 3910), 3: (261, 4050), 4: (341, 4120), 5: (141, 4130),
                6: (111, 3900), 7: (131, 3840), 8: (161, 4820), 9: (191, 3960), 10: (211, 4120)
            },
            'dyk':{
                1: (421, 4270), 2: (101, 3890), 3: (151, 3870), 4: (201, 3940), 5: (121, 3880),
                6: (141, 3880), 7: (121, 3860), 8: (141, 4230), 9: (151, 3850), 10: (131, 3860)
            },
            'wzy':{
                1: (301, 3980), 2: (261, 4090), 3: (301, 4000), 4: (251, 4000), 5: (271, 4220),
                6: (221, 3970), 7: (251, 4140), 8: (191, 4970), 9: (201, 4050), 10: (321, 4480)
            },
            'xft':{
                1: (341, 3990), 2: (171, 3840), 3: (191, 3990), 4: (151, 3900), 5: (171, 3910),
                6: (161, 3920), 7: (161, 3900), 8: (181, 3900), 9: (101, 3320), 10: (171, 3910),
                11: (101, 3900), 12: (161, 3870), 13: (151, 3900)
            },
            'nmt':{
                1: (241, 3980), 2: (91, 3910), 3: (191, 3960), 4: (311, 4090), 5: (151, 3960),
                6: (211, 3980), 7: (161, 3960), 8: (251, 6340), 9: (151, 2390), 10: (111, 3990),
                11: (291, 4060), 12: (291, 4070), 13: (390, 4230), 14: (271, 4020), 15: (211, 3970)
            }
        }


def move_to_device(data_dict, device):
    return {key: torch.from_numpy(value).to(device) for key, value in data_dict.items()}


hand_model = smplx.MANOLayer(data_dir='data',
                             model_path=os.path.join('data', 'mano/'),
                             gender='neutral',
                             create_body_pose=False).to(device)


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


def process_GT(model_outputs):
    outputs = {}
    outputs['betas'] = model_outputs['betas'].reshape(-1, 10)
    outputs['hand_pose'] = batch_rodrigues(model_outputs['hand_pose'].reshape(-1, 3)).reshape(-1, 15, 3, 3)
    outputs['global_orient'] = batch_rodrigues(model_outputs['global_orient'].reshape(-1, 3)).reshape(-1, 1, 3, 3)
    outputs['translation'] = model_outputs['transl'].reshape(-1, 3)
    return outputs

def gen_mano(outputs):
    # n_frames = outputs['global_orient'].shape[0]
    #
    # # 创建全零的global_orient
    # zero_global_orient = torch.zeros((n_frames, 3), device=outputs['global_orient'].device)
    #
    # # 将零矩阵转换为旋转矩阵格式
    # zero_global_orient = batch_rodrigues(zero_global_orient).reshape(-1, 1, 3, 3)

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

mpjpe_results = []

for user, file_indices in test_set.items():
    for file_idx in file_indices:
        # 加载数据
        GT_data = dict(np.load(f'/workspace/nmt/camera0_GT/{user}/{user}_{file_idx}_pose.npz'))
        Prediction_data = dict(np.load(f'/workspace/nmt/Pressure_HPE/test_result/model_test_result/20251021_single_person_model_prediction/{user}_{file_idx}_predictions.npz'))

        # 移动数据到设备
        GT_data = move_to_device(GT_data, device)
        Prediction_data = move_to_device(Prediction_data, device)

        # 处理GT数据
        GT_data = process_GT(GT_data)

        # 生成MANO输出
        GT_data = gen_mano(GT_data)
        Prediction_data = gen_mano(Prediction_data)

        # 提取关节数据
        gt_kp_3d = GT_data['kp_3d']
        pred_kp_3d = Prediction_data['kp_3d']

        # 获取初始对齐帧
        start_frame = valid_frames[user][file_idx][0]  # 取出初始帧的左边值,用于对齐标签和预测值（只有有效帧）
        # 特殊处理需要拆分的组
        if user == 'xft' and file_idx == 9:
            # xft的第9组直接作为自由动作（第0组）
            aligned_gt_kp_3d = gt_kp_3d[start_frame:start_frame + pred_kp_3d.shape[0]]
            mpjpe = ((pred_kp_3d - aligned_gt_kp_3d) ** 2).sum(dim=-1).sqrt()
            mean_mpjpe_per_frame = mpjpe.mean(dim=1)
            average_mpjpe = mean_mpjpe_per_frame.mean()
            std_mpjpe = mean_mpjpe_per_frame.std()
            mpjpe_results.append((user, 0, average_mpjpe.item(), std_mpjpe.item()))  # 将组号改为0
            # np.save(f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/Prediction/Pre_{user}_0_3d_joints.npy",pred_kp_3d.cpu().detach().numpy())
            # np.save(f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/GT/GT_{user}_0_3d_joints.npy", aligned_gt_kp_3d.cpu().detach().numpy())
            # data1 = np.load(f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_all_joint/GT/GT_{user}_0_3d_joints.npy")
            # data2 = np.load(f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_all_joint/Prediction/Pre_{user}_0_3d_joints.npy")

        elif ((user in ['twj', 'wzj'] and file_idx == 9) or
              (user not in ['twj', 'wzj', 'xft'] and file_idx == 8)):
            # 需要拆分的组（常规动作和自由动作）

            # 确定常规动作的帧数
            regular_frames = 1900 if user == 'oygw' else 3400

            # 处理常规动作部分
            aligned_gt_kp_3d = gt_kp_3d[start_frame:start_frame + regular_frames]
            mpjpe = ((pred_kp_3d[:regular_frames] - aligned_gt_kp_3d) ** 2).sum(dim=-1).sqrt()
            mean_mpjpe_per_frame = mpjpe.mean(dim=1)
            average_mpjpe = mean_mpjpe_per_frame.mean()
            std_mpjpe = mean_mpjpe_per_frame.std()
            mpjpe_results.append((user, file_idx, average_mpjpe.item(), std_mpjpe.item()))  # 常规动作保持原组号


            # 处理自由动作部分
            aligned_gt_kp_3d = gt_kp_3d[start_frame + regular_frames:start_frame + pred_kp_3d.shape[0]]
            mpjpe = ((pred_kp_3d[regular_frames:] - aligned_gt_kp_3d) ** 2).sum(dim=-1).sqrt()
            mean_mpjpe_per_frame = mpjpe.mean(dim=1)
            average_mpjpe = mean_mpjpe_per_frame.mean()
            std_mpjpe = mean_mpjpe_per_frame.std()
            mpjpe_results.append((user, 0, average_mpjpe.item(), std_mpjpe.item()))  # 自由动作使用0组号
            # np.save(
            #     f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/Prediction/Pre_{user}_0_3d_joints.npy",
            #     pred_kp_3d.cpu().detach().numpy())
            # np.save(
            #     f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/GT/GT_{user}_0_3d_joints.npy",
            #     aligned_gt_kp_3d.cpu().detach().numpy())

        else:
            # 其他组正常处理
            aligned_gt_kp_3d = gt_kp_3d[start_frame:start_frame + pred_kp_3d.shape[0]]
            mpjpe = ((pred_kp_3d - aligned_gt_kp_3d) ** 2).sum(dim=-1).sqrt()
            mean_mpjpe_per_frame = mpjpe.mean(dim=1)
            average_mpjpe = mean_mpjpe_per_frame.mean()
            std_mpjpe = mean_mpjpe_per_frame.std()
            mpjpe_results.append((user, file_idx, average_mpjpe.item(), std_mpjpe.item()))
            # np.save(
            #     f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/Prediction/Pre_{user}_{file_idx}_3d_joints.npy",
            #     pred_kp_3d.cpu().detach().numpy())
            # np.save(
            #     f"/workspace/nmt/Pressure_HPE/test_result/model_evalute_3d_joint/18user_Cross_Val_joint/GT/GT_{user}_{file_idx}_3d_joints.npy",
            #     aligned_gt_kp_3d.cpu().detach().numpy())

    # 输出所有MPJPE结果
for user, file_idx, error, std_dev in mpjpe_results:
    print(f"受试者: {user}, 组号: {file_idx}, MPJPE: {error:.4f}, Std: {std_dev:.4f}")

    # 定义颜色映射
color_map = {
    'Instructed Gesture':  '#DDE554',  #
    'Free Gesture': '#88CF73', #
    'Clothing cover': '#39B086', #
    'Offset sleeve': '#008E8A', #
    'Stand up':  '#176A7A',
}
colors = ['#DDE554','#DFE662', '#88CF73', '#39B086', '#008E8A', '#176A7A','#2E4857']

# 定义动作类型映射
motion_type_map = {
    'Instructed Gesture': lambda user, idx: (user in ['twj', 'wzj'] and idx == 9) or (
                user not in ['twj', 'wzj'] and idx == 8),
    'Free Gesture': lambda user, idx: idx == 0,
    'Clothing cover': lambda user, idx: (user in ['twj', 'wzj', 'xft'] and idx == 10) or (user == 'nmt' and idx == 11) or (
                user not in ['twj', 'wzj', 'nmt'] and idx == 9),
    'Offset sleeve': lambda user, idx: (user in ['twj', 'wzj'] and idx == 11) or (user in ['nmt', 'xft'] and idx == 13) or (
                user not in ['twj', 'wzj', 'nmt'] and idx == 10),
    'Stand up': lambda user, idx: (user == 'fcf' and idx == 12) or (user == 'nmt' and idx == 15)
}

# 重新组织数据
organized_data = {}
for user, file_idx, error, std_dev in mpjpe_results:
    if user not in organized_data:
        organized_data[user] = {}

    # 确定动作类型
    for motion_type, condition in motion_type_map.items():
        if condition(user, file_idx):
            organized_data[user][motion_type] = error * 1000  # 转换为毫米

motion_type_averages = {}
motion_type_errors = {}  # 存储每种动作类型的所有误差值

# 收集每种动作类型的所有误差值
for user, data in organized_data.items():
    for motion_type in color_map.keys():
        if motion_type in data:
            if motion_type not in motion_type_errors:
                motion_type_errors[motion_type] = []
            motion_type_errors[motion_type].append(data[motion_type])

# 计算每种动作类型的平均值和标准差
for motion_type, errors in motion_type_errors.items():
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    motion_type_averages[motion_type] = {
        'mean': mean_error,
        'std': std_error,
        'count': len(errors)
    }
    print(f"\n{motion_type}:")
    print(f"平均误差: {mean_error:.2f} mm")
    print(f"标准差: {std_error:.2f} mm")
    print(f"样本数量: {len(errors)}")
    print("各受试者误差:", [f"{err:.2f}" for err in errors])

#
# # 创建柱状图
# plt.figure(figsize=(7, 5))
#
# # 设置柱子宽度和间隔
# bar_width = 0.6
# user_gap = 0.9
#
#
# legend_label = set()
# # 绘制柱状图
# current_pos = 0
# x_positions = []
# x_labels = []
#
# for user in organized_data.keys():
#     user_data = organized_data[user]
#
#     # 绘制该用户的所有动作类型
#     for i, motion_type in enumerate(color_map.keys()):
#         if motion_type in user_data:
#             label = motion_type if motion_type not in legend_label else ""
#             plt.bar(current_pos + i * bar_width,
#                     user_data[motion_type],
#                     width=bar_width,
#                     color=color_map[motion_type],
#                     label=label)
#             if label:
#                 legend_label.add(motion_type)
#
#             plt.text(current_pos + i * bar_width,  # x坐标
#                      user_data[motion_type],  # y坐标
#                      f'{user_data[motion_type]:.2f}',  # 数值（保留两位小数）
#                      ha='center',  # 水平居中对齐
#                      va='bottom',  # 垂直对齐到底部
#                      fontsize=10)  # 字体大小
#
#     x_positions.append(current_pos + (len(color_map) - 1) * bar_width / 2)
#     x_labels.append(user)
#     current_pos += len(color_map) * bar_width + user_gap
#
# # 设置图表属性
# plt.xlabel('Subjects')
# plt.ylabel('MPJPE (mm)')
# plt.title('MPJPE: nmt-with-global-orient-in-UDModel')
# plt.xticks(x_positions, x_labels)
#
# plt.ylim([0, 70])
# # 添加图例
# plt.legend(title='Motion Types')
#
# # 调整布局
# plt.tight_layout()
# plt.savefig(f'/workspace/nmt/Pressure_HPE/test_result/nmt-with-global-orient-in-UDModel.png')
# # 显示图表
# plt.show()



