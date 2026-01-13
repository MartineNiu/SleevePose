import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from data_loader import HandPoseDataset
from model import ResNetTransformerModel
import cv2
import os
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
from geometry.geometry import batch_rodrigues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hand_model = smplx.MANOLayer(data_dir='data',
                             model_path=os.path.join('data', 'mano/'),
                             gender='neutral',
                             create_body_pose=False).to(device)

def process_model_output(model_outputs, translation):
    outputs = {}
    outputs['betas'] = model_outputs['betas'].reshape(-1, 10)
    outputs['hand_pose'] = batch_rodrigues(model_outputs['hand_pose'].reshape(-1, 3)).reshape(-1, 15, 3, 3)
    outputs['global_orient'] = batch_rodrigues(model_outputs['global_orient'].reshape(-1, 3)).reshape(-1, 1, 3, 3)
    outputs['translation'] = translation.reshape(-1, 3)

    return outputs

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


def Model_test(model_path, device='cuda'):
    num_betas = 10
    num_hand_pose = 45
    num_global_orient = 3


    # 模型初始化与权重加载
    model = ResNetTransformerModel(
        num_betas=num_betas,
        num_hand_pose=num_hand_pose,
        num_global_orient=num_global_orient
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_set = {
        # 'qy': [8, 9, 10],
        # 'lby': [8, 9, 10],
        # 'xyf': [8, 9, 10],
        # 'yqj': [8, 9, 10],
        # 'wq': [8, 9, 10],
        # 'dhy': [8, 9, 10],
        # 'fcf': [8, 9, 10, 12],
        # 'zyk': [8, 9, 10],
        # 'ky': [8, 9, 10],
        # 'oygw': [8, 9, 10],
        # 'wzj': [9, 10, 11],
        # 'zch': [8, 9, 10],
        # 'wyc': [8, 9, 10],
        # 'jwq': [8, 9, 10],
        # 'dyk': [8, 9, 10],
        # 'wzy': [8, 9, 10],
        'xft': [8, 9, 10, 13],
        # 'nmt': [8, 11, 13, 15],
    }

    test_loaders = {}
    for user, file_indices in test_set.items():
        for file_index in file_indices:
            dataset = HandPoseDataset(
                pressure_dir=f'/workspace/nmt/Pressure_DATA/filtered_pressure/{user}',
                label_dir=f'/workspace/nmt/camera0_GT/{user}',
                file_indices=[file_index],
                user=user,
                sequence_length=16,
                step=16
            )
            test_loaders[(user, file_index)] = DataLoader(dataset, batch_size=16, shuffle=False)


    with torch.no_grad():
        final_outputs = {}
        mpjpe = []

        for (user, file_index), loader in test_loaders.items():
            print(f'Testing {user}/{file_index}')

            final_outputs = {}

            for data in loader:
                # pressure_data = data['pressure']
                # hand_label = data['hand_label']
                # translation = data['translation']
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

                # print(hand_label['hand_pose'].reshape(256, -1)[:, -6:], model_outputs['hand_pose'][:, -6:])

                gt =  process_model_output(hand_label, translation)
                outputs = process_model_output(model_outputs, translation)
                for key in outputs.keys():
                    if key not in final_outputs:
                        final_outputs[key] = outputs[key]
                    else:
                        final_outputs[key] = torch.cat((final_outputs[key], outputs[key]))

                with torch.no_grad():
                    mano_output = hand_model(global_orient=outputs['global_orient'],
                                             hand_pose=outputs['hand_pose'],
                                             betas=outputs['betas'],
                                             transl=outputs['translation'],
                                             pose2rot=False)
                    joints = mano_output['joints']
                    vertices = mano_output['vertices']
                    outputs['kp_3d'] = mano_joints_with_fingertips(joints, vertices)

                with torch.no_grad():
                    mano_output = hand_model(global_orient=gt['global_orient'],
                                             hand_pose=gt['hand_pose'],
                                             betas=gt['betas'],
                                             transl=gt['translation'],
                                             pose2rot=False)
                    joints = mano_output['joints']
                    vertices = mano_output['vertices']
                    gt['kp_3d'] = mano_joints_with_fingertips(joints, vertices)

                twist_angle = model_outputs['hand_pose'].clone().detach().cpu().numpy()
                # print('train:', twist_angle[:, [21, 24]].abs().mean(0))

            mpjpe.append(((outputs['kp_3d'] - gt['kp_3d'])**2).sum(-1).sqrt().cpu().detach().numpy())
            # return final_outputs

            # np.savez(f'/workspace/nmt/Pressure_HPE/test_result/model_test_result/2_Cross_validation_1_4/{user}_{file_index}_predictions.npz', **{key: value.cpu().numpy() for key, value in final_outputs.items()})

            # print(np.concatenate(mpjpe, axis=0).mean())
            save_dir = '/workspace/nmt/Pressure_HPE/test_result/model_test_result/xft_train_with_offset'

            # 创建目录（如果不存在）
            os.makedirs(save_dir, exist_ok=True)

            # 然后保存文件
            np.savez(
                os.path.join(save_dir, f'{user}_{file_index}_predictions.npz'),
                **{key: value.cpu().numpy() for key, value in final_outputs.items()}
            )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model_test(model_path=r'/workspace/nmt/Pressure_HPE/test_result/models/xft_with_offset.pth', device=device)