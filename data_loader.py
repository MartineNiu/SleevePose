import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HandPoseDataset(Dataset):
    def __init__(self, pressure_dir, label_dir, file_indices, user, sequence_length=8, step=4, reshape_size=(13, 9)):
        self.pressure_dir = pressure_dir
        self.label_dir = label_dir
        self.reshape_size = reshape_size
        self.sequence_length = sequence_length
        self.step = step
        self.user = user

        # 定义valid_frames
        self.valid_frames = {
            "qy": {
                1: (581, 4460), 2: (621, 4360), 3: (311, 4140), 4: (201, 3960), 5: (141, 3810),
                6: (301, 4090), 7: (191, 3990), 8: (161, 4570), 9: (181, 3980), 10: (1911, 5680),
                11: (221, 4020), 12: (211, 3980)
            },
            'twj': {
                1: (431, 4240), 2: (271, 4010), 3: (211, 3970), 4: (191, 4020), 5: (191, 4120),
                6: (281, 4010), 7: (211, 3960), 8: (301, 4710), 9: (201, 4660), 10: (211, 3980),
                11: (281, 4110)
            },
            'lby': {
                1: (571, 4330), 2: (221, 4010), 3: (321, 4110), 4: (211, 4040), 5: (171, 4080),
                6: (161, 3970), 7: (831, 4600), 8: (131, 4470), 9: (151, 4020), 10: (171, 3950)
            },
            'xyf': {
                1: (171, 3990), 2: (971, 4730), 3: (231, 4010), 4: (111, 3960), 5: (131, 3940),
                6: (311, 4080), 7: (171, 3940), 8: (221, 4330), 9: (171, 3910), 10: (301, 4110)
            },
            'yqj': {
                1: (361, 4140), 2: (171, 4000), 3: (231, 3980), 4: (151, 3970), 5: (211, 3980),
                6: (141, 3920), 7: (201, 3950), 8: (161, 4390), 9: (321, 4070), 10: (191, 3870)
            },
            'wq': {
                1: (611, 4330), 2: (301, 4050), 3: (371, 4100), 4: (391, 4090), 5: (491, 4230),
                6: (191, 3960), 7: (301, 4020), 8: (281, 5000), 9: (531, 4340), 10: (521, 4240)
            },
            'dhy': {
                1: (231, 4030), 2: (241, 4000), 3: (151, 3920), 4: (151, 3930), 5: (271, 4020),
                6: (201, 3960), 7: (171, 3910), 8: (171, 4440), 9: (211, 4000), 10: (231, 3980)
            },
            'fcf': {
                1: (181, 3840), 2: (181, 3920), 3: (141, 3960), 4: (171, 3930), 5: (141, 3970),
                6: (251, 4030), 7: (141, 4020), 8: (171, 4440), 9: (191, 3910), 10: (341, 4090),
                11: (191, 3930), 12: (481, 4390)
            },
            'zyk': {
                1: (141, 4030), 2: (171, 4020), 3: (181, 3980), 4: (201, 4000), 5: (241, 4040),
                6: (171, 3900), 7: (191, 3880), 8: (191, 4430), 9: (311, 4001), 10: (221, 3950)
            },
            'ky': {
                1: (411, 4100), 2: (161, 3940), 3: (141, 3940), 4: (131, 3950), 5: (181, 3910),
                6: (221, 3940), 7: (181, 3960), 8: (381, 4720), 9: (321, 4000), 10: (551, 4270)
            },
            'oygw': {
                1: (221, 4010), 2: (231, 4020), 3: (231, 3950), 4: (181, 3960), 5: (161, 3930),
                6: (181, 3880), 7: (341, 4080), 8: (201, 2510), 9: (171, 3950), 10: (171, 3930)
            },
            'wzj': {
                1: (391, 4130), 2: (181, 3970), 3: (341, 4020), 4: (281, 4080), 5: (161, 4150),
                6: (291, 4180), 7: (281, 4050), 8: (181, 3910), 9: (181, 4610), 10: (221, 4650),
                11: (201, 3940)
            },
            'mch': {
                1: (461, 4260), 2: (171, 3950), 3: (351, 4090), 4: (151, 3940), 5: (151, 3910),
                6: (131, 3880), 7: (141, 3900), 8: (171, 4680), 9: (311, 4070), 10: (231, 3910)
            },
            'zch': {
                1: (231, 3960), 2: (191, 4010), 3: (301, 4130), 4: (201, 4020), 6: (151, 3900),
                7: (211, 4000), 8: (161, 4330), 9: (341, 4550), 10: (281, 4040)
            },
            'czh': {
                1: (631, 4560), 2: (251, 4000), 3: (331, 4050), 4: (211, 3990), 5: (291, 4040),
                6: (201, 4010), 7: (251, 3980), 8: (331, 3800), 9: (701, 4810), 10: (121, 1830),
                11: (221, 3950), 12: (271, 3950)
            },
            'wyc': {
                1: (481, 4330), 2: (431, 4220), 3: (251, 4060), 4: (141, 4000), 5: (331, 4130),
                6: (221, 4000), 7: (171, 3930), 8: (221, 4390), 9: (351, 4020), 10: (211, 3950)
            },
            'jwq': {
                1: (491, 4120), 2: (141, 3910), 3: (261, 4050), 4: (341, 4120), 5: (141, 4130),
                6: (111, 3900), 7: (131, 3840), 8: (161, 4820), 9: (191, 3960), 10: (211, 4120)
            },
            'dyk': {
                1: (421, 4270), 2: (101, 3890), 3: (151, 3870), 4: (201, 3940), 5: (121, 3880),
                6: (141, 3880), 7: (121, 3860), 8: (141, 4230), 9: (151, 3850), 10: (131, 3860)
            },
            'wzy': {
                1: (301, 3980), 2: (261, 4090), 3: (301, 4000), 4: (251, 4000), 5: (271, 4220),
                6: (221, 3970), 7: (251, 4140), 8: (191, 4970), 9: (201, 4050), 10: (321, 4480)
            },
            'xft': {
                1: (341, 3990), 2: (171, 3840), 3: (191, 3990), 4: (151, 3900), 5: (171, 3910),
                6: (161, 3920), 7: (161, 3900), 8: (181, 3900), 9: (101, 3320), 10: (171, 3910),
                11: (101, 3900), 12: (161, 3870), 13: (151, 3900)
            },
            'nmt': {
                1: (241, 3980), 2: (91, 3910), 3: (191, 3960), 4: (311, 4090), 5: (151, 3960),
                6: (211, 3980), 7: (161, 3960), 8: (251, 6340), 9: (151, 2390), 10: (111, 3990),
                11: (291, 4060), 12: (291, 4070), 13: (390, 4230), 14: (271, 4020), 15: (211, 3970)
            }
        }[user]

        # 存储所有原始数据
        self.all_data = {}  # {user: {file_idx: {'pressure': array, 'label': dict}}}

        # 存储序列索引信息
        self.sequence_indices = []  # [(user, file_idx, start_idx, end_idx), ...]

        # 加载数据
        self._load_all_data(file_indices)

        print(f"Total sequences: {len(self.sequence_indices)}")

    def _load_all_data(self, file_indices):
        # 初始化用户数据字典
        if self.user not in self.all_data:
            self.all_data[self.user] = {}

        # 加载每个文件的数据
        for idx in file_indices:
            pressure_path = os.path.join(self.pressure_dir, f'fil_{self.user}{idx}.csv')
            label_path = os.path.join(self.label_dir, f'{self.user}_{idx}_pose.npz')

            # 加载压力数据
            raw_pressure = np.loadtxt(pressure_path, delimiter=',').astype(np.float32).reshape(-1, *self.reshape_size)
            raw_pressure_16 = np.zeros(((raw_pressure.shape[0], 16, 16)))
            raw_pressure_16[:, 1:14, 3:12] = raw_pressure
            raw_pressure = raw_pressure_16.astype(np.float32)
            raw_pressure[raw_pressure > 600] = 600
            raw_pressure = raw_pressure / 600

            # 加载标签数据
            label_data = np.load(label_path)

            # 存储数据
            self.all_data[self.user][idx] = {
                'pressure': raw_pressure,
                'label': label_data
            }

            # 获取有效帧范围
            start, end = self.valid_frames[idx]

            # 生成序列索引
            for start_idx in range(start, end - self.sequence_length + 1, self.step):
                end_idx = start_idx + self.sequence_length
                self.sequence_indices.append((self.user, idx, start_idx, end_idx))

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        user, file_idx, start_idx, end_idx = self.sequence_indices[idx]
        data = self.all_data[user][file_idx]

        return {
            'pressure': torch.tensor(data['pressure'][start_idx:end_idx], dtype=torch.float32).to(device),
            'hand_label': {
                'betas': torch.tensor(data['label']['betas'][start_idx:end_idx], dtype=torch.float32).to(device),
                'hand_pose': torch.tensor(data['label']['hand_pose'][start_idx:end_idx], dtype=torch.float32).to(
                    device),
                'global_orient': torch.tensor(data['label']['global_orient'][start_idx:end_idx],
                                              dtype=torch.float32).to(device)
            },
            'translation': torch.tensor(data['label']['transl'][start_idx:end_idx], dtype=torch.float32).to(device),
            'metadata': {
                'user': user,
                'file_idx': file_idx,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        }