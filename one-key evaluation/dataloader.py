from torch.utils import data
import os
import numpy as np
import open3d as o3d
import torch


class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name in lst_pred and name.endswith('.ply'):
                lst.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))

    def __getitem__(self, item):
        pred = o3d.io.read_point_cloud(self.image_path[item])
        gt = o3d.io.read_point_cloud(self.label_path[item])
        pred, gt = torch.tensor(np.array(pred.colors))[..., 0], torch.tensor(np.array(gt.colors))[..., 0]

        assert pred.size() == gt.size(), "Prediction must have the same size as groundtruth!"
        gt[gt != 0] = 1
        return pred, gt

    def __len__(self):
        return len(self.image_path)
