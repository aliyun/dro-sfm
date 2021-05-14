
import re
from collections import defaultdict
import os

from torch.utils.data import Dataset
import numpy as np
from dro_sfm.utils.image import load_image
import cv2, IPython
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def dummy_calibration(image):
    return np.array([[5.703422047415297129e+02 , 0.    , 3.200000000000000000e+02],
                     [0.    , 5.703422047415297129e+02 , 2.400000000000000000e+02],
                     [0.    , 0.    , 1.          ]])


########################################################################################################################
#### DATASET
########################################################################################################################

class DemonDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        # assert depth_type is None or depth_type == '', \
        #     'NYUDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 1, \
            'NYUDataset currently only supports stride of 1.'

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.root_dir = root_dir
        self.split = split
        with open(os.path.join(self.root_dir, split), "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            path = os.path.join(root_dir, fname.split()[0])
            # if os.path.exists(path):
            self.paths.append(path)


        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = strides[0]

        self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def __getitem__(self, idx):
        filepath = self.paths[idx]

        image = load_image(os.path.join(filepath, '0000.jpg'))
        depth = np.load(os.path.join(filepath, '0000.npy'))

        rgb_contexts = [load_image(os.path.join(filepath, '0001.jpg'))]

        poses = [p.reshape((3, 4)) for p in np.genfromtxt(os.path.join(filepath, 'poses.txt')).astype(np.float64)]
        pos0 = np.zeros((4, 4))
        pos1 = np.zeros((4, 4))
        pos0[:3, :] = poses[0]
        pos0[3, 3] = 1.
        pos1[:3, :] = poses[1]
        pos1[3, 3] = 1.
        pos = np.matmul(pos1, np.linalg.inv(pos0))
        # pos = np.matmul(np.linalg.inv(pos1), pos0)
        pose_context = [pos.astype(np.float32)]

        intr = np.genfromtxt(os.path.join(filepath, 'cam.txt'))

        sample = {
            'idx': idx,
            'filename': '%s' % (filepath.split('/')[-1]),
            'rgb': image,
            'depth': depth,
            'pose_context': pose_context,
            'intrinsics': intr
        }

        if self.has_context:
            sample['rgb_context'] = rgb_contexts

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
