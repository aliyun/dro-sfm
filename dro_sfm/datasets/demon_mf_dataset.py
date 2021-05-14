
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
            view3_rgb = os.path.join(path, "0002.jpg")
            view3_depth = os.path.join(path, "0002.npy")
            if not (forward_context == 1 and back_context == 1):      
                if os.path.exists(view3_rgb) and os.path.exists(view3_depth):
                    self.paths.append((path, True))
                else:
                    self.paths.append((path, False))
            else:
                if os.path.exists(view3_rgb) and os.path.exists(view3_depth):
                    self.paths.append((path, True))
                    
        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = (self.backward_context > 0 ) or (self.forward_context > 0)
        self.strides = strides[0]

        self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def _get_view2(self, filepath):
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
        
        pos01 = np.matmul(pos1, np.linalg.inv(pos0))
        pose_context = [pos01.astype(np.float32)]
    
        return image, depth, rgb_contexts, pose_context
    
    def _get_view3_dummy(self, filepath):
        image = load_image(os.path.join(filepath, '0000.jpg'))
        depth = np.load(os.path.join(filepath, '0000.npy'))
        rgb_contexts = [load_image(os.path.join(filepath, '0001.jpg')), load_image(os.path.join(filepath, '0001.jpg'))]

        poses = [p.reshape((3, 4)) for p in np.genfromtxt(os.path.join(filepath, 'poses.txt')).astype(np.float64)]
        pos0 = np.zeros((4, 4))
        pos1 = np.zeros((4, 4))
        pos0[:3, :] = poses[0]
        pos0[3, 3] = 1.
        pos1[:3, :] = poses[1]
        pos1[3, 3] = 1.
        
        pos01 = np.matmul(pos1, np.linalg.inv(pos0))
        pose_context = [pos01.astype(np.float32), pos01.astype(np.float32)]
    
        return image, depth, rgb_contexts, pose_context
    
    def _get_view3(self, filepath):
        image = load_image(os.path.join(filepath, '0001.jpg'))
        depth = np.load(os.path.join(filepath, '0001.npy'))
        rgb_contexts = [load_image(os.path.join(filepath, '0000.jpg')), load_image(os.path.join(filepath, '0002.jpg'))]

        poses = [p.reshape((3, 4)) for p in np.genfromtxt(os.path.join(filepath, 'poses.txt')).astype(np.float64)]
    
        pos0 = np.eye(4)
        pos1 = np.eye(4)
        pos2 = np.eye(4)
        pos0[:3, :] = poses[0]
        pos1[:3, :] = poses[1]
        pos2[:3, :] = poses[2]
        
        pos10 = np.matmul(pos0, np.linalg.inv(pos1))
        pos12 = np.matmul(pos2, np.linalg.inv(pos1))
        pose_context = [pos10.astype(np.float32), pos12.astype(np.float32)]      
    
        return image, depth, rgb_contexts, pose_context


    def __getitem__(self, idx):
        filepath, mf = self.paths[idx]

        if self.forward_context == 1 and self.backward_context == 0:
            image, depth, rgb_contexts, pose_context = self._get_view2(filepath)
            
        elif self.forward_context == 1 and self.backward_context == 1:
            image, depth, rgb_contexts, pose_context = self._get_view3(filepath)

        elif self.forward_context == 1 and self.backward_context == -1:
            if np.random.random() > 0.5 and mf:
                image, depth, rgb_contexts, pose_context = self._get_view3(filepath)
            else:
                image, depth, rgb_contexts, pose_context = self._get_view3_dummy(filepath)
        else:
            raise NotImplementedError
        
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
