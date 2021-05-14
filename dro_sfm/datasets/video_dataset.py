
import re
from collections import defaultdict
import os
from numpy.core.defchararray import join

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

def dummy_calibration(w, h):
    fx = w * 1.2
    fy = w * 1.2
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]])


def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.depth', '.h5'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if (ext is None or entry.path.lower().endswith(tuple(ext))):
                files[directory].append(relpath)
    return files

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)

    # assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    if (np.max(depth_png) > 255):
        depth = depth_png.astype(np.float) / 256.
    else:
        depth = depth_png.astype(np.float)
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

########################################################################################################################
#### DATASET
########################################################################################################################

class VideoDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        assert depth_type is None or depth_type == '', \
            'VideoDataset currently does not support depth types'
        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.root_dir = root_dir
        self.split = split

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = strides[0]

        self.files = []
        self.file_tree = read_files(root_dir, ext=(".jpg", ".png", ".bmp", ".jpeg"))
        for k, v in self.file_tree.items():
            file_list = sorted(v)
            files = [fname for fname in file_list if self._has_context(k, fname, file_list)]
            self.files.extend([[k, fname] for fname in files])

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _has_context(self, session, filename, file_list):
        context_paths = self._get_context_file_paths(filename, file_list)
        return all([os.path.exists(os.path.join(self.root_dir, session, f)) for f in context_paths])

    def _get_context_file_paths(self, filename, filelist):
        # fidx = get_idx(filename)
        fidx = filelist.index(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)

        return [filelist[fidx+i] if (0 <= (fidx+i)) and ((fidx+i) < len(filelist)) else 'none' for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        context_paths = self._get_context_file_paths(filename, sorted(self.file_tree[session]))
        return [self._read_rgb_file(session, filename)
                for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        file_path = os.path.join(self.root_dir, session, filename)
        rgb = load_image(file_path)
        return rgb

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        if session == self.root_dir:
            session = ''
        
        rgb = self._read_rgb_file(session, filename)
                        
        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            'rgb': rgb,
            'intrinsics': dummy_calibration(w=rgb.size[0], h=rgb.size[1])
        }

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': None,
            })

        if self.has_context:
            sample['rgb_context'] = \
                self._read_rgb_context_files(session, filename)

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
