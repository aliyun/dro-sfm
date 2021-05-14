
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
    return np.array([[518.85790117450188 , 0.    , 325.58244941119034],
                     [0.    , 519.46961112127485 , 253.73616633400465],
                     [0.    , 0.    , 1.          ]])

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

class NYUDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(5,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        # assert depth_type is None or depth_type == '', \
        #     'NYUDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 5, \
            'NYUDataset currently only supports stride of 5.'

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.root_dir = root_dir
        self.split = split
        # Get depth file list from split
        with open(os.path.join(self.root_dir, split), "r") as f:
            data = f.readlines()
        self.dpaths = []
        for i, fname in enumerate(data):
            self.dpaths.append(fname)
        # get color file list
        with open(os.path.join(self.root_dir, split).replace('depth', 'color'), "r") as f:
            data = f.readlines()
        self.rpaths = []
        for i, fname in enumerate(data):
            self.rpaths.append(fname)
        assert len(self.dpaths) == len(self.rpaths), \
            'len(dpaths) != len(rpaths).'

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = strides[0]

        self.files = []
        self.file_tree = read_files(root_dir)
        self.rgb_tree = read_files(os.path.join(os.path.dirname(root_dir), 'train'), ext=('.jpg', '.ppm'))
        for k, v in self.file_tree.items():
            file_list = sorted(v)
            files = [fname for fname in file_list if self._has_context(k, fname, file_list)]
            self.files.extend([[k, fname] for fname in files])

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def _has_context(self, session, filename, file_list):

        idx_test = int(filename.split('.')[0])
        raw_rgb_path = self.rpaths[idx_test-1]
        raw_rgb_sess, raw_rgb_name = raw_rgb_path.split('\n')[0].split('/')

        if raw_rgb_sess not in self.rgb_tree.keys():
            return False
        raw_rgb_filelist = self.rgb_tree[raw_rgb_sess]

        context_paths = self._get_context_file_paths(raw_rgb_sess, raw_rgb_name, sorted(raw_rgb_filelist), output=True)
        
        return all([f in raw_rgb_filelist for f in context_paths])

    def _get_context_file_paths(self, session, filename, filelist, output=False):
        # fidx = get_idx(filename)
        fidx = filelist.index(filename)
        for stride in range(self.strides, 0, -1):
            idxs = list(np.arange(-self.backward_context * stride, 0, stride)) + \
                   list(np.arange(0, self.forward_context * stride, stride) + stride)
            if all([0 <= fidx+i < len(filelist) for i in idxs]):
                # if output and stride != self.strides:
                #     print(idxs, session, filename)
                return [filelist[fidx+i] for i in idxs]
        if output: print("!!!!!!!!!! file no context !!!!!!!!!!", session, filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
                   list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        if fidx + idxs[0] < 0:
            idxs[0] = idxs[1] - 1
        elif fidx + idxs[1] >= len(filelist):
            idxs[1] = idxs[0] + 1
        return [filelist[fidx+i] if 0 <= fidx+i < len(filelist) else 'none' for i in idxs]
        # idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
        #        list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        # return [filelist[fidx+i] if 0 <= fidx+i < len(filelist) else 'none' for i in idxs]

    def _read_rgb_file(self, session, filename):
        raw_root_dir = os.path.join(os.path.dirname(self.root_dir), 'train')
        return load_image(os.path.join(raw_root_dir, session, filename))

    def _get_color_file(self, session, depth_name):
        """Get the corresponding color file from an depth file."""
        depth_ts = float(depth_name[2:].split('-')[0])
        session_files = self.rd_tree[session]
        idx = session_files.index(depth_name[2:])
        search = 1
        color_files = []
        while True:
            if (0 < idx-search < len(session_files) and session_files[idx-search].endswith('.ppm')):
                color_files.append(session_files[idx-search])
            if (0 < idx+search < len(session_files) and session_files[idx+search].endswith('.ppm')):
                color_files.append(session_files[idx+search])
            if color_files:
                time_diffs = [abs(float(f.split('-')[0]) - depth_ts) for f in color_files]
                color_file = color_files[0]
                td = time_diffs[0]
                for i in range(1, len(time_diffs)):
                    if time_diffs[i] < td:
                        td = time_diffs[i]
                        color_file = color_files[i]
                break
            search += 1
        color_file = 'r-' + color_file
        
        return color_file

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        if session == self.root_dir:
            session = ''

        file_path = os.path.join(self.root_dir, session, filename)
        h5f = h5py.File(file_path, "r")
        rgb = np.array(h5f['rgb'])
        image = np.transpose(rgb, (1, 2, 0))
        # image = rgb[...,::-1]
        image_pil = Image.fromarray(image)
        depth = np.array(h5f['depth'])

        idx_test = int(filename.split('.')[0])
        raw_rgb_path = self.rpaths[idx_test-1]
        raw_rgb_sess, raw_rgb_name = raw_rgb_path.split('\n')[0].split('/')

        # if raw_rgb_sess in 
        rgb_context_paths = self._get_context_file_paths(raw_rgb_sess, raw_rgb_name, sorted(self.rgb_tree[raw_rgb_sess]))

        rgb_contexts = [self._read_rgb_file(raw_rgb_sess, p) for p in rgb_context_paths]

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            'rgb': image_pil,
            'depth': depth,
            'intrinsics': dummy_calibration(image)
        }

        if self.has_context:
            sample['rgb_context'] = rgb_contexts

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
