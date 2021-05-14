
import re
from collections import defaultdict
import os

from torch.utils.data import Dataset
import numpy as np
from dro_sfm.utils.image import load_image
import IPython, cv2
########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.png', '.jpg', '.jpeg', '.ppm'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            color_path = os.path.join(entry.path, 'color')
            d_files = read_files(color_path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath + '/color'] = d_files[color_path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                pose_path = entry.path.replace('color', 'pose').replace('.jpg', '.txt')
                pose = np.genfromtxt(pose_path)
                if not np.isinf(pose).any():
                    files[directory].append(relpath)
    return files

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)

    depth = depth_png.astype(np.float) / 1000.
    # assert (np.max(depth_png) > 1000.), 'Wrong .png depth file'
    # if (np.max(depth_png) > 1000.):
    #     depth = depth_png.astype(np.float) / 1000.
    # else:
    #     depth = depth_png.astype(np.float)
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

########################################################################################################################
#### DATASET
########################################################################################################################

class ScannetBADataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        # assert depth_type is None or depth_type == '', \
        #     'ImageDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 1, \
            'ImageDataset currently only supports stride of 1.'

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.root_dir = root_dir
        self.split = split

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = (self.backward_context > 0 ) or (self.forward_context > 0)
        self.strides = strides[0]

        self.files = []

        source = ''
        if source == 'Folder':
            # ================= load from folder ====================
            # test split
            with open(os.path.join(os.path.dirname(self.root_dir), "splits/test_split.txt"), "r") as f:
                test_data = f.readlines()
            test_scenes = [d.split('/')[0] for d in test_data]
            
            self.file_tree = read_files(root_dir)
            # remove test scenes
            for scene in test_scenes:
                key = scene + '/color'
                if key in self.file_tree:
                    self.file_tree.pop(key, None)
                    print('remove test scene:', scene)
            # sort
            for k in self.file_tree:
                self.file_tree[k].sort(key=lambda x: int(x.split('.jpg')[0]))
            # save train list
            fo = open(os.path.join(os.path.dirname(self.root_dir), "splits/train_all_list.txt"), "w")
            for k, v in self.file_tree.items():
                for data in v:
                    fo.write(k + ' ' + data + '\n')
            fo.close()
        else:
            # =================== load from txt ====================
            self.file_tree = defaultdict(list)
            with open(os.path.join(os.path.dirname(self.root_dir), self.split), "r") as f:
                split_data = f.readlines()
            for data in split_data:
                scene, filename = data.split()
                self.file_tree[scene].append(filename)
        
        Create_batrain = True
        self.bafile_tree = defaultdict(list)
        self.bacontext1_tree = defaultdict(list)
        self.bacontext2_tree = defaultdict(list)
        self.bacontext3_tree = defaultdict(list)
        self.bacontext4_tree = defaultdict(list)
        if Create_batrain:
            # =================== create train_balist.txt file ====================
            with open(os.path.join(os.path.dirname(self.root_dir), "splits/banet_train.txt"), "r") as f:
                banet_data = f.readlines()
            banet_target = banet_data[::7]
            banet_context = banet_data[1:][::7]

            for d0, d1 in zip(banet_target, banet_context):
                sce = d0.split('/')[3] + '/color'
                id0 = d0.split('/')[-1].split('.')[0].split('frame-')[-1] + '.jpg'
                id1 = d1.split('/')[-1].split('.')[0].split('frame-')[-1] + '.jpg'
                if id0 not in self.bafile_tree[sce]:
                    self.bafile_tree[sce].append(id0)
                    self.bacontext1_tree[sce].append(id1)
                    if int(id1.split('.')[0]) > int(id0.split('.')[0]):
                        id2 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 5)
                        id3 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 5)
                        id4 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 10)
                    else:
                        id2 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 5)
                        id3 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 5)
                        id4 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 10)
                    self.bacontext2_tree[sce].append(id2)
                    self.bacontext3_tree[sce].append(id3)
                    self.bacontext4_tree[sce].append(id4)

            for k, v in self.bafile_tree.items():
                files = []
                for i, fname in enumerate(v):
                    if k in self.file_tree and self._has_context(k, fname, self.file_tree[k], i):
                        files.append(fname)
                self.files.extend([[k, fname] for fname in files])

            # save train list
            fo = open(os.path.join(os.path.dirname(self.root_dir), "splits/train_balist_save.txt"), "w")
            for session, fname in self.files:
                idx = self.bafile_tree[session].index(fname)
                context1 = self.bacontext1_tree[session][idx]
                context2 = self.bacontext2_tree[session][idx]
                context3 = self.bacontext3_tree[session][idx]
                context4 = self.bacontext4_tree[session][idx]
                fo.write(session + ' ' + fname + ' ' + context1 + ' ' + context2 + ' '
                         + context3 + ' ' + context4 + '\n')
            fo.close()

        else:
            # =================== load from train_balist.txt file ====================
            with open(self.root_dir + "/splits/train_balist.txt", "r") as f:
                split_data = f.readlines()
            for data in split_data:
                scene, filename, context1, context2, context3, context4 = data.split()
                self.bafile_tree[scene].append(filename)
                self.bacontext1_tree[scene].append(context1)
                self.bacontext2_tree[scene].append(context2)
                self.bacontext3_tree[scene].append(context3)
                self.bacontext4_tree[scene].append(context4)

            for k, v in self.bafile_tree.items():
                files = []
                for i, fname in enumerate(v):
                    files.append(fname)
                self.files.extend([[k, fname] for fname in files])

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def _has_context(self, session, filename, file_list, idx):
        context_paths = [self.bacontext1_tree[session][idx], self.bacontext2_tree[session][idx],
                         self.bacontext3_tree[session][idx], self.bacontext4_tree[session][idx]]
        return all([f in file_list for f in context_paths])

    def _get_context_file_paths(self, filename, filelist):
        # fidx = get_idx(filename)
        fidx = filelist.index(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        return [filelist[fidx+i] if 0 <= fidx+i < len(filelist) else 'none' for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        context_paths = self._get_context_file_paths(filename, self.file_tree[session])
        
        return [load_image(os.path.join(self.root_dir, session, filename))
                for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        return load_image(os.path.join(self.root_dir, session, filename))

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        depth_file = image_file.replace('color', 'depth').replace('image', 'depth')
        depth_file = depth_file.replace('jpg', 'png')
        return depth_file

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        image = self._read_rgb_file(session, filename)

        if self.with_depth:
            depth = self._read_depth(self._get_depth_file(os.path.join(self.root_dir, session, filename)))
            resized_depth = cv2.resize(depth, image.size, interpolation = cv2.INTER_NEAREST)

        intr_path = os.path.join(self.root_dir, session, filename).split('color')[0] + 'intrinsic/intrinsic_color.txt'
        intr = np.genfromtxt(intr_path)[:3, :3]

        ba_idx = self.bafile_tree[session].index(filename)

        if self.forward_context == 2 and self.backward_context == 2:
            context_paths = [os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx]),
                             os.path.join(self.root_dir, session, self.bacontext2_tree[session][ba_idx]),
                             os.path.join(self.root_dir, session, self.bacontext3_tree[session][ba_idx]),
                             os.path.join(self.root_dir, session, self.bacontext4_tree[session][ba_idx])]
        elif self.forward_context == 1 and self.backward_context == 1:
            context_paths = [os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx]),
                             os.path.join(self.root_dir, session, self.bacontext2_tree[session][ba_idx])]
        elif self.forward_context == 1 and self.backward_context == 0:
            context_paths = [os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx])]
        elif self.forward_context == 1 and self.backward_context == -1:
            if np.random.random() < 0.5:
                # !!!repeat!!!
                context_paths = [os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx]), 
                                 os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx])]
            else:
                context_paths = [os.path.join(self.root_dir, session, self.bacontext1_tree[session][ba_idx]),
                                 os.path.join(self.root_dir, session, self.bacontext2_tree[session][ba_idx])]  
        else:
            raise NotImplementedError

        context_images = [load_image(os.path.join(self.root_dir, session, filename))
                                for filename in context_paths]
        pose_path = os.path.join(self.root_dir, session, filename).replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.genfromtxt(pose_path)
        context_pose_paths = [os.path.join(self.root_dir, session, x).replace('color', 'pose').
                                replace('.jpg', '.txt') for x in context_paths]
        context_poses = [np.genfromtxt(x) for x in context_pose_paths]

        #rel_poses = [np.matmul(x, np.linalg.inv(pose)).astype(np.float32) for x in context_poses]
        rel_poses = [np.matmul(np.linalg.inv(x), pose).astype(np.float32) for x in context_poses]

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session.split('/')[0], os.path.splitext(filename)[0]),
            'rgb': image,
            'intrinsics': intr,
            'pose_context': rel_poses
        }

        # print(filename, context_paths)

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': resized_depth,
            })

        if self.has_context:
            sample['rgb_context'] = context_images

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
