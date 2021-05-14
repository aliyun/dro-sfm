import sys
import os
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_dir)

import argparse
from ast import parse
import numpy as np
import torch
from glob import glob

from dro_sfm.models.model_wrapper import ModelWrapper
from dro_sfm.utils.horovod import hvd_disable
from dro_sfm.datasets.augmentations import resize_image, to_tensor
from dro_sfm.utils.image import load_image
from dro_sfm.utils.config import parse_test_file
from dro_sfm.utils.load import set_debug
from dro_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from dro_sfm.utils.image import write_image
from scripts import vis
from multiprocessing import Queue
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='dro-sfm inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)', required=True)
    parser.add_argument('--input', type=str, help='Input folder or video', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=True)
    parser.add_argument('--data_type', type=str, choices=['kitti', 'indoor', 'general'], required=True)
    parser.add_argument('--sample_rate', type=int, default=10, help='sample rate', required=True)
    parser.add_argument('--ply_mode', action="store_true", help='vis point cloud')

    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    return args


def get_intrinsics(image_shape_raw, image_shape, data_type):
    if data_type == "kitti":
        intr = np.array([7.215376999999999725e+02, 0.000000000000000000e+00, 6.095593000000000075e+02,
                         0.000000000000000000e+00, 7.215376999999999725e+02, 1.728540000000000134e+02,
                         0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00], dtype=np.float32).reshape(3, 3)
    elif data_type == "indoor":
        intr = np.array([1170.187988, 0.000000, 647.750000, 
                         0.000000, 1170.187988, 483.750000,
                         0.000000, 0.000000, 1.000000], dtype=np.float32).reshape(3, 3)
    else:
        # print("fake intrinsics")
        w, h = image_shape_raw
        fx = w * 1.2
        fy = w * 1.2
        cx = w / 2.0
        cy = h / 2.0
        intr = np.array([[fx, 0., cx],
                         [0., fy, cy],
                         [0., 0., 1.]])
    
    orig_w, orig_h = image_shape_raw
    out_h, out_w = image_shape
    
    # Scale intrinsics
    intr[0] *= out_w / orig_w
    intr[1] *= out_h / orig_h
    
    return  intr
    

@torch.no_grad()
def infer_and_save_pose(input_file_refs, input_file, model_wrapper, image_shape, data_type,
                        save_depth_root, save_vis_root):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file_refs : list(str)
        Reference image file paths
    input_file : str
        Image file for pose estimation
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """    
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    image_raw_wh = load_image(input_file).size
    # Load image
    def process_image(filename):
        image = load_image(filename)
        # Resize and to tensor
        intr = get_intrinsics(image.size, image_shape, data_type) #(3, 3)
        image = resize_image(image, image_shape)
        image = to_tensor(image).unsqueeze(0)
        intr = torch.from_numpy(intr).unsqueeze(0) #(1, 3, 3)
        # Send image to GPU if available
        if torch.cuda.is_available():
            image = image.to('cuda')
            intr = intr.to('cuda')
        return image, intr
    image_ref = [process_image(input_file_ref)[0] for input_file_ref in input_file_refs]
    image, intrinsics = process_image(input_file)

    batch = {'rgb': image, 'rgb_context': image_ref, "intrinsics": intrinsics}
    
    output = model_wrapper(batch)
    inv_depth = output['inv_depths'][0] #(1, 1, h, w)
    depth = inv2depth(inv_depth)[0, 0].detach().cpu().numpy() #(h, w)
    
    pose21 = output['poses'][0].mat[0].detach().cpu().numpy() #(4, 4)  #TODO check: targe -> ref[0]
    pose23 = output['poses'][1].mat[0].detach().cpu().numpy() #(4, 4)  #TODO check: targe -> ref[0]

    vis_depth = viz_inv_depth(inv_depth[0]) * 255
    
    vis_depth_upsample = cv2.resize(vis_depth, image_raw_wh, interpolation=cv2.INTER_LINEAR)
    write_image(os.path.join(save_vis_root, f"{base_name}.jpg"), vis_depth_upsample)
    
    depth_upsample = cv2.resize(depth, image_raw_wh, interpolation=cv2.INTER_NEAREST)
    np.save(os.path.join(save_depth_root, f"{base_name}.npy"), depth_upsample)
    
    return depth, pose21, pose23, intrinsics[0].detach().cpu().numpy(), image[0].permute(1, 2, 0).detach().cpu().numpy() * 255


def start_visualization(queue_g, cinematic=False, render_path=None, clear_points=False, is_kitti=True):
    """ Start interactive slam visualization in seperate process """
    # visualization is a Process Object
    viz = vis.InteractiveViz(queue_g, cinematic, render_path, clear_points, is_kitti=is_kitti)
    viz.start()
    
    return viz

def get_coordinate_xy(coord_shape, device):
    """get meshgride coordinate of x, y and the shape is (B, H, W)"""
    bs, height, width = coord_shape
    y_coord, x_coord = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),\
                                       torch.arange(0, width, dtype=torch.float32, device=device)])
    y_coord, x_coord = y_coord.contiguous(), x_coord.contiguous()
    y_coord, x_coord = y_coord.unsqueeze(0).repeat(bs, 1, 1), \
                       x_coord.unsqueeze(0).repeat(bs, 1, 1)

    return x_coord, y_coord

def reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords):
    """project the reference point cloud into the source view, then project back"""
    intrinsics_ref, extrinsics_ref = ref_pose["intr"], ref_pose["extr"]
    intrinsics_src, extrinsics_src = src_pose["intr"], src_pose["extr"]

    bs, height, width = depth_ref.shape[:3]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = xy_coords  # (B, H, W)
    x_ref, y_ref = x_ref.view([bs, 1, -1]), y_ref.view([bs, 1, -1])  # (B, 1, H*W)

    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), \
                           torch.cat([x_ref, y_ref, torch.ones_like(x_ref)], dim=1) * \
                           depth_ref.view([bs, 1, -1]))  # (B, 3, H*W)
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)), \
                           torch.cat([xyz_ref, torch.ones_like(x_ref)], dim=1))[:, :3]

    # source view x, y
    k_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:, :2] / (k_xyz_src[:, 2:3].clamp(min=1e-10))  # (B, 2, H*W)

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].view([bs, height, width]).float()
    y_src = xy_src[:, 1].view([bs, height, width]).float()

    x_src_norm = x_src / ((width - 1) / 2) - 1
    y_src_norm = y_src / ((height - 1) / 2) - 1
    xy_src_norm = torch.stack([x_src_norm, y_src_norm], dim=3)
    sampled_depth_src = torch.nn.functional.grid_sample(depth_src.unsqueeze(1), xy_src_norm, \
                                                        mode="nearest", padding_mode="zeros")
    sampled_depth_src = sampled_depth_src.squeeze(1)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src), \
                           torch.cat([xy_src, torch.ones_like(x_ref)], dim=1) * \
                           sampled_depth_src.view([bs, 1, -1]))

    # reference 3D space:#(B, 3, H, W)
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)), \
                                   torch.cat([xyz_src, torch.ones_like(x_ref)], dim=1))[:, :3]

    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].view([bs, height, width]).float()
    depth_reprojected = depth_reprojected * (sampled_depth_src > 0)
    k_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:, :2] / (k_xyz_reprojected[:, 2:3].clamp(min=1e-10))
    x_reprojected = xy_reprojected[:, 0].view([bs, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].view([bs, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected


def check_geometric_consistency_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords,\
                                      thres_p_dist=1, thres_d_diff=0.01):
    """check geometric consistency
    consider two factor:
    1.disparity < 1
    2.relative depth differ ratio < 0.001
    """
    x_ref, y_ref = xy_coords  # (B, H, W)
    depth_reprojected, x2d_reprojected, y2d_reprojected = \
        reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords)

    # check |p_reproj-p_1| < p_dist
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < d_diff
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / (depth_ref.clamp(min=1e-10))

    mask = (dist < thres_p_dist) & (relative_depth_diff < thres_d_diff)
    # mask = (dist < thres_p_dist) & (depth_diff < 0.1)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected

def gemo_filter_fusion(depth_ref, depth_srcs, ref_pose, src_poses, intr, thres_view):
    depth_ref = torch.from_numpy(depth_ref).unsqueeze(0).to("cuda") #(1, H, W)
    ref_pose = torch.from_numpy(ref_pose).unsqueeze(0).to("cuda") #(1, 4, 4)
    intr = torch.from_numpy(intr).unsqueeze(0).to("cuda").float() #(1, 3, 3)

    depth_srcs = [torch.from_numpy(depth_src).unsqueeze(0).to("cuda") for depth_src in depth_srcs]
    src_poses = [torch.from_numpy(src_pose).unsqueeze(0).to("cuda") for src_pose in src_poses]
    
    xy_coords = get_coordinate_xy(depth_ref.shape, device=depth_ref.device)
    
    params = {"thres_p_dist": 1, "thres_d_diff": 0.001}
    
    geo_mask_sum = torch.zeros_like(depth_ref)
    all_srcview_depth_ests = torch.zeros_like(depth_ref)


    for depth_src, src_pose in zip(depth_srcs, src_poses):
        geo_mask, depth_reprojected = check_geometric_consistency_batch( \
            depth_ref, depth_src, \
            ref_pose={"intr": intr, "extr": ref_pose}, \
            src_pose={"intr": intr, "extr": src_pose}, \
            xy_coords=xy_coords, thres_p_dist=params["thres_p_dist"],\
            thres_d_diff=params["thres_d_diff"])
        geo_mask_sum += geo_mask.float()
        all_srcview_depth_ests += depth_reprojected
    
    
     # fusion
    geo_mask = (geo_mask_sum - thres_view) >= 0
    depth_ests_averaged = (all_srcview_depth_ests + depth_ref) / (geo_mask_sum + 1)
    depth_ests_averaged = depth_ests_averaged * geo_mask
    
    return depth_ests_averaged[0].detach().cpu().numpy()


def parse_video(video_file, save_root, sample_rate=10):
    os.makedirs(save_root, exist_ok=True)
    
    cap = cv2.VideoCapture(video_file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    sample_count = 0
    
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if count % sample_rate == 0:
                save_path = os.path.join(save_root, f"{sample_count}".zfill(6) + ".jpg")
                cv2.imwrite(save_path, img)
                sample_count += 1
            count += 1
        else:
            break
    print(f"video total frames num: {count},  sampled frames num:{sample_count}")    


def init_model(args):
    print("init model start...................")
    hvd_disable()
    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    print(f"input image shape:{image_shape}")
    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)
    
    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda')
    else:
        raise RuntimeError("cuda is not available")
        
    # Set to eval mode
    model_wrapper.eval()
    
    print("init model finish...................")
    return model_wrapper, image_shape
    


def inference(model_wrapper, image_shape, input, sample_rate,
              output_depths_npy, output_vis_video, output_tmp_dir,
              data_type="general", ply_mode=False, sfm_params=None):
    
    assert os.path.exists(input)
    assert os.path.exists(output_tmp_dir)
    save_depth_root = os.path.join(output_tmp_dir, "depth")
    save_vis_root = os.path.join(output_tmp_dir, "depth_vis")
    os.makedirs(save_depth_root, exist_ok=True)
    os.makedirs(save_vis_root, exist_ok=True)
    
    input_type = "folder"
    # processs input data
    if not os.path.isdir(input):
        print("processing video input:.........")
        input_type = "video"
        assert  os.path.splitext(input)[1] in [".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv"]
        input_video_images = os.path.join(output_tmp_dir, "input_video_images")
        parse_video(input, input_video_images, sample_rate)
        # update input
        input = input_video_images

    files = []
    for ext in ['png', 'jpg', 'bmp']:
        files.extend(glob((os.path.join(input, '*.{}'.format(ext)))))
    
    if input_type == "folder":
        print("processing folder input:...........")
        print(f"folder total frames num: {len(files)}")    
        files = files[::sample_rate]
    
    files.sort()
    print('Found total {} files'.format(len(files)))
    assert len(files) > 2
    
    # Process each file
    list_of_files = list(zip(files[:-2],
                              files[1:-1],
                              files[2:]))
    

    if ply_mode:
        # visulation
        # new points and poses get added to the queue
        queue_g = Queue()
        vis_counter = 0
        render_path=os.path.join(output_tmp_dir, "renders")
        os.makedirs(render_path, exist_ok=True)
        start_visualization(queue_g, cinematic=True, render_path=render_path,
                            clear_points=False, is_kitti= data_type=="kitti")

    pose_prev = None
    pose_23_prev = None
    depth_list = []
    pose_list = []
    
    print(f"*********************data_type:{data_type}")
    print("inference start.....................")
    for fn1, fn2, fn3 in list_of_files:
        depth, pose21, pose23, intr, rgb = infer_and_save_pose([fn1, fn3], fn2, model_wrapper, 
                                                                image_shape, data_type,
                                                                save_depth_root, save_vis_root)
        depth_list.append(depth)
        if ply_mode:
            if pose_23_prev is not None:
                s = np.linalg.norm(np.linalg.norm(pose_23_prev[:3, 3])) / np.linalg.norm(pose21[:3, 3])
                pose21[:3, 3] = pose21[:3, 3] * s
            pose_23_prev = pose23
            
            pose = pose21
            
            depth_pad = np.pad(depth, [(0, 1), (0, 1)], "constant")

            depth_grad = (depth_pad[1:, :-1] - depth_pad[:-1, :-1])**2 + (depth_pad[:-1, 1:] - depth_pad[:-1, :-1])**2
            depth[depth_grad > sfm_params["filer_depth_grad_max"]] = 0
            
            depth[depth > sfm_params["filer_depth_max"]] = 0
                        
            crop_h = sfm_params["depth_crop_h"]
            crop_w = sfm_params["depth_crop_w"]
              
            depth[:crop_h, :crop_w] = 0
            depth[-crop_h:, -crop_w:] = 0
            
            print(f"depth median:{np.median(depth)}")
            
            if pose_prev is not None:
                pose = np.matmul(pose_prev, pose)
            pose_prev = pose
            
            pose_list.append(pose)
            num_view = sfm_params["fusion_view_num"]
            if len(pose_list) >= num_view:
                depth = gemo_filter_fusion(depth_list[-1], depth_list[-num_view:-1], pose_list[-1],
                                        pose_list[-num_view:-1], intr, thres_view=sfm_params["fusion_thres_view"])
            
            pcd_colors = rgb.reshape(-1, 3) #TODO rgb or bgr
        
            h, w = depth.shape[:2]
            x_m, y_m = np.meshgrid(np.arange(0, w), np.arange(0, h))
            xy_m = np.stack([x_m, y_m, np.ones_like(x_m)], axis=2).reshape(-1, 3) #(h*w, 3)
            depth = depth.reshape(-1)[np.newaxis, :] #(1, N)
            p3d = np.multiply(depth, np.matmul(np.linalg.inv(intr), xy_m.transpose(1, 0))) #(3, N)
    
            p3d_trans = np.matmul(pose[:3], np.concatenate([p3d, np.ones((1, p3d.shape[1]))], axis=0)) #(3, N)
            
            pcd_coords = p3d_trans.transpose(1, 0)
            pointcloud = (pcd_coords, pcd_colors)

            vis_counter += 1
            pose = np.linalg.inv(pose)
            queue_g.put((pointcloud, pose))

    # save all depths and vis depths
    depth_npy_list = []
    for file in sorted(glob(os.path.join(save_depth_root, "*.npy"))):
        depth_npy_list.append(np.load(file))
    np.save(output_depths_npy, np.stack(depth_npy_list, axis=0))
    
    
    files = sorted(glob(os.path.join(save_vis_root, "*.jpg")))
    image_hw = cv2.imread(files[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_vis_video, fourcc, 1.0, (image_hw[1], image_hw[0]))
    for file in files:
        video_writer.write(cv2.imread(file))
    video_writer.release()
    print("inference finish.....................")
    
def main():
    args = parse_args()
    
    sfm_params = {
        "filer_depth_grad_max": 0.05,
        "filer_depth_max": 6 if not args.data_type=='kitti' else 15,
        "depth_crop_h": 32,
        "depth_crop_w": 32,
        "depth_crop_w": 32,
        "fusion_view_num": 5,
        "fusion_thres_view": 1,
    }
    
    model_wrapper, image_shape = init_model(args)
   
    
    input = args.input
    output_depths_npy = os.path.join(args.output, "depths.npy")
    output_vis_video = os.path.join(args.output, "depths_vis.avi")
    output_tmp_dir = os.path.join(args.output, "tmp")
    sample_rate = args.sample_rate
    data_type = args.data_type 
    ply_mode = args.ply_mode
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(output_tmp_dir, exist_ok=True)
    
    inference(model_wrapper, image_shape, input, sample_rate=sample_rate, 
              output_depths_npy=output_depths_npy, output_vis_video=output_vis_video, 
              output_tmp_dir=output_tmp_dir, data_type=data_type,
              ply_mode=ply_mode, sfm_params=sfm_params)
    
    # clean tmp dir
    # import shutil
    # shutil.rmtree(output_tmp_dir)

if __name__ == '__main__':
    main()
