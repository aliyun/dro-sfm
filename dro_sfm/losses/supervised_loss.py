
import torch
import torch.nn as nn

from dro_sfm.utils.image import match_scales
from dro_sfm.losses.loss_base import LossBase, ProgressiveScaling
from dro_sfm.geometry.camera import Camera, Pose
from dro_sfm.utils.depth import depth2inv, inv2depth
########################################################################################################################

class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.

        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.

        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss

########################################################################################################################

def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))

########################################################################################################################

class SupervisedLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_method='sparse-l1',
                 supervised_num_scales=4, progressive_scaling=0.0, **kwargs):
        super().__init__()
        self.loss_func = get_loss_func(supervised_method)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

########################################################################################################################

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        # If using a sparse loss, mask invalid pixels for all scales
        if self.supervised_method.startswith('sparse'):
            for i in range(self.n):
                mask = (gt_inv_depths[i] > 0.).detach()
                inv_depths[i] = inv_depths[i][mask]
                gt_inv_depths[i] = gt_inv_depths[i][mask]
        # Return per-scale average loss
        return sum([self.loss_func(inv_depths[i], gt_inv_depths[i])
                    for i in range(self.n)]) / self.n

    def forward(self, inv_depths, gt_inv_depth,
                return_logs=False, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Match predicted scales for ground-truth
        gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n,
                                     mode='nearest', align_corners=None)
        # Calculate and store supervised loss
        loss = self.calculate_loss(inv_depths, gt_inv_depths)
        self.add_metric('supervised_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }
        
        

########################################################################################################################

class SupervisedDepthPoseLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_method='sparse-l1',
                 supervised_num_scales=4, progressive_scaling=0.0, min_depth=0.1,
                 max_depth=100, **kwargs):
        super().__init__()
        self.loss_func = get_loss_func(supervised_method)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)
        self.min_depth = min_depth
        self.max_depth = max_depth
    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

########################################################################################################################

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        total_loss = 0
        total_w = 0
        gamma = 0.85
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        for i in range(self.n):
            w = gamma**(self.n - i - 1)
            total_w += w
            
            valid = ((gt_inv_depths[i] > min_disp) & (gt_inv_depths[i] < max_disp)).detach()
            valid = valid.squeeze(1)

            loss_depth = torch.mean(valid * torch.abs(gt_inv_depths[i] - inv_depths[i]).squeeze(1))
            loss_i = loss_depth
            total_loss += w * loss_i
        
        return total_loss / total_w

    
    def get_ref_coords(self, pose, K, ref_K, depth, scale_factor, device):
        if not isinstance(pose, Pose):
            pose = Pose(pose)

        cam = Camera(K=K.float()).scaled(scale_factor).to(device) # tcw = Identity
        ref_cam = Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device)
    
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
        valid_mask = (ref_coords >= -1) & (ref_coords <= 1)
        return ref_coords, valid_mask
    
        
    def calc_pose_loss(self, pred_poses, gt_pose_context, gt_depth, K, ref_K):
        device = gt_pose_context[0].device
        scale_factor = 1
        MAX_ERROR = 1
        total_loss = 0
        w_totoal = 0
        gamma = 0.85
        
        min_depth = self.min_depth
        max_depth = self.max_depth / 4.0
        
        gt_depth_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        gt_depth_mask = gt_depth_mask.permute(0, 2, 3 ,1)
        for i in range(self.n):
            loss_it = 0
            w = gamma**(self.n - i - 1)
            w_totoal += w
                
            for view_i, gt_pose_view in enumerate(gt_pose_context):
                pred_pose_view = pred_poses[view_i][i]
                
                coords_gt, mask_gt = self.get_ref_coords(gt_pose_view, K, ref_K, gt_depth, scale_factor, device)
                coords_pred, mask_pred = self.get_ref_coords(pred_pose_view, K, ref_K, gt_depth, scale_factor, device)
                valid_mask = mask_gt * mask_pred * gt_depth_mask
                reproj_diff = valid_mask * torch.abs(coords_pred - coords_gt).clamp(-MAX_ERROR, MAX_ERROR)
                reporj_loss = torch.mean(reproj_diff)
                loss_it += reporj_loss
                
            loss_it = loss_it / len(gt_pose_context) 
            total_loss += loss_it * w
        
        return total_loss / w_totoal

    
    def forward(self, image, context, inv_depths, gt_inv_depth, gt_pose_context, 
                K, ref_K, poses, return_logs=False, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = len(inv_depths) #self.progressive_scaling(progress)
        # Match predicted scales for ground-truth
        
        gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n,
                                     mode='nearest', align_corners=None)
        
        # Calculate and store supervised loss
        loss_depth = self.calculate_loss(inv_depths, gt_inv_depths)
        
        loss_pose = self.calc_pose_loss(poses, gt_pose_context, inv2depth(gt_inv_depth), K, ref_K)
        
        self.add_metric('depth_loss', loss_depth)
        self.add_metric('pose_loss', loss_pose)
        self.add_metric('all_loss', loss_depth + loss_pose)

        loss = loss_depth + loss_pose
                
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }