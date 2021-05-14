import torch
from dro_sfm.models.SelfSupModelMF import SelfSupModelMF, SfmModelMF
from dro_sfm.losses.supervised_loss import SupervisedDepthPoseLoss as SupervisedLoss
from dro_sfm.models.model_utils import merge_outputs
from dro_sfm.utils.depth import depth2inv


class SemiSupModelMFPose(SelfSupModelMF):
    """
    Model that inherits a depth and pose networks, plus the self-supervised loss from
    SelfSupModel and includes a supervised loss for semi-supervision.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)

        print(f"=================supervised_loss_weight:{supervised_loss_weight}====================")
        # # Pose network is only required if there is self-supervision
        # self._network_requirements['pose_net'] = self.supervised_loss_weight < 1
        # # GT depth is only required if there is supervision
        self._train_requirements['gt_depth'] = self.supervised_loss_weight > 0
        self._train_requirements['gt_pose'] = True

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, image, ref_images, inv_depths, gt_depth, gt_poses, poses,
                             intrinsics, return_logs=False, progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            image, ref_images, inv_depths, depth2inv(gt_depth), gt_poses, intrinsics, intrinsics, poses,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModelMF.forward(self, batch)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModelMF.forward(self, batch)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModelMF.forward(self, batch)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']
            # Calculate and weight supervised loss
            sup_output = self.supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                self_sup_output['inv_depths'],  batch['depth'], batch['pose_context'], self_sup_output['poses'], batch['intrinsics'],
                return_logs=return_logs, progress=progress)
            loss += self.supervised_loss_weight * sup_output['loss']
            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, sup_output),
            }