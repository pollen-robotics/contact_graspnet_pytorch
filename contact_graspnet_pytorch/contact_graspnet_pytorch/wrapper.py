import os
from importlib.resources import files
from typing import Any

import contact_graspnet_pytorch.config_utils as config_utils
import numpy as np

# from contact_graspnet_pytorch.checkpoint_io import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.visualization_utils_o3d import (
    show_image,
    visualize_grasps,
)


class ContactGraspNetWrapper:
    def __init__(
        self,
    ):
        ckpt_dir = config_utils.get_checkpoints_dir_path()
        global_config = config_utils.load_config(ckpt_dir, batch_size=1)
        self.grasp_estimator = GraspEstimator(global_config)
        model_checkpoint_dir = os.path.join(ckpt_dir, "checkpoints")

        # checkpoint_io = CheckpointIO(
        #     checkpoint_dir=model_checkpoint_dir, model=self.grasp_estimator.model
        # )
        # try:
        #     load_dict = checkpoint_io.load("model.pt")
        # except FileExistsError:
        #     print("No model checkpoint found")
        #     load_dict = {}

    def infer(
        self, segmap, rgb, depth, cam_K, pc_full=None, pc_colors=None, visualize=False
    ):
        """
        Returns:
            - dict {1: [list of grasp poses], 2: [list of grasp poses]...}
            - scores
            - contact_pts
        """
        if pc_full is None:
            print("Converting depth to point cloud(s)...")
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
                depth,
                cam_K,
                segmap=segmap,
                rgb=rgb,
                skip_border_objects=False,
                z_range=[0.2, 1.8],
            )

        pred_grasps_cam, scores, contact_pts, _ = (
            self.grasp_estimator.predict_scene_grasps(
                pc_full,
                pc_segments=pc_segments,
                local_regions=True,
                filter_grasps=True,
                forward_passes=1,
            )
        )

        if visualize:
            show_image(rgb, segmap)
            visualize_grasps(
                pc_full,
                pred_grasps_cam,
                scores,
                plot_opencv_cam=True,
                pc_colors=pc_colors,
            )
        return pred_grasps_cam, scores, contact_pts


if __name__ == "__main__":
    c = ContactGraspNetWrapper()

    data = np.load("/home/antoine/Téléchargements/0.npy", allow_pickle=True).item()
    grasp_poses, _, _ = c.infer(data["seg"], data["rgb"], data["depth"], data["K"])
