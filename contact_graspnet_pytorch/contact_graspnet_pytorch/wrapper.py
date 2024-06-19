import contact_graspnet_pytorch.config_utils as config_utils
import numpy as np
import numpy.typing as npt
import open3d as o3d
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.visualization_utils_o3d import (
    show_image,
    visualize_grasps,
)
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from contact_graspnet_pytorch.checkpoints import CheckpointIO


def filter_pcd(pcd_input):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_input)

    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # o3d.visualization.draw_geometries([uni_down_pcd])
    # print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=128, std_ratio=3.0)
    pcd = pcd.select_by_index(ind)
    # print("Radius oulier removal")
    cl, ind = pcd.remove_radius_outlier(nb_points=128, radius=0.03)
    pcd = pcd.select_by_index(ind)
    return np.asarray(pcd.points)


def normalize_pose(pose: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    r = R.from_matrix(pose[:3, :3])
    q = r.as_quat()  # astuce, en convertissant en quaternion il normalise
    r2 = R.from_quat(q)
    good_rot = r2.as_matrix()
    pose[:3, :3] = good_rot

    return pose


class ContactGraspNetWrapper:
    def __init__(
        self,
    ):
        global_config = config_utils.load_config(batch_size=1)
        self.grasp_estimator = GraspEstimator(global_config)
        checkpoint_io = CheckpointIO(
            checkpoint_dir="/tmp/", model=self.grasp_estimator.model
        )
        model_path = hf_hub_download(
            repo_id="pollen-robotics/contact_graspnet",
            filename="checkpoints/contact_graspnet/checkpoints/model.pt",
        )
        try:
            load_dict = checkpoint_io.load(model_path)
        except FileExistsError:
            print("No model checkpoint found")
            load_dict = {}
            exit()

    def infer(
        self, segmap, rgb, depth, cam_K, pc_full=None, pc_colors=None, filtering=False
    ):
        """
        Returns grasps sorted by scores in descending order
        Returns:
            - dict {1: [list of grasp poses], 2: [list of grasp poses]...}
            - scores
            - contact_pts
        """
        print("Contact graspnet infer")
        if pc_full is None:
            print("Converting depth to point cloud(s)...")
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
                depth,
                cam_K,
                segmap=segmap,
                rgb=rgb,
                skip_border_objects=True,
                z_range=[0.2, 1.8],
            )

        if filtering:
            pc_full = filter_pcd(pc_full)
            pc_seg_fil = {}
            for segment in pc_segments.items():
                pc_seg_fil[segment[0]] = filter_pcd(segment[1])
            pc_segments = pc_seg_fil

        pred_grasps_cam, scores, contact_pts, gripper_openings = (
            self.grasp_estimator.predict_scene_grasps(
                pc_full,
                pc_segments=pc_segments,
                local_regions=True,
                filter_grasps=True,
                forward_passes=1,
            )
        )

        if 1 not in scores.keys() or len(scores[1]) == 0:
            return (
                pred_grasps_cam,
                scores,
                contact_pts,
                gripper_openings,
                pc_full,
                pc_colors,
            )

        sorted_grasps = {}
        sorted_scores = {}
        sorted_contact_pts = {}
        print(f"scores: {scores}")
        for k in scores.keys():
            print(
                f"SCORES: {k} {scores[k].shape} {pred_grasps_cam[k].shape} {contact_pts[k].shape}"
            )
            (sorted_scores[k], sorted_grasps[k], sorted_contact_pts[k]) = zip(
                *sorted(
                    zip(
                        scores[k],
                        # normalize_pose(pred_grasps_cam[k]), # TODO bug ?
                        pred_grasps_cam[k],
                        contact_pts[k],
                    ),
                    reverse=True,
                    key=lambda x: x[0],
                )
            )

        return (
            sorted_grasps,
            sorted_scores,
            sorted_contact_pts,
            gripper_openings,
            pc_full,
            pc_colors,
        )

    def visualize(
        self,
        rgb,
        segmap,
        pc_full,
        pred_grasps_cam,
        scores,
        pc_colors,
        gripper_openings=None,
    ):
        show_image(rgb, segmap)
        visualize_grasps(
            pc_full,
            pred_grasps_cam,
            scores,
            plot_opencv_cam=True,
            gripper_openings=gripper_openings,
            pc_colors=pc_colors,
        )


if __name__ == "__main__":
    c = ContactGraspNetWrapper()

    data = np.load(
        "/home/antoine/Téléchargements/0.npy",
        allow_pickle=True,
    ).item()
    grasp_poses, scores, contact_pts, pc_full, pc_colors = c.infer(
        data["seg"], data["rgb"], data["depth"], data["K"]
    )
    c.visualize(data["rgb"], data["seg"], pc_full, grasp_poses, scores, pc_colors)
