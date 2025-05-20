import cupoch as cph
import numpy as np
import cupy as cp

import rospy
import logging
import time
import utils.troubleshoot as troubleshoot

import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from std_msgs.msg import Float64MultiArray

class PerceptionPipeline():
    def __init__(self):
        self.logger = logging.getLogger("perception_pipeline_logger")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("/home/geriatronics/pmaf_ws/src/dataset_generator/logs/perception_pipeline.log", mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)    
        self.logger.addHandler(file_handler)

        self.check_cuda()
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def check_cuda(self):
        try:
            subprocess.check_output(["nvidia-smi"])
            rospy.loginfo("CUDA is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            rospy.logerr("CUDA is not available - running CPU fallback")
            return False

    def setup(self):
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.scene_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cubic_size = self.cubic_size
        voxel_resolution = self.voxel_resolution
        self.voxel_size = cubic_size/voxel_resolution
        self.voxel_min_bound = (-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0)
        self.voxel_max_bound = (cubic_size/2.0, cubic_size/2.0, cubic_size/2.0)

    def parse_pointcloud(self, pointcloud_msg, tf_matrix=None, downsample: bool = False, log_performance: bool = False):
        start = time.time()
        try:
            pcd = cph.geometry.PointCloud()
            temp = cph.io.create_from_pointcloud2_msg(
                pointcloud_msg.data,
                cph.io.PointCloud2MsgInfo.default_dense(
                    pointcloud_msg.width, pointcloud_msg.height, pointcloud_msg.point_step
                )
            )
            pcd.points = temp.points
            if tf_matrix is not None:
                pcd = pcd.transform(tf_matrix)
            pcd = pcd.crop(self.scene_bbox)
            if downsample:
                pcd = pcd.uniform_down_sample(3)
            return pcd
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))
            return None

    def perform_robot_body_subtraction(self, pointcloud, log_performance: bool = False):
        try:
            aabb = self.robot_aabb
            indices = aabb.get_point_indices_within_bounding_box(pointcloud.points)
            filtered = pointcloud.select_by_index(indices, invert=True)
            return filtered
        except Exception as e:
            rospy.logerr(f"Failed robot subtraction: {troubleshoot.get_error_text(e)}")
            return pointcloud

    def perform_voxelization(self, pcd:cph.geometry.PointCloud, log_performance:bool=False):
        return cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=self.voxel_size,
            min_bound=self.voxel_min_bound,
            max_bound=self.voxel_max_bound,
        )
    
    def convert_voxels_to_primitives(self, voxel_grid:cph.geometry.VoxelGrid, log_performance:bool=False):
        try:
            voxels = voxel_grid.voxels.cpu()
            primitives_pos = np.array(list(voxels.keys()), dtype=np.float32)
            if primitives_pos.size == 0:
                rospy.logwarn("No voxels found; returning empty array")
                return np.empty((0,3), dtype=np.float32)

            # ensure contiguous
            primitives_pos = np.ascontiguousarray(primitives_pos)

            # GPU path
            try:
                primitives_gpu = cp.asarray(primitives_pos)
                offset = cp.asarray(voxel_grid.get_min_bound(), dtype=cp.float32)
                vsize  = cp.asarray(self.voxel_size, dtype=cp.float32)
                mins   = cp.min(primitives_gpu, axis=0)
                primitives_gpu = (primitives_gpu - mins[None,:]) * vsize + (offset + vsize/2)
                primitives_out = cp.asnumpy(primitives_gpu)
            except Exception as e:
                rospy.logwarn(f"GPU pipeline failed ({e}); using CPU fallback")
                # CPU fallback
                mins = primitives_pos.min(axis=0)
                primitives_out = (primitives_pos - mins) * self.voxel_size + (np.array(voxel_grid.get_min_bound()) + self.voxel_size/2)

            return primitives_out
        except Exception as e:
            rospy.logerr(f"convert_voxels_to_primitives failed: {troubleshoot.get_error_text(e)}")
            return np.empty((0,3), dtype=np.float32)

    def publish_primitives(self, primitives_pos):
        sphere_msg = Float64MultiArray()
        flat_data = []
        for pos in primitives_pos:
            flat_data.extend([pos[0], pos[1], pos[2], self.voxel_size])
        sphere_msg.data = flat_data
        self.voxel_grid_pub.publish(sphere_msg)

    def run_pipeline(self, pointcloud_msg, tf_matrix, log_performance: bool = False):
        start = time.time()
        pcd         = self.parse_pointcloud(pointcloud_msg, tf_matrix, downsample=True)
        pcd         = self.perform_robot_body_subtraction(pcd)
        voxel_grid  = self.perform_voxelization(pcd)
        primitives  = self.convert_voxels_to_primitives(voxel_grid)
        self.publish_primitives(primitives)
        if log_performance:
            self.logger.info(f"Perception pipeline total: {time.time()-start:.3f}s")
        return primitives
