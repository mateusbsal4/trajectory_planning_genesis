#!/usr/bin/env python3

import rospy

import threading
from concurrent.futures import ThreadPoolExecutor

import utils.troubleshoot as troubleshoot

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import numpy as np
import open3d as o3d    
import cupy as cp


class PerceptionNode:
    def __init__(self, max_threads=5):
        
        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)

        # Buffer to store latest pointclouds
        self.pointcloud_buffer = {}
        self.buffer_lock = threading.Lock()

        # Publisher for results
        self.primitives_publisher = rospy.Publisher('primitives', PointCloud2, queue_size=10)

    def run_pipeline(self, pointcloud_buffer, tfs):
        try:
            if self.save_cloud: # only generats the point cloud of each scene
                pcd = self.pipeline.parse_pointcloud(pointcloud_buffer, tfs, True)
                pcd = self.pipeline.perform_robot_body_subtraction(pcd)
                
                # Convert pcd (cph.geometry.PointCloud) to numpy array
                points = np.asarray(pcd.points.cpu())
                # Create Open3D point cloud from NumPy array
                scene_path = f"/home/geriatronics/pmaf_ws/src/dataset_generator/data/inputs/{self.scene}.ply"
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(points)

                # Save and optionally visualize
                o3d.io.write_point_cloud(scene_path, o3d_pcd)
                #o3d.visualization.draw_geometries([o3d_pcd])

                rospy.loginfo(f"Point cloud saved to {scene_path}")
                rospy.signal_shutdown("Saved cloud; exiting")
                return
            primitives_pos_result = self.pipeline.run_pipeline(
                pointcloud_buffer, tfs, True)  
            #print("primitives_pos_result", primitives_pos_result)   
            primitives_pos_msg = self.make_pointcloud_msg(primitives_pos_result)
            # publish messages
            if primitives_pos_msg is not None:
                self.primitives_publisher.publish(primitives_pos_msg)

        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

    def make_pointcloud_msg(self, points_array):
        # Define header
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"

        points_list = points_array

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, points_list)
        return point_cloud_msg
    
    def shutdown(self):
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")