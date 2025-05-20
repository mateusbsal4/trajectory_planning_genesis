#!/usr/bin/env python3

import rospy

import argparse
import utils.troubleshoot as troubleshoot

import numpy as np
import cupoch as cph  # Assuming cupoch is imported as cph

import tf2_ros  
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState

from perception_pipeline import PerceptionPipeline
from perception_node import PerceptionNode

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from utils.camera_helpers import create_tf_matrix_from_msg


class SimPerceptionPipeline(PerceptionPipeline):
    def __init__(self):
        super().__init__()

        # load configs
        self.load_and_setup_pipeline_configs()

        # Define subscriber for robot´s AABB
        self.aabb_sub = rospy.Subscriber('robot_aabb', Float32MultiArray, self.aabb_callback)

        # Define publisher for voxel spheres
        self.voxel_grid_pub = rospy.Publisher('scene_voxels', Float64MultiArray, queue_size=1)

        # finish setup
        super().setup()

    def load_and_setup_pipeline_configs(self):
        self.perception_pipeline_config = rospy.get_param("perception_pipeline_config/", None)  
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']

    def aabb_callback(self, msg):
        # Check if the data contains exactly 6 elements
        if len(msg.data) != 6:
            rospy.logerr("Received AABB data length is not 6: {}".format(len(msg.data)))
            return

        # Extract the min and max bounds
        min_bound = np.array(msg.data[0:3], dtype=np.float32)
        max_bound = np.array(msg.data[3:6], dtype=np.float32)

        # Create the Cupoch AABB object
        self.robot_aabb = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        rospy.loginfo("Received robot AABB: min {} | max {}".format(min_bound, max_bound))
    
class SimPerceptionNode(PerceptionNode):
    def __init__(self):
        rospy.init_node('sim_perception_node')
        self.save_cloud = rospy.get_param("~save_cloud")
        self.scene = rospy.get_param("~scene")
        super().__init__()
        
        # Initialize pipeline
        self.pipeline = SimPerceptionPipeline()

        self.setup_ros_subscribers()

    def setup_ros_subscribers(self):
        rospy.loginfo("Setting up subscribers")
        self.ptcloud_subscriber = rospy.Subscriber(
            'camera/depth/points', PointCloud2, self.static_camera_callback)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def static_camera_callback(self, msg):
        with self.buffer_lock:
            # Directly store the point cloud message since there's only one camera
            self.pointcloud_buffer = msg

            # Read the camera matrix from the tf buffer
            try:
                transform = self.tfBuffer.lookup_transform("world", "camera_depth_optical_frame", rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Failed to get transform from world to camera_depth_optical_frame")
            
            # Create the 4x4 transformation matrix 
            tf_matrix = create_tf_matrix_from_msg(transform)
            
            # Submit the pipeline task with the single point cloud
            future = self.executor.submit(self.run_pipeline, self.pointcloud_buffer, tf_matrix)


def main():
    node = SimPerceptionNode()

    return node

if __name__ == "__main__":
    try:
        node = main()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()

