#!/usr/bin/env python3
import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from genesis_inverse_kinematics.task_setup import setup_task    
from genesis_inverse_kinematics.static_transform_publisher import publish_transforms
from genesis_inverse_kinematics.perception_utils import create_depth_image_msg, create_camera_info_msg

class IK_Controller:
    def __init__(self):
        # Initialize ROS node (anonymous so remapped nodes are unique)
        rospy.init_node('ik_genesis_node', anonymous=True)

        # Publishers for start and goal positions
        self.start_pos_pub = rospy.Publisher("start_position", Point, queue_size=1)
        self.goal_pos_pub = rospy.Publisher("goal_position", Point, queue_size=1)
        
        # Create one publisher per environment for current position.
        self.n_envs = rospy.get_param("~n_envs", 16)
        self.current_pos_pubs = []
        for i in range(self.n_envs):
            topic_name = f"current_position_env{i}"
            pub = rospy.Publisher(topic_name, Point, queue_size=1)
            self.current_pos_pubs.append(pub)

        # Other publishers
        self.depth_image_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        self.aabb_pub = rospy.Publisher('/robot_aabb', Float32MultiArray, queue_size=1)

        self.rate = rospy.Rate(10)

        # Create a target position array of shape (n_envs, 3)
        self.target_pos = np.zeros((self.n_envs, 3))
        # Subscribe to each planner’s target position topic
        self.target_pos_subs = []
        for i in range(self.n_envs):
            topic_name = f"agent_position_env{i}"
            sub = rospy.Subscriber(topic_name, Point, lambda msg, idx=i: self.target_pos_callback(msg, idx))
            self.target_pos_subs.append(sub)

        # Genesis initialization and task setup
        gs.init(backend=gs.gpu)
        self.scene, self.franka, self.cam, _ = setup_task()
        self.goal_pos = np.array([0.4, -0.2, 0.25])

        # Build the scene in parallel 
        self.scene.build(n_envs=self.n_envs, env_spacing=(1.0, 1.0))

        # Set up camera transforms
        cam_pose = np.array([[0, 0, 1, 3.0],
                             [1, 0, 0, 0],
                             [0, 1, 0, 1],
                             [0, 0, 0, 1]])
        self.cam_pose_rviz = np.array([[0, 0, -1, 3.0],
                                        [1, 0, 0, 0],
                                        [0, -1, 0, 1],
                                        [0, 0, 0, 1]])
        publish_transforms(self.cam_pose_rviz)
        self.cam.set_pose(cam_pose)

        # self.scene.draw_debug_sphere(pos=self.goal_pos, radius=0.02, color=(1, 1, 0))

        # Publish start and goal positions (for the planner)
        self.end_effector = self.franka.get_link("hand")
        start_pos = self.end_effector.get_pos(0)        #same start and goal position for all envs
        if isinstance(start_pos, torch.Tensor):
            start_pos = start_pos.cpu().numpy()
        self.start_pos_msg = Point(x=start_pos[0], y=start_pos[1], z=start_pos[2])
        self.start_pos_pub.publish(self.start_pos_msg)
        self.goal_pos_msg = Point(x=self.goal_pos[0], y=self.goal_pos[1], z=self.goal_pos[2])
        self.goal_pos_pub.publish(self.goal_pos_msg)

        # Render the depth image of the scene
        _, self.depth_img, _, _ = self.cam.render(depth=True, segmentation=True, normal=True)
        self.publish_robot_aabb()
        self.configure_controller()

        # Storage for paths (for cost computation later)
        self.executed_path = []
        self.TCP_path = []

    def target_pos_callback(self, msg, idx):
        # Update the target position for the given environment index.
        self.target_pos[idx, :] = np.array([msg.x, msg.y, msg.z])
        rospy.loginfo(f"Received target pos for env {idx}: {self.target_pos[idx, :]}")

    def configure_controller(self):
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )

    def publish_robot_aabb(self):
        aabb_list = self.franka.get_AABB().cpu().numpy().tolist()  # [min_bound, max_bound]
        data = aabb_list[0] + aabb_list[1]
        msg = Float32MultiArray(data=data)
        self.aabb_pub.publish(msg)
        rospy.loginfo(f"Published robot AABB: {data}")

    def run(self):
        # Get end-effector positions for all environments
        self.prev_eepos = self.end_effector.get_pos(envs_idx=list(range(self.n_envs)))
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()
        while not rospy.is_shutdown():
            timestamp = rospy.Time.now()
            # Publish depth image and camera info
            depth_image_msg = create_depth_image_msg(self.depth_img, timestamp)
            camera_info_msg = create_camera_info_msg(timestamp, self.cam)
            self.depth_image_pub.publish(depth_image_msg)
            self.camera_info_pub.publish(camera_info_msg)
            publish_transforms(self.cam_pose_rviz)
            self.start_pos_pub.publish(self.start_pos_msg)
            self.goal_pos_pub.publish(self.goal_pos_msg)

            # Publish each environment's current position
            ee_pos = self.end_effector.get_pos(envs_idx=list(range(self.n_envs)))
            if isinstance(ee_pos, torch.Tensor):
                ee_pos = ee_pos.cpu().numpy()
            for i in range(self.n_envs):
                current_pos_msg = Point(x=ee_pos[i, 0], y=ee_pos[i, 1], z=ee_pos[i, 2])
                self.current_pos_pubs[i].publish(current_pos_msg)

            # For each environment, if a target has been set (nonzero), compute and apply IK.
            for i in range(self.n_envs):
                target = self.target_pos[i, :]
                if np.linalg.norm(target) > 1e-6:
                    qpos = self.franka.inverse_kinematics(
                        link=self.end_effector,
                        pos=target,
                        quat=np.array([0, 1, 0, 0])
                    )
                    qpos[-2:] = 0.04  # Set gripper open
                    self.franka.control_dofs_position(qpos[:-2], np.arange(7))

            # Update path storage (debug drawing functions are commented out)
            ee_pos_single = self.end_effector.get_pos()  # Single (default) env position for logging
            if isinstance(ee_pos_single, torch.Tensor):
                ee_pos_single = ee_pos_single.cpu().numpy()
            self.TCP_path.append(ee_pos_single)
            links_pos = self.franka.get_links_pos()
            if isinstance(links_pos, torch.Tensor):
                links_pos = links_pos.cpu().numpy()
            self.executed_path.append(links_pos)
            self.scene.step()
            self.rate.sleep()

if __name__ == "__main__":
    controller = IK_Controller()
    controller.run()
