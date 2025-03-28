
import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from sensor_msgs.msg import RegionOfInterest
#from sensor_msgs.msg import PointCloud2, PointField
#import sensor_msgs.point_cloud2 as pc2
from genesis_inverse_kinematics.evaluate_path import compute_cost
from genesis_inverse_kinematics.task_setup import setup_task
from genesis_inverse_kinematics.static_transform_publisher import publish_transforms

class IK_Controller:
    def __init__(self):
        self.data_received = False
        self.executed_path = []
        self.TCP_path = []
        # ROS node initializations
        rospy.init_node('ik_genesis_node', anonymous=True)
        self.start_pos_pub = rospy.Publisher("start_position", Point, queue_size=1)
        self.goal_pos_pub = rospy.Publisher("goal_position", Point, queue_size=1)
        self.current_pos_pub = rospy.Publisher("current_position", Point, queue_size=1) 
        self.target_pos_sub = rospy.Subscriber("agent_position", Point, self.target_pos_callback)
        self.depth_image_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        self.rate = rospy.Rate(10)  

        # Genesis initialization
        gs.init(backend=gs.gpu)

        # Setup the task
        self.scene, self.franka, self.cam, self.target_pos = setup_task()
        self.goal_pos= self.target_pos

        # Build the scene
        self.scene.build()
        print("Camera pose: ", self.cam.transform)
        cam_pose = np.array([[ 0, 0, 1, 3.0],
                             [ 1, 0, 0, 0],
                             [ 0, 1, 0, 1],
                             [ 0, 0, 0, 1]])
        #
        cam_pose_rviz = np.array([[ 0, 0, -1, 3.0],
                             [ 1, 0, 0, 0],
                             [ 0, -1, 0, 1],
                             [ 0, 0, 0, 1]])
        publish_transforms(cam_pose_rviz)
        self.cam.set_pose(cam_pose) #x right, y up, z out of the screen
        self.scene.draw_debug_sphere(
            pos=self.goal_pos,
            radius=0.02,
            color=(1, 1, 0),
        )

        # Convert and publish the start position to the PMAF Planner
        self.end_effector = self.franka.get_link("hand")
        start_pos = self.end_effector.get_pos()
        self.start_pos_msg = Point()
        self.start_pos_msg.x = start_pos[0]
        self.start_pos_msg.y = start_pos[1]
        self.start_pos_msg.z = start_pos[2]
        self.start_pos_pub.publish(self.start_pos_msg)

        # Convert and publish the goal position to the PMAF Planner
        self.goal_pos_msg = Point()
        self.goal_pos_msg.x = self.goal_pos[0]
        self.goal_pos_msg.y = self.goal_pos[1]
        self.goal_pos_msg.z = self.goal_pos[2]
        self.goal_pos_pub.publish(self.goal_pos_msg)

        #Render the depth image of the scene
        _, self.depth_img, _, _ = self.cam.render(depth=True, segmentation=True, normal=True)
        print("Type of depth image: ", type(self.depth_img))
        print("Shape of depth image: ", self.depth_img.shape)    




        # Set control gains
        self.configure_controller()


    def target_pos_callback(self, data):
        self.data_received = True
        self.target_pos[0] = data.x
        self.target_pos[1] = data.y
        self.target_pos[2] = data.z

    def configure_controller(self):
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )

    def create_depth_image_msg(self, timestamp):
        bridge = CvBridge()
        # Convert the depth image to a sensor_msgs/Image
        depth_image_msg = bridge.cv2_to_imgmsg(self.depth_img, encoding="32FC1")
        # Set the timestamp and frame ID
        depth_image_msg.header.stamp = timestamp
        depth_image_msg.header.frame_id = "camera_depth_optical_frame"
        return depth_image_msg

    def create_camera_info_msg(self, timestamp):
        cam_info = CameraInfo()
        cam_info.header.stamp = timestamp
        cam_info.header.frame_id = "camera_depth_optical_frame"
        width = 640
        height = 480
        cam_info.width = width
        cam_info.height = height
        cam_info.distortion_model = "plumb_bob"
        cam_info.D = [0, 0, 0, 0, 0]
        # Compute focal length from the horizontal FOV
        fov_deg = 30.0
        fov_rad = fov_deg * np.pi / 180.0
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx  # assuming square pixels
        cx = width / 2.0
        cy = height / 2.0
        cam_info.K = [fx, 0.0, cx,
                      0.0, fy, cy,
                      0.0, 0.0, 1.0]
        cam_info.R = [1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0]
        cam_info.P = [fx, 0.0, cx, 0.0,
                      0.0, fy, cy, 0.0,
                      0.0, 0.0, 1.0, 0.0]
        cam_info.binning_x = 0
        cam_info.binning_y = 0
        # Region of Interest (full image)
        cam_info.roi.x_offset = 0
        cam_info.roi.y_offset = 0
        cam_info.roi.width = width
        cam_info.roi.height = height
        cam_info.roi.do_rectify = False
        return cam_info


    def run(self):
        #self.end_effector = self.franka.get_link("hand")
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()
        while not rospy.is_shutdown():
            timestamp = rospy.Time.now()
            # Publish the depth image
            depth_image_msg = self.create_depth_image_msg(timestamp)
            camera_info_msg = self.create_camera_info_msg(timestamp)
            self.depth_image_pub.publish(depth_image_msg)
            self.camera_info_pub.publish(camera_info_msg)
            # Publish the start and goal positions
            self.start_pos_pub.publish(self.start_pos_msg)
            self.goal_pos_pub.publish(self.goal_pos_msg)
            #self.pointcloud_pub.publish(self.pointcloud)
            if self.data_received:
                self.scene.draw_debug_sphere(
                    pos=self.target_pos,
                    radius=0.02,
                    color=(1, 0, 0),
                )
                qpos = self.franka.inverse_kinematics(
                    link=self.end_effector,
                    pos=self.target_pos,
                    quat=np.array([0, 1, 0, 0]),
                )
                # Gripper open pos
                qpos[-2:] = 0.04
                self.franka.control_dofs_position(qpos[:-2], np.arange(7))

                # Trajectory visualization
                ee_pos = self.end_effector.get_pos()
                if isinstance(ee_pos, torch.Tensor):
                    ee_pos = ee_pos.cpu().numpy()

                current_pos_msg = Point()
                current_pos_msg.x = ee_pos[0]
                current_pos_msg.y = ee_pos[1]
                current_pos_msg.z = ee_pos[2]
                self.current_pos_pub.publish(current_pos_msg)


                self.TCP_path.append(ee_pos)
                self.scene.draw_debug_line(
                    start=self.prev_eepos,
                    end=ee_pos,
                    color=(0, 1, 0),
                )
                self.prev_eepos = ee_pos

                links_pos = self.franka.get_links_pos()     #Store position of all robot´s links to executed_path
                if isinstance(links_pos, torch.Tensor):
                    links_pos = links_pos.cpu().numpy()
                self.executed_path.append(links_pos)

                self.scene.step()
            #if np.allclose(self.target_pos, self.goal_pos, atol=1e-3):
            #    cost = compute_cost(self.executed_path, self.TCP_path, self.obstacle_centers, self.obs_radius)
            #    print("Path cost: ", cost)
            self.rate.sleep()

if __name__ == "__main__":
    ik_controller = IK_Controller()
    ik_controller.run()