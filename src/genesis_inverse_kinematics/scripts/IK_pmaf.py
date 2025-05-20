
import genesis as gs
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import logging 
import time
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray
from genesis_inverse_kinematics.evaluate_path import compute_cost
from genesis_inverse_kinematics.task_setup import create_scene, recreate_task     
from genesis_inverse_kinematics.static_transform_publisher import publish_transforms
from genesis_inverse_kinematics.perception_utils import create_depth_image_msg, create_camera_info_msg
import os


class IK_Controller:
    def __init__(self):
        self.data_received = False
        self.TCP_path = []
        self.min_dists = []
        # ROS node initializations  
        rospy.init_node('ik_genesis_node', anonymous=True)
        self.scene_config = rospy.get_param("~scene")  # Default to "base_scene.yaml"
        self.recreate = rospy.get_param("~recreate")  # Default to False
        self.evaluate = rospy.get_param("~evaluate", False)  # Default to False
        self.bo = rospy.get_param("~bo") 
        self.dataset_scene = rospy.get_param("~dataset_scene", False)  # Default to False
        self.start_pos_pub = rospy.Publisher("start_position", Point, queue_size=1)
        self.goal_pos_pub = rospy.Publisher("goal_position", Point, queue_size=1)
        self.tcp_pos_pub = rospy.Publisher("tcp_pos", Point, queue_size=1)
        self.current_pos_pub = rospy.Publisher("current_position", Point, queue_size=1) 
        self.target_pos_sub = rospy.Subscriber("agent_position", Point, self.target_pos_callback)
        self.depth_image_pub = rospy.Publisher('camera/depth/image_rect_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("camera/depth/camera_info", CameraInfo, queue_size=1)
        self.aabb_pub = rospy.Publisher('robot_aabb', Float32MultiArray, queue_size=1)
        self.voxel_grid_sub = rospy.Subscriber("scene_voxels", Float64MultiArray, self.voxel_grid_callback)
        #self.cost_pub = rospy.Publisher("cost", Float32, queue_size=1)
        self.cost_pub = rospy.Publisher("cost", Float32MultiArray, queue_size=1)
        self.rate = rospy.Rate(10)  
        # Setup logging
        self.logger = logging.getLogger("genesis_logger")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("/home/geriatronics/pmaf_ws/src/dataset_generator/logs/genesis.log", mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)    
        self.logger.addHandler(file_handler)
        
        # Genesis initialization
        gs.init(backend=gs.gpu)
        # Setup the task
        start = time.time()
        if self.recreate:    #recreate a specific scene
            print("Recreating scene from file: ", self.scene_config)
            self.scene, self.franka, self.cam, self.target_pos = recreate_task(self.scene_config + ".yaml", from_dataset = self.dataset_scene)
        else:           #setup a new random scene
            self.scene, self.franka, self.cam, self.target_pos = create_scene(randomize = True)
        scene_built = time.time()
        self.logger.info(f"Scene creation time [sec]: {scene_built - start}")
        goal_pos_TCP = self.target_pos.copy()               #TCP position - midpoint of the two gripper fingers
        self.goal_pos = np.array([goal_pos_TCP[0], goal_pos_TCP[1], goal_pos_TCP[2] + 0.10365])  # Gripper base position - 0.1m above the TCP position
        # Build the scene
        self.scene.build()
        #self.scene.build(compile_kernels=False)  
        self.logger.info(f"Scene build time [sec]: {time.time() - scene_built}")
        cam_pose = np.array([[ 0, 0, 1, 3.0],
                             [ 1, 0, 0, 0.5],
                             [ 0, 1, 0, 1.5],
                             [ 0, 0, 0, 1]])
        self.cam_pose_rviz = np.array([[ 0, 0, -1, 3.0],
                                        [ 1, 0, 0, 0.5],
                                        [ 0, -1, 0, 1.5],
                                        [ 0, 0, 0, 1]])
        publish_transforms(self.cam_pose_rviz)
        self.cam.set_pose(cam_pose) #x right, y up, z out of the screen
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
        
        # Render the depth image of the scene
        _, self.depth_img, _, _ = self.cam.render(depth=True, segmentation=True, normal=True)  
        # Draw goal position for the hand and TCP frames
        self.scene.draw_debug_sphere(           
            pos=self.goal_pos,
            radius=0.02,
            color=(1, 1, 0),
        )
        self.scene.draw_debug_sphere(           
            pos=goal_pos_TCP,
            radius=0.02,
            color=(0, 1, 1),
        )
        # Publish the robot´s AABB
        self.publish_robot_aabb()
        # Set control gains
        self.configure_controller()
        self.planning_initiated = False

    def target_pos_callback(self, data):
        if not self.planning_initiated:
            self.start_time = time.time()
        self.planning_initiated = True
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

    def publish_robot_aabb(self):
        # Get the AABB from the simulation, assuming it returns a Cupoch AABB
        aabb_list = self.franka.get_AABB().cpu().numpy().tolist()  # aabb_list = [min_bound, max_bound]
        # Concatenate min and max bounds into a single list of six floats
        data = aabb_list[0] + aabb_list[1]
        # Create the Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = data
        # Publish the message
        self.aabb_pub.publish(msg)
        #rospy.loginfo("Published robot AABB: {}".format(data))

    def voxel_grid_callback(self, data):
        # Convert the received Float64MultiArray data to a NumPy array
        flat_data = np.array(data.data)    
        # Reshape the flat data into a 2D array where each row is [x, y, z, radius]
        obstacles = flat_data.reshape(-1, 4)
        # Extract obstacle centers
        self.obs_centers = obstacles[:, :3]  # First three columns are x, y, z
        # Extract the radius (all obstacles have the same radius, so take the first one)
        self.obs_radius = obstacles[0, 3]
        

    def run(self):
        started_recording = False
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()
        while not rospy.is_shutdown():
            timestamp = rospy.Time.now()
            # Render the depth image of the scene
            #_, self.depth_img, _, _ = self.cam.render(depth=True, segmentation=True, normal=True)  
            # Publish the depth image
            depth_image_msg = create_depth_image_msg(self.depth_img, timestamp)
            camera_info_msg = create_camera_info_msg(timestamp, self.cam)
            self.depth_image_pub.publish(depth_image_msg)
            self.camera_info_pub.publish(camera_info_msg)
            # Publish the camera frame        
            publish_transforms(self.cam_pose_rviz)
            # Publish the start and goal positions
            self.start_pos_pub.publish(self.start_pos_msg)
            self.goal_pos_pub.publish(self.goal_pos_msg)
            # Publish the robot´s AABB
            #self.publish_robot_aabb()

            if self.data_received:
                if self.evaluate and not started_recording:
                    self.cam.start_recording()
                    started_recording = True
                qpos = self.franka.inverse_kinematics(
                    link=self.end_effector,
                    pos=self.target_pos,
                    quat=np.array([0, 1, 0, 0]),
                )
                # Gripper open pos
                qpos[-2:] = 0.04
                self.franka.control_dofs_position(qpos[:-2], np.arange(7))

                ee_pos = self.end_effector.get_pos()
                if isinstance(ee_pos, torch.Tensor):
                    ee_pos = ee_pos.cpu().numpy()

                # Draw the end-effector (hand frame) position
                #self.scene.draw_debug_sphere(          
                #    pos=ee_pos,
                #    radius=0.005,
                #    color=(0, 0, 1),
                #)
                # Draw the TCP position
                quat_hand = self.end_effector.get_quat()        #get the hand orientation    
                if isinstance(quat_hand, torch.Tensor):
                    quat_hand = quat_hand.cpu().numpy()       
                R_hand = R.from_quat(quat_hand)                
                R_hand = R_hand.as_matrix()                    #convert to rotation matrix
                TCP_pos = ee_pos + R_hand.dot(np.array([0, 0, 0.10365]))  # TCP position - 0.1m below the hand position
                self.scene.draw_debug_sphere(           #TCP position
                    pos=TCP_pos,
                    radius=0.005,
                    color=(0, 1, 0),
                )
                # Publish the current position  
                current_pos_msg = Point()
                current_pos_msg.x = ee_pos[0]
                current_pos_msg.y = ee_pos[1]
                current_pos_msg.z = ee_pos[2]
                self.current_pos_pub.publish(current_pos_msg)

                #Publish the TCP pos 
                tcp_pos_msg = Point()
                tcp_pos_msg.x = TCP_pos[0]
                tcp_pos_msg.y = TCP_pos[1]
                tcp_pos_msg.z = TCP_pos[2]
                self.tcp_pos_pub.publish(tcp_pos_msg)


                self.TCP_path.append(ee_pos)
                self.prev_eepos = ee_pos

                #robot_verts = self.franka.get_verts()   # Retrieve robots vertices - already a torch tensor 
                robot_verts = torch.empty(
                    (0, 3), 
                    device='cuda' if torch.cuda.is_available() else 'cpu', 
                    dtype=torch.float32)
                for link in self.franka.links:
                    if link.name != "link0":    #Disregard the base link
                        robot_verts = torch.cat((robot_verts, link.get_verts()), dim=0)  # Concatenate the vertices of all links
                obs_centers_tensor = torch.tensor(
                    self.obs_centers, 
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype=torch.float32)
                #Compute distances between the robot and the obstacles
                distances = torch.cdist(robot_verts, obs_centers_tensor)
                # Find the minimum distance
                min_distance = torch.min(distances).item()
                self.min_dists.append(min_distance)
                #print("Minimum distance to obstacles: ", min_distance)
                self.scene.step()    
                if started_recording:
                    self.cam.render()             
                planning_time = time.time() - self.start_time
                if (np.allclose(ee_pos, self.goal_pos, atol=1e-3) or (planning_time >= 60 and not self.evaluate) 
                or len(self.franka.detect_collision()) > 0):            # planning stops upon reaching the goal position, after 30s (except when evaluating) or if the robot collides
                    costs = compute_cost(self.TCP_path, self.min_dists, self.obs_radius, self.goal_pos)
                    msg = Float32MultiArray()
                    msg.data = costs           
                    self.cost_pub.publish(msg)         
                    if len(self.franka.detect_collision()) > 0:
                        print("Robot collisions detected!") 
                    # Append cost to a file and stop recording if evaluating
                    if self.evaluate:
                        # Determine paths based on whether Bayesian Optimization (BO) is enabled
                        base_path = f"/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/eval_costs/{self.scene_config}"
                        subfolder = "optimal" if self.bo else "predefined"
                        full_path = os.path.join(base_path, subfolder)
                        # Ensure the full directory path exists
                        os.makedirs(full_path, exist_ok=True)
                        cost_log_path = f"{base_path}/{subfolder}/cost_log.txt"
                        video_path = f"{base_path}/{subfolder}/video.mp4"
                        cost = np.sum(costs)  # Sum the individual costs
                        with open(cost_log_path, "a") as file:
                            file.write(f"{cost}\n")
                        # Stop recording and save the video
                        self.cam.stop_recording(save_to_filename=video_path, fps=15)
                    break
            self.rate.sleep()

if __name__ == "__main__":
    ik_controller = IK_Controller()
    ik_controller.run()