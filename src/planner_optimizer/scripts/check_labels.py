#!/usr/bin/env python3
import os
import re
import time
import yaml
import numpy as np
import shutil
import rospy
import roslaunch
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray

DEPTH2PTCLOUD_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud.launch'
SIM_STATIC_LAUNCH    = '/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch'
IK_LAUNCH            = '/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch'
PLANNER_LAUNCH       = '/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/launch/main_demo.launch'
TEMP_YAML_FILE = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
TOL = 0.04

goal_pos = np.zeros((3,1))
ee_pos   = np.zeros((3,1))
task_done = False

def cost_callback(msg):
    """
    Callback to handle the received list of costs and signal task completion.
    """
    global task_done
    task_done = True

def goal_pos_callback(msg):
    global goal_pos
    goal_pos[:] = [[msg.x], [msg.y], [msg.z-0.10365]]

def ee_pos_callback(msg):
    global ee_pos
    ee_pos[:] = [[msg.x], [msg.y], [msg.z]]

def launch_task(scene_name):
    """
    Launch depth->pointcloud, simulation, IK and planner for a given scene,
    wait until `cost_callback` fires, then shut everything down
    and return the Euclidean error between goal and end-effector.
    """
    global task_done
    task_done = False

    ns     = "/"
    uuid_n = roslaunch.rlutil.get_or_generate_uuid(None, False)

    # 1) launch perception
    depth_args = [f"ns:={ns}"]
    depth_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(DEPTH2PTCLOUD_LAUNCH, depth_args)])
    depth_parent.start()
    time.sleep(2)

    # 2) launch static sim
    sim_args = [f"ns:={ns}", "save_cloud:=false", f"scene:={scene_name}"]
    sim_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(SIM_STATIC_LAUNCH, sim_args)])
    sim_parent.start()
    time.sleep(2)

    # 3) launch IK
    ik_args = [f"scene:={scene_name}", "dataset_scene:=true"]
    ik_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(IK_LAUNCH, ik_args)])
    ik_parent.start()

    # 4) launch planner
    planner_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(PLANNER_LAUNCH, [])])
    planner_parent.start()
    rospy.loginfo("Planner node launched.")

    # wait for cost signal
    while not task_done:
        time.sleep(0.1)

    # shut everything down
    depth_parent.shutdown()
    sim_parent.shutdown()
    ik_parent.shutdown()
    planner_parent.shutdown()

    return np.linalg.norm(goal_pos - ee_pos)

if __name__ == "__main__":
    rospy.init_node("label_verifier", anonymous=True)

    # subscribers
    rospy.Subscriber("goal_position", Point, goal_pos_callback)
    rospy.Subscriber("tcp_pos",   Point, ee_pos_callback)
    rospy.Subscriber("cost",         Float32MultiArray, cost_callback)

    first_scene = int(rospy.get_param('~first_scene', 0))

    base_dir = "/home/geriatronics/pmaf_ws/src/dataset_generator/data"
    opt_results_dir = os.path.join(base_dir, "opt_results")
    opt_success_yml = os.path.join(base_dir, "opt_successfull.yaml")
    os.makedirs(os.path.dirname(opt_success_yml), exist_ok=True)

    # load existing results if any
    if os.path.exists(opt_success_yml):
        with open(opt_success_yml, 'r') as f:
            results = yaml.safe_load(f) or {}
    else:
        results = {}

    # gather scene files
    scene_files = sorted(
        [f for f in os.listdir(opt_results_dir) if f.endswith('.yaml')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )[first_scene:]

    for fname in scene_files:
        scene_name = os.path.splitext(fname)[0]
        rospy.loginfo(f"Starting scene '{scene_name}'")
        
        # copy scene YAML into TEMP_YAML_FILE before launching
        src_path = os.path.join(opt_results_dir, fname)
        try:
            shutil.copy(src_path, TEMP_YAML_FILE)
            rospy.loginfo(f"Copied {src_path} to {TEMP_YAML_FILE}")
        except Exception as e:
            rospy.logerr(f"Failed to copy {fname} to temp file: {e}")
            break

        try:
            error = launch_task(scene_name)
            print("Goal pos: ", goal_pos)
            print("hand end pos: ", ee_pos)            
            print("ERROR: ", error)
            success = bool(error < TOL)
            results[scene_name] = success
            # write updated YAML
            with open(opt_success_yml, 'w') as f:
                yaml.safe_dump(results, f)
            rospy.loginfo(f"Scene '{scene_name}' finished with error {error:.4f} → success={success}")
        except Exception as e:
            rospy.logerr(f"Failed on scene '{scene_name}': {e}")
            break
        time.sleep(1)
