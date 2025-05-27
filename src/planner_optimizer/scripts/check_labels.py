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
TEMP_YAML_FILE       = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
TOL = 0.04

goal_pos = np.zeros((3, 1))
ee_pos   = np.zeros((3, 1))
task_done = False

def cost_callback(msg):
    """
    Callback to handle the received list of costs and signal task completion.
    """
    global task_done
    task_done = True

def goal_pos_callback(msg):
    global goal_pos
    goal_pos[:] = [[msg.x], [msg.y], [msg.z - 0.10365]]

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


def sort_yaml_by_scene(yaml_path):
    """
    Read an existing YAML file of format {scene_name: bool, ...},
    sort keys by the integer suffix (scene_N), and rewrite the file
    so that scene_1 appears first, scene_2 second, etc.
    Returns the sorted dictionary.
    """
    if not os.path.exists(yaml_path):
        return {}

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    # Sort entries by the integer after "scene_"
    sorted_items = sorted(
        data.items(),
        key=lambda kv: int(re.search(r'_(\d+)$', kv[0]).group(1))
    )
    sorted_dict = {k: v for k, v in sorted_items}

    # Rewrite file in sorted order, preserving insertion order:
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(sorted_dict, f, sort_keys=False)

    return sorted_dict


if __name__ == "__main__":
    rospy.init_node("label_verifier", anonymous=True)

    # Subscribers
    rospy.Subscriber("goal_position", Point, goal_pos_callback)
    rospy.Subscriber("tcp_pos",     Point, ee_pos_callback)
    rospy.Subscriber("cost",        Float32MultiArray, cost_callback)

    # Read parameter for first_scene (integer scene ID)
    first_scene = int(rospy.get_param('~first_scene', 0))

    base_dir = "/home/geriatronics/pmaf_ws/src/dataset_generator/data"
    opt_results_dir = os.path.join(base_dir, "opt_results")
    opt_success_yml = os.path.join(base_dir, "opt_successfull.yaml")
    os.makedirs(os.path.dirname(opt_success_yml), exist_ok=True)

    # 1) On startup, sort any existing opt_successfull.yaml by scene number:
    results = sort_yaml_by_scene(opt_success_yml)

    # 2) List all "scene_NNN.yaml" files and sort by their numeric suffix:
    scene_files = sorted(
        (f for f in os.listdir(opt_results_dir) if f.endswith('.yaml')),
        key=lambda x: int(re.search(r'_(\d+)\.yaml$', x).group(1))
    )

    # 3) Filter to only those with NNN >= first_scene
    filtered_files = [
        f for f in scene_files
        if int(re.search(r'_(\d+)\.yaml$', f).group(1)) >= first_scene
    ]
    rospy.loginfo(f"First scene ID = {first_scene}, total to process = {len(filtered_files)}")

    for fname in filtered_files:
        scene_name = os.path.splitext(fname)[0]  # e.g. "scene_219"
        rospy.loginfo(f"Starting scene '{scene_name}'")

        # Copy the scene YAML into TEMP_YAML_FILE before launching
        src_path = os.path.join(opt_results_dir, fname)
        try:
            shutil.copy(src_path, TEMP_YAML_FILE)
            rospy.loginfo(f"Copied {src_path} to {TEMP_YAML_FILE}")
        except Exception as e:
            rospy.logerr(f"Failed to copy {fname} to temp file: {e}")
            break

        try:
            error = launch_task(scene_name)
            rospy.loginfo(f"Goal pos: {goal_pos.flatten()}")
            rospy.loginfo(f"End-effector pos: {ee_pos.flatten()}")
            rospy.loginfo(f"ERROR: {error:.6f}")

            success = bool(error < TOL)
            results[scene_name] = success

            # Re-sort the entire results dict by scene number, then re-dump
            sorted_items = sorted(
                results.items(),
                key=lambda kv: int(re.search(r'_(\d+)$', kv[0]).group(1))
            )
            sorted_dict = {k: v for k, v in sorted_items}
            with open(opt_success_yml, 'w') as f:
                yaml.safe_dump(sorted_dict, f, sort_keys=False)

            rospy.loginfo(f"Scene '{scene_name}' → error={error:.4f}, success={success}")

        except Exception as e:
            rospy.logerr(f"Failed on scene '{scene_name}': {e}")
            break

        time.sleep(1)
