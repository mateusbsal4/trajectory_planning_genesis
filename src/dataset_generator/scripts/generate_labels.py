#!/usr/bin/env python3
import signal
# Monkey-patch signal.signal before any roslaunch imports to avoid thread errors
signal.signal = lambda *args, **kwargs: None

import os, sys, re, time
import rospy, roslaunch
from threading import Thread

# --- Configuration ------------------------------------------------------
MAX_PARALLEL = 4
CONFIG_DIR   = '/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs'

# Launch file paths
DEPTH2PTCLOUD_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud.launch'
SIM_STATIC_LAUNCH    = '/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch'
OPTIMIZER_LAUNCH     = '/home/geriatronics/pmaf_ws/src/planner_optimizer/launch/bayesian_optimizer.launch'


def launch_scene(scene, global_idx):
    ns     = f"instance{global_idx}"
    uuid_n = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # Child threads inherit logging setup; no configure_logging here

    perception_args = [f"ns:={ns}"]
    depth_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(DEPTH2PTCLOUD_LAUNCH, perception_args)])
    depth_parent.start(); time.sleep(2)

    sim_parent   = roslaunch.parent.ROSLaunchParent(uuid_n, [(SIM_STATIC_LAUNCH, perception_args)])
    sim_parent.start();   time.sleep(2)

    rospy.loginfo(f"[Scene {scene}] Perception up in ns={ns}")

    opt_args   = [f"ns:={ns}", f"scene:={scene}", "include_in_dataset:=true"]
    opt_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(OPTIMIZER_LAUNCH, opt_args)])
    opt_parent.start()
    rospy.loginfo(f"[Scene {scene}] Optimizer up in ns={ns}")

    opt_parent.spin()
    rospy.loginfo(f"[Scene {scene}] Optimizer done in ns={ns}")

    depth_parent.shutdown()
    sim_parent.shutdown()
    rospy.loginfo(f"[Scene {scene}] Perception stopped in ns={ns}")


def run_batch(batch_idx, scenes_with_idx):
    for global_idx, scene in scenes_with_idx:
        try:
            launch_scene(scene, global_idx)
        except Exception as e:
            rospy.logerr(f"[Batch {batch_idx}] Failed on '{scene}': {e}")
            break
        time.sleep(1)
        rospy.loginfo(f"[Batch {batch_idx}] Finished '{scene}'")


def chunk_scenes(indexed_scenes):
    batches = [[] for _ in range(MAX_PARALLEL)]
    for idx, scene in indexed_scenes:
        batches[idx % MAX_PARALLEL].append((idx, scene))
    return batches


if __name__ == '__main__':
    rospy.init_node('dataset_generator_node', anonymous=True)
    # perform logging setup in main thread
    uuid_main = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid_main)

    first_scene = int(rospy.get_param('~first_scene', 0))

    if not os.path.isdir(CONFIG_DIR):
        rospy.logerr(f"Scene configs folder not found: {CONFIG_DIR}")
        sys.exit(1)

    files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.yaml')]
    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    indexed = [(i, os.path.splitext(f)[0]) for i, f in enumerate(sorted_files) if i >= first_scene]
    if not indexed:
        rospy.logwarn("No scenes after first_scene; exiting.")
        sys.exit(0)

    batches = chunk_scenes(indexed)

    # --- spawn one Thread per batch ---
    threads = []
    for batch_idx, scenes_with_idx in enumerate(batches):
        if not scenes_with_idx:
            continue
        t = Thread(target=run_batch, args=(batch_idx, scenes_with_idx), daemon=True)
        t.start()
        threads.append(t)

    # wait for all threads to finish
    for t in threads:
        t.join()

    rospy.loginfo("[dataset_generator] All scenes processed. Exiting.")
