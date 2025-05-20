#!/usr/bin/env python3
import rospy, roslaunch 
import os, re, sys, time
DEPTH2PTCLOUD_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud.launch'
SIM_STATIC_LAUNCH    = '/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch'



def launch_scene(scene: str):
    ns     = "/"
    uuid_n = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # Child threads inherit logging setup; no configure_logging here

    perception_args = [f"ns:={ns}"]
    depth_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(DEPTH2PTCLOUD_LAUNCH, perception_args)])
    depth_parent.start() 
    time.sleep(2)
    sim_args = [f"ns:={ns}", "save_cloud:=true", f"scene:={scene}"]
    sim_parent   = roslaunch.parent.ROSLaunchParent(uuid_n, [(SIM_STATIC_LAUNCH, sim_args)])
    sim_parent.start()
    time.sleep(2)

    # 2) launch IK Genesis (no namespace)
    ik_launch = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch"
    ik_args = [f"scene:={scene}", "dataset_scene:=true"]
    ik_parent = roslaunch.parent.ROSLaunchParent(
        uuid_n, [(ik_launch, ik_args)]
    )
    ik_parent.start()
    rospy.loginfo(f"Launched IK node for scene '{scene}'")
    scene_path = f"/home/geriatronics/pmaf_ws/src/dataset_generator/data/inputs/{scene}.ply"
    while not os.path.exists(scene_path):
        time.sleep(0.1)

    # shutdown nodes
    sim_parent.shutdown()
    depth_parent.shutdown()
    ik_parent.shutdown()

if __name__ == "__main__":
    rospy.init_node("input_generator_node", anonymous=True)

    first_scene = int(rospy.get_param('~first_scene', 0))

    config_dir = '/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs'
    if not os.path.isdir(config_dir):
        rospy.logerr(f"Scene configs folder not found: {config_dir}")
        sys.exit(1)

    # sort by scene index
    scene_files = sorted(
        (f for f in os.listdir(config_dir) if f.endswith('.yaml')),
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    scene_files = scene_files[first_scene:]

    for fname in scene_files:
        scene_name = os.path.splitext(fname)[0]
        rospy.loginfo(f"[dataset_generator] Starting capture for '{scene_name}'")
        try:
            launch_scene(scene_name)
        except Exception as e:
            rospy.logerr(f"Failed on scene '{scene_name}': {e}")
            break
        time.sleep(1)
        rospy.loginfo(f"[dataset_generator] Finished scene '{scene_name}'")
