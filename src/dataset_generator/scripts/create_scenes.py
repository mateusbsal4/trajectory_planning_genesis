#import genesis as gs
import numpy as np
import yaml
import os
import argparse
import sys
sys.path.append('/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/src')
from genesis_inverse_kinematics.task_setup import setup_task


def main():
    parser = argparse.ArgumentParser(description="Generate scenes for dataset.")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear the current scene configs directory.")
    args = parser.parse_args()
    scene_configs_dir = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs"
    # Clear the directory if the -c flag is passed
    if args.clear:
        if os.path.isdir(scene_configs_dir):
            for file in os.listdir(scene_configs_dir):
                file_path = os.path.join(scene_configs_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all files   in {scene_configs_dir}")
        else:
            print(f"Directory {scene_configs_dir} does not exist.")
    num_scenes = 1000
    for i in range(num_scenes):
        n_floating_obs = np.random.randint(5, 10)
        setup_task(randomize = True, include_in_dataset=True, n_floating_primitives=n_floating_obs)

if __name__ == "__main__":
    main()  