#!/usr/bin/env python3
import rospy
import roslaunch
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import logging
from std_msgs.msg import Float32MultiArray

sys.path.append("/home/geriatronics/miniconda3/envs/ros_perception/lib/python3.9/site-packages")
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace



# Global variables to store costs
global_cost = None
individual_costs = None


logger = logging.getLogger("bayesian_optimizer")
logger.setLevel(logging.INFO)
# Create a file handler
file_handler = logging.FileHandler("/home/geriatronics/pmaf_ws/src/dataset_generator/logs/optimizer.log", mode='a')
file_handler.setLevel(logging.INFO)
# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)    
# Add the handler to the logger
logger.addHandler(file_handler)

def cost_callback(msg):
    """
    Callback to handle the received list of costs and compute the total cost.
    """
    global global_cost, individual_costs
    individual_costs = msg.data  # Store the list of costs
    global_cost = sum(individual_costs)  # Compute the total cost as the sum of individual costs

def launch_experiment(ns, scene, include_in_dataset):
    """
    Launch the IK_pmaf node and the planner node using their launch files.
    """
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # Launch IK node with the scene parameter
    ik_launch_file = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch"
    ik_args = [
        f"scene:={scene}", 
        f"dataset_scene:={'true' if include_in_dataset else 'false'}"
    ]
    ik_parent = roslaunch.parent.ROSLaunchParent(uuid, [(ik_launch_file, ik_args)])
    ik_parent.start()
    rospy.loginfo(f"IK node launched with scene: {scene}")
    time.sleep(5)  # Wait to ensure the IK node is running

    # Launch planner node
    planner_launch_file = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/launch/main_demo.launch"
    planner_args = []
    planner_parent = roslaunch.parent.ROSLaunchParent(uuid, [(planner_launch_file, planner_args)])
    planner_parent.start()
    rospy.loginfo("Planner node launched.")
    #time.sleep(10)  # Wait to ensure the planner node is running
    return ik_parent, planner_parent

def shutdown_experiment(ik_parent, planner_parent):
    ik_parent.shutdown()
    planner_parent.shutdown()
    rospy.loginfo("Experiment nodes shutdown.")

def run_experiment(ns, scene, include_in_dataset):
    """
    Launch the experiment nodes, wait for a cost message, and return the total cost.
    """
    global global_cost, individual_costs
    global_cost = None
    individual_costs = None
    ik_parent, planner_parent = launch_experiment(ns, scene, include_in_dataset)
    cost_sub = rospy.Subscriber("cost", Float32MultiArray, cost_callback)
    wait_time = 0
    print("Ik called")
    while global_cost is None:
        time.sleep(1)
        wait_time += 1
    
    total_cost = global_cost
    rospy.loginfo("Obtained total cost: {:.2f}".format(total_cost))
    rospy.loginfo("Individual costs: {}".format(individual_costs))
    shutdown_experiment(ik_parent, planner_parent)
    time.sleep(5)  # Allow nodes to shut down completely
    return total_cost, individual_costs

if __name__ == "__main__":
    rospy.init_node("bayes_optimizer_node", anonymous=True)
    scene = rospy.get_param("~scene")
    ns = rospy.get_param('~ns', '')
    if not ns:
        rospy.logerr("No namespace provided (param '~ns' is empty)!  Exiting.")
        sys.exit(1)
    include_in_dataset = rospy.get_param("~include_in_dataset", False)
    # Define a HEBO design space
    design_list = [{'name': 'detect_shell_rad', 'type': 'num', 'lb': 0.25, 'ub': 0.75}]
    for name, lb, ub in [
        ('k_a_ee', 1.0, 5.0),
        ('k_c_ee', 1.0, 5.0),
        ('k_r_ee', 1.0, 5.0),   
        ('k_d_ee', 1.0, 5.0),
        ('k_manip', 1.0, 5.0)
    ]:
        for j in range(7):
            design_list.append({'name': f'{name}_{j}', 'type': 'num', 'lb': lb, 'ub': ub})
    space = DesignSpace().parse(design_list)
    hebo_batch = HEBO(space, model_name='svgp', rand_sample=4) 
    # Temporary config file path
    temp_yaml_file = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
    #num_iterations = 9
    num_iterations = 7
    # Initialize cost tracking
    costs = []
    individual_costs_history = []
    best_cost = float('inf')    
    if not include_in_dataset:
        results_base_path = "/home/geriatronics/pmaf_ws/src/planner_optimizer/results/svgp"
        figures_base_path = "/home/geriatronics/pmaf_ws/src/planner_optimizer/figures/svgp"
        results_path = os.path.join(results_base_path, scene)
        figures_path = os.path.join(figures_base_path, scene)
    else:
        figures_path = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_cost_history"
        results_path = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/opt_results"
        labels_csv = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/labels.csv"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    start_time = time.time()
    best_it = 0
    it = 0
    for i in range(num_iterations):
        init_suggestion_instant = time.time()
        rec_x = hebo_batch.suggest(n_suggestions=8)
        logger.info(f"HEBO acq function optimization time: {time.time() - init_suggestion_instant}")
        #rec_x = hebo_batch.suggest(n_suggestions=2)
        rospy.loginfo("Iteration {}: Suggested parameters batch:".format(i))
        cost_list = []
        individual_costs_batch = []
        for j in range(len(rec_x)):
            single_x = rec_x.iloc[[j]]
            rospy.loginfo("Processing suggestion {} in batch:".format(j))
            rec_dict = single_x.to_dict(orient='records')[0]
            param_dict = {
                'detect_shell_rad': rec_dict['detect_shell_rad'],
                'agent_mass': 1.0,
                'agent_radius': 0.2,
                'velocity_max': 0.5,
                'approach_dist': 0.25,
                'k_a_ee': [rec_dict[f'k_a_ee_{k}'] for k in range(7)],
                'k_c_ee': [rec_dict[f'k_c_ee_{k}'] for k in range(7)],
                'k_r_ee': [rec_dict[f'k_r_ee_{k}'] for k in range(7)],
                'k_r_force': [0.0]*7,
                'k_d_ee': [rec_dict[f'k_d_ee_{k}'] for k in range(7)],
                'k_manip': [rec_dict[f'k_manip_{k}'] for k in range(7)]
            }
            with open(temp_yaml_file, 'w') as f:
                yaml.dump(param_dict, f)
            total_cost, indiv_costs = run_experiment(ns, scene, include_in_dataset)
            cost_list.append([total_cost])
            individual_costs_batch.append(indiv_costs)
            if total_cost < best_cost:
                best_it = it
                best_cost = total_cost 
                best_params = param_dict
            costs.append(total_cost)
            individual_costs_history.append(indiv_costs)
            # Save the best-found parameters and cost to a YAML file
            if not include_in_dataset:
                output_yaml_file = os.path.join(results_path, "best_parameters.yaml")
            best_params['best_cost'] = best_cost
            if not include_in_dataset:
                with open(output_yaml_file, 'w') as f:
                    yaml.dump(best_params, f, default_flow_style=False)
                rospy.loginfo(f"Best parameters and cost saved to {output_yaml_file}")
            it += 1
        cost_array = np.array(cost_list)
        gp_fitting_init = time.time()
        hebo_batch.observe(rec_x, cost_array)
        logger.info(f"HEBO gp fitting time: {time.time() - gp_fitting_init}")
        if not include_in_dataset:
            # Save the global cost evolution plot
            cost_plot_path = os.path.join(figures_path, "cost_evolution.png")
            plt.figure(figsize=(8, 6))
            plt.plot(costs, 'x-')
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost Evolution")
            plt.savefig(cost_plot_path)
            plt.close()
            # Save the individual cost components evolution plot
            individual_costs_plot_path = os.path.join(figures_path, "individual_costs_evolution.png")
            plt.figure(figsize=(8, 6))
            individual_costs_array = np.array(individual_costs_history)
            for idx, label in enumerate(["C_cl", "C_pl", "C_sm", "C_gd"]):
                plt.plot(individual_costs_array[:, idx], label=label)
            plt.xlabel("Iterations")
            plt.ylabel("Individual Costs")
            plt.title("Individual Costs Evolution")
            plt.legend()
            plt.savefig(individual_costs_plot_path)
            plt.close()
    end_time = time.time()
    if include_in_dataset:
        scene_file_path = os.path.join(results_path, scene + ".yaml")
        best_params['optimization_time_min'] = (end_time - start_time)/60  
        best_params['best_params_found_in_it'] = best_it         
        with open(scene_file_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        rospy.loginfo(f"Parameters for scene '{scene}' saved to {scene_file_path}")
        # Flatten best_params into a single list of values in the desired order
        header = [
            "scene",
            "detect_shell_rad",
            "agent_mass",
            "agent_radius",
            "velocity_max",
            "approach_dist",
        ]
        # Now add the seven entries for each k_* vector in order
        for name in ("k_a_ee", "k_c_ee", "k_r_ee", "k_d_ee", "k_manip"):
            for idx in range(1, 8):
                header.append(f"{name}_{idx}")
        # Build the row
        row = [
            scene,
            best_params["detect_shell_rad"],
            best_params.get("agent_mass", 1.0),
            best_params.get("agent_radius", 0.2),
            best_params.get("velocity_max", 0.5),
            best_params.get("approach_dist", 0.25),
        ]
        # Extend with each vector’s entries
        for name in ("k_a_ee", "k_c_ee", "k_r_ee", "k_d_ee", "k_manip"):
            vals = best_params[name]  # this is a list  of length 7
            row.extend(vals)
        # If file doesn’t exist or is empty, write header first
        write_header = not os.path.isfile(labels_csv) or os.path.getsize(labels_csv) == 0
        with open(labels_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        rospy.loginfo(f"Appended best params for scene '{scene}' to {labels_csv}")
        cost_plots_path = os.path.join(figures_path, scene + ".png")
        plt.plot(costs, 'x-')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost Evolution")
        plt.savefig(cost_plots_path)
        plt.close()
    

