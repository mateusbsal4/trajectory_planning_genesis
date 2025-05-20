import genesis as gs
import numpy as np
import yaml
import os

# Static parameters
base_box_height = 0.5
base_box_width  = 0.4
desk_thickness  = 0.05
wall_height     = 2.0
wall_width      = 0.1
wall_length     = 5.0
r_pillar        = 0.15
box_edge        = 0.02
margin          = 0.05

def create_sphere(scene, pose, radius):
    scene.add_entity(
        gs.morphs.Sphere(pos=pose, fixed=True, radius=radius),
        surface=gs.surfaces.Plastic(color=(0.5,0.5,0.5,1))
    )

def create_cuboid(scene, pose, size):
    scene.add_entity(
        gs.morphs.Box(pos=pose[:3], euler=pose[3:], fixed=True, size=size),
        surface=gs.surfaces.Plastic(color=(0.5,0.5,0.5,1))
    )

def create_cylinder(scene, pose, radius, height):
    scene.add_entity(
        gs.morphs.Cylinder(pos=pose[:3], euler=pose[3:], fixed=True, height=height, radius=radius),
        surface=gs.surfaces.Metal(color=(0.5,0.5,0.5,1))
    )

def create_scene_from_config(config):
    """
    Populate the given Genesis `scene` with all environment entities described in `config`.
    Does not add the robot or camera.
    """
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0, 1.0), camera_lookat=(0, 0, 0.5),
            camera_fov=60, max_FPS=60
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False, show_FPS=False
    )
    scene.add_entity(gs.morphs.Plane())
    cam = scene.add_camera(res=(640,480), pos=(3,0,1.0), lookat=(0,0,0.5), fov=60, GUI=False)
    # Unpack config
    wp = tuple(config['wall_pos'])
    pp = tuple(config['pillar_pos'])
    dc = tuple(config['desk_center']); ds = tuple(config['desk_size'])
    sp = [tuple(x) for x in config['support_pillars']]
    cc = tuple(config['cube_center']); cbc = tuple(config['cracker_box_center'])
    # Base box
    create_cuboid(scene, pose=(0, 0, base_box_height / 2, 0, 0, 0), size=(0.3, base_box_width, base_box_height))
    # Wall
    create_cuboid(scene, pose=(wp[0], wp[1], wp[2], 0, 0, 0), size=(wall_width, wall_length, wall_height))
    # Pillar
    create_cylinder(scene, pose=(pp[0], pp[1], pp[2], 0, 0, 0), radius=r_pillar, height=wall_height)
    # Main desk
    create_cuboid(scene, pose=(dc[0], dc[1], dc[2], 0, 0, 0), size=ds)
    # Desk supports
    for supp in sp:
        create_cylinder(scene, pose=(supp[0], supp[1], supp[2], 0, 0, 0), radius=0.05, height=2 * supp[2])
    # Cube
    create_cuboid(scene, pose=(cc[0], cc[1], cc[2], 0, 0, 0), size=(box_edge, box_edge, box_edge))
    # Cracker box mesh (non-primitive, remains unchanged)
    scene.add_entity(
        gs.morphs.Mesh(
            file="/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/model/YCB/cracker_box/textured.obj",
            pos=cbc
        )
    )
    for p in range(config['n_floating_primitives']):
        # Cuboid
        pose = config['pose_cuboid' + str(p)]
        size = config['size_cuboid' + str(p)]
        create_cuboid(scene, pose, size)

        # Cylinder
        pose = config['pose_cylinder' + str(p)]
        radius = config['radius_cylinder' + str(p)]
        height = config['height_cylinder' + str(p)]
        create_cylinder(scene, pose, radius, height)
        # Sphere
        pose = config['pose_sphere' + str(p)]
        radius = config['radius_sphere' + str(p)]
        create_sphere(scene, pose, radius)
    # Optional boxes
    if 'box_1_center' in config:
        create_cuboid(scene, pose=tuple(config['box_1_center']) + (0, 0, 0), size=tuple(config['box_1_size']))
        create_cuboid(scene, pose=tuple(config['box_2_center']) + (0, 0, 0), size=tuple(config['box_2_size']))
    # Optional second desk
    if 'desk_2_center' in config:
        d2c = tuple(config['desk_2_center'])
        d2s = tuple(config['desk_2_size'])
        create_cuboid(scene, pose=d2c + (0, 0, 0), size=d2s)
        for supp in config['support_pillars_2']:
            s2 = tuple(supp)
            create_cylinder(scene, pose=s2 + (0, 0, 0), radius=0.05, height=2 * s2[2])
    # Optional second pillar
    if 'pillar2_center' in config:
        p2c = tuple(config['pillar2_center'])
        create_cylinder(scene, pose=p2c + (0, 0, 0), radius=config['pillar2_radius'], height=wall_height / 2)
    # Add robot
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0,0,0.4))
    )
    return scene, franka, cam

def create_scene(randomize=False, config_filename=None):
    cfg, goal_pos = setup_task(randomize=randomize, config_filename=config_filename)
    scene, franka, cam = create_scene_from_config(cfg)
    return scene, franka, cam, goal_pos

def setup_task(randomize=False, config_filename=None, include_in_dataset=False, n_floating_primitives=5):
    """
    Initialize office scene. Randomize placement if requested.
    Returns: scene, franka, cam, goal_pos
    """
    # Prepare config path
    base_dir = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs" if include_in_dataset else "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/scene"
    os.makedirs(base_dir, exist_ok=True)
    if config_filename is None:
        idx = 1
        while True:
            name = f"scene_{idx}.yaml"
            if not os.path.exists(os.path.join(base_dir, name)):
                config_filename = name
                break
            idx += 1
    config_path = os.path.join(base_dir, config_filename)
    wall_pos_default = (-0.5, 0.0, wall_height/2)
    pillar_pos_default = (-0.3, 0.6, wall_height/2)
    desk_center_default = (0.5, pillar_pos_default[1], base_box_height - desk_thickness/2)
    desk_size_default = (2*desk_center_default[0], 2*(desk_center_default[1]-base_box_width), desk_thickness)
    half_length = desk_size_default[0]/2
    half_width = desk_size_default[1]/2
    support1_default = (desk_center_default[0]/2, desk_center_default[1], (desk_center_default[2]-desk_thickness/2)/2)
    support2_default = (1.5*desk_center_default[0], desk_center_default[1], (desk_center_default[2]-desk_thickness/2)/2)
    cube_center_default = (box_edge/2,
                                   desk_center_default[1] + half_width - box_edge,
                                   desk_center_default[2] + desk_thickness/2 + box_edge/2)
    cracker_box_center_default = (0.05, desk_center_default[1], desk_center_default[2] + desk_thickness/2)
    goal_pos_default = [cube_center_default[0], cube_center_default[1], cube_center_default[2]]
    # Assign positions
    if not randomize:
        # Save YAML config
        goal_pos = goal_pos_default
        config = {
            'n_floating_primitives': 0,
            'wall_pos': wall_pos_default,
            'pillar_pos': pillar_pos_default,
            'desk_center': desk_center_default,
            'desk_size': desk_size_default,
            'support_pillars': [support1_default, support2_default],
            'cube_center': cube_center_default,
            'cracker_box_center': cracker_box_center_default,
            'goal_pos': goal_pos
        }
    else:
        #n_floating_primitives = 6    
        add_boxes = np.random.randn() > 0.5
        #add_boxes = False
        #add_second_desk = np.random.randn() > 0.5
        add_second_desk = False     
        #add_second_pillar = np.random.randn() > 0.5  
        add_second_pillar = False #enable for testing 
        # Random wall
        wall_x = np.random.uniform(-0.6, -0.3)
        wall_y = np.random.uniform(-0.2, 0.2)
        wall_pos = (wall_x, wall_y, wall_height/2)
        # Random pillar
        min_pillar_x = wall_x + wall_width/2 + r_pillar + margin
        min_pillar_y = base_box_width/2 + r_pillar + margin
        pillar_x = np.random.uniform(min_pillar_x, min_pillar_x+0.1)
        pillar_y = np.random.uniform(min_pillar_y, min_pillar_y+0.3)
        pillar_pos = (pillar_x, pillar_y, wall_height/2)
        # Random desk
        min_desk_center_x = pillar_x + r_pillar + margin + half_length
        min_desk_center_y = pillar_y + r_pillar + margin + half_width
        desk_center_x = np.random.uniform(min_desk_center_x, min_desk_center_x+0.2)
        desk_center_y = np.random.uniform(min_desk_center_y, min_desk_center_y+0.2)
        desk_center_z = base_box_height - desk_thickness/2
        desk_center = (desk_center_x, desk_center_y, desk_center_z)
        desk_size = desk_size_default
        # Random supports
        support1 = (desk_center_x - half_width + margin, desk_center_y, (desk_center_z - desk_thickness/2)/2)
        support2 = (desk_center_x + half_width - margin, desk_center_y, (desk_center_z - desk_thickness/2)/2)
        # Random cube on desk
        cube_center = (
            desk_center_x - np.random.uniform(0, half_length-box_edge),
            desk_center_y + np.random.uniform(box_edge, half_width-box_edge),
            desk_center_z + desk_thickness/2 + box_edge/2
        )
        # Random cracker box
        cracker_box_center = (
            desk_center_x - np.random.uniform(0, half_length-box_edge),
            desk_center_y - np.random.uniform(box_edge, half_width-box_edge),
            cube_center[2]
        )
        # Random goal
        goal_pos = [
            np.random.uniform(0.2,0.6),
            np.random.uniform(-0.4,desk_center_y - desk_size[1]/2 - 0.1),
            np.random.uniform(base_box_height/2, wall_height/2)
        ]
        base = np.array([0.0, 0.0, base_box_height])
        # Ensure goal position is within the robot´s reachable WS
        if np.linalg.norm(goal_pos - base) > 1.0:
            axis = np.random.choice([0, 1, 2])   
            diff = np.array(goal_pos) - base
            other_sq = diff[(axis+1)%3]**2 + diff[(axis+2)%3]**2
            goal_pos[axis] = float((np.sqrt(1 - other_sq) + base[axis])*np.sign(goal_pos[axis]))


        # Save YAML config
        config = {
            'n_floating_primitives': n_floating_primitives,
            'wall_pos': wall_pos,
            'pillar_pos': pillar_pos,
            'desk_center': desk_center,
            'desk_size': desk_size,
            'support_pillars': [support1, support2],
            'cube_center': cube_center,
            'cracker_box_center': cracker_box_center,
            'goal_pos': goal_pos
        }
        # Add random floating primitive shapes
        for p in range(n_floating_primitives):
            x_range = (0.2, 1.0)            #cant start at 0.0 otherwise obstacles might intersect with the robot
            y_range = (-2.0, desk_center_y - desk_size[1]/2)
            z_range = (0.0, wall_height)
            config['pose_cuboid'+str(p)] = (np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range),
                                                   np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))     
            config['size_cuboid'+str(p)] = (np.random.uniform(0.05, 0.1), np.random.uniform(0.05, 0.1), np.random.uniform(0.05, 0.1)) 
            config['pose_cylinder'+str(p)] = (np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range),
                                                     np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))
            config['radius_cylinder'+str(p)] = np.random.uniform(0.05, 0.1)
            config['height_cylinder'+str(p)] = np.random.uniform(0.1, 0.3)
            config['pose_sphere'+str(p)] = (np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)) 
            config['radius_sphere'+str(p)] = np.random.uniform(0.05, 0.1)
        # Optional boxes
        if add_boxes:
            pillar_left_edge_y = pillar_pos[1] - r_pillar
            wall_left_edge_y   = wall_pos[1] - wall_length/2
            box_1_width        = pillar_left_edge_y - wall_left_edge_y
            box_1_height       = base_box_height
            box_1_length       = abs(-base_box_width/2 - (wall_x + wall_width/2))
            box_1_center_x     = wall_x + wall_width/2 + box_1_length/2
            box_1_center_y     = -(box_1_width/2 - pillar_left_edge_y)
            box_1_center       = (box_1_center_x, box_1_center_y, box_1_height/2)
            box_1_size         = (box_1_length, box_1_width, box_1_height)

            box_2_width        = pillar_left_edge_y
            box_2_height       = wall_height/2 - base_box_height
            box_2_length       = box_1_length
            box_2_size         = (box_2_length, box_2_width, box_2_height)
            box_2_center       = (box_1_center_x, pillar_left_edge_y/2, base_box_height + box_2_height/2)
            config['box_1_center'] = box_1_center
            config['box_1_size']   = box_1_size
            config['box_2_center'] = box_2_center
            config['box_2_size']   = box_2_size
        # Optional second desk
        if add_second_desk:
            desk_2_size   = (desk_size_default[0], 2.5*desk_size_default[1], desk_thickness)
            desk_2_center = (
                desk_2_size[0]/2,
                -base_box_width/2 - desk_2_size[1]/2 - 0.2,
                base_box_height - desk_thickness/2 - 0.1
            )
            # supports for second desk
            support1_2 = (desk_2_center[0]/2, desk_2_center[1], (desk_2_center[2]-desk_thickness/2)/2)
            support2_2 = (1.5*desk_2_center[0], desk_2_center[1], (desk_2_center[2]-desk_thickness/2)/2)
            config['desk_2_center'] = desk_2_center
            config['desk_2_size'] = desk_2_size
            config['support_pillars_2'] = [support1_2, support2_2]
        if add_second_pillar:
            pillar2_radius = 0.1
            pillar2_center = (desk_size_default[0]/2, -pillar2_radius, wall_height/4)
            config['pillar2_radius'] = pillar2_radius
            config['pillar2_center'] = pillar2_center
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    return config, goal_pos


def recreate_task(config_filename, from_dataset=False):
    """
    Recreate scene from existing YAML config.
    """
    base_dir = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs" if from_dataset else "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/scene"
    os.makedirs(base_dir, exist_ok=True)
    config_filename = os.path.join(base_dir, config_filename)
    print("Loading config from", config_filename)
    with open(config_filename, 'r') as f:
        cfg = yaml.safe_load(f)
    print("cfg", cfg)
    # Build environment
    scene, franka, cam = create_scene_from_config(cfg)
    goal = list(cfg['goal_pos'])
    return scene, franka, cam, goal
