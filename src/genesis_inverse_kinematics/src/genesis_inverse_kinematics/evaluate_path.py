import numpy as np

def compute_cost(TCP_path, min_dists, obs_radius, goal_pos):
    C_cl = 0  # clearance cost
    C_pl = 0  # path length cost 
    C_sm = 0  # smoothness cost 
    C_gd = 0  # goal deviation cost
    collision_penalty = 0  # penalty for collision, needs tuning 
    for i, (d_min, TCP_pos) in enumerate(zip(min_dists, TCP_path)):
        ## Clearance cost ##
        if d_min > 0:
           C_cl += 1.0/d_min
        else: # Collision detected
            C_gd = 10*np.linalg.norm(TCP_pos - goal_pos)
            C_sm *= 0.01
            C_cl *= 0.03
            C_pl *= 0.3
            print("Cost terms ")
            print("Clearance cost ", C_cl/i)
            print("Smoothness cost ", C_sm/i)
            print("Path length cost ", C_pl)
            print("Goal deviation cost ", C_gd)
            #J = (C_cl + C_sm)/i + C_pl + C_gd + collision_penalty
            #return J
            return [C_cl, C_pl, C_sm, C_gd]  # return individual costs
        ## Path length cost ##
        if i > 0:
            C_pl += np.linalg.norm(TCP_pos - TCP_path[i - 1])
        ## Smoothness cost ##
        if 0 < i < len(TCP_path) - 1:
            C_sm += np.linalg.norm(TCP_path[i + 1] - 2 * TCP_pos + TCP_path[i - 1]) ** 2
    C_cl /= len(TCP_path)  # normalize clearance cost 
    C_sm /= len(TCP_path)  #nsormalize smoothness cost 
    C_gd = 10*np.linalg.norm(TCP_path[-1] - goal_pos)  # reach goal cost
    C_sm *= 0.01
    C_cl *= 0.03
    C_pl *= 0.3
    print("Cost terms ")
    print("Clearance cost ", C_cl)
    print("Smoothness cost ", C_sm)
    print("Path length cost ", C_pl)
    print("Goal deviation cost ", C_gd)
    #J = C_cl + C_pl + C_sm + C_gd # combine costs
    #return J
    return [C_cl, C_pl, C_sm, C_gd]  # return individual costs