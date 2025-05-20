#import sys
#sys.path.append('../')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D  

from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from hebo.design_space.design_space import DesignSpace

import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# Branin objective function
# ---------------------------
def branin_objective(x  : pd.DataFrame) -> np.ndarray:
    """
    Compute the Branin function for each input (x0, x1) provided as a DataFrame.
    """
    X = x[['x0', 'x1']].values
    num_x = X.shape[0]
    
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    
    results = np.zeros((num_x, 1))
    for i in range(num_x):
        x0, x1 = X[i]
        results[i, 0] = a * (x1 - b*x0**2 + c*x0 - r)**2 + s*(1-t)*np.cos(x0) + s
    return results


# ---------------------------
# Define design space for optimization
# ---------------------------
space = DesignSpace().parse([
    {'name': 'x0', 'type': 'num', 'lb': -5, 'ub': 10},
    {'name': 'x1', 'type': 'num', 'lb': 0,  'ub': 15}
])


# ---------------------------
# Bayesian Optimization (BO) Loop
# ---------------------------
bo = BO(space, model_name='gp')
last_bo_rec = None  # Will store the last rec_x
start_time = time.time()
for i in range(64):
    rec_x = bo.suggest()
    last_bo_rec = rec_x.copy()  # store last suggestion
    precomputed = branin_objective(rec_x)
    bo.observe(rec_x, precomputed)
    if i % 4 == 0:
        print('BO Iter %d, best_y = %.2f' % (i, bo.y.min()))
print('BO time: %.2f seconds' % (time.time() - start_time))


# ---------------------------
# HEBO Sequential Loop (using GP model)
# ---------------------------
hebo_seq = HEBO(space, model_name='gp', rand_sample=4)  
last_hebo_seq_rec = None 
start_time = time.time()
for i in range(64):
    rec_x = hebo_seq.suggest(n_suggestions=1)
    last_hebo_seq_rec = rec_x.copy()
    precomputed = branin_objective(rec_x)
    hebo_seq.observe(rec_x, precomputed)
    if i % 4 == 0:
        print('HEBO Seq Iter %d, best_y = %.2f' % (i, hebo_seq.y.min()))
print("HEBO Sequential time: %.2f seconds" % (time.time() - start_time))


# ---------------------------
# HEBO GP Batch Loop
# ---------------------------
hebo_batch = HEBO(space, model_name='gp', rand_sample=4)
last_hebo_batch_rec = None
start_time = time.time()
for i in range(16):
    rec_x = hebo_batch.suggest(n_suggestions=8)
    last_hebo_batch_rec = rec_x.copy()
    precomputed_list = []
    for j in range(len(rec_x)):
        single_x = rec_x.iloc[[j]]
        precomputed = branin_objective(single_x)
        precomputed_list.append(precomputed)
    precomputed_array = np.vstack(precomputed_list)
    hebo_batch.observe(rec_x, precomputed_array)
    print('HEBO GP Batch Iter %d, best_y = %.2f' % (i, hebo_batch.y.min()))
print('HEBO GP Batch time: %.2f seconds' % (time.time() - start_time))


# ---------------------------
# HEBO RF Batch Loop
# ---------------------------
hebo_rf_batch = HEBO(space, model_name='rf', rand_sample=4)
last_hebo_rf_batch_rec = None
start_time = time.time()
for i in range(16):
    rec_x = hebo_rf_batch.suggest(n_suggestions=8)
    last_hebo_rf_batch_rec = rec_x.copy()
    precomputed_list = []
    for j in range(len(rec_x)):
        single_x = rec_x.iloc[[j]]
        precomputed = branin_objective(single_x)
        precomputed_list.append(precomputed)
    precomputed_array = np.vstack(precomputed_list)
    hebo_rf_batch.observe(rec_x, precomputed_array)
    print('HEBO RF Batch Iter %d, best_y = %.2f' % (i, hebo_rf_batch.y.min()))
print('HEBO RF Batch time: %.2f seconds' % (time.time() - start_time))


# ---------------------------
# HEBO SVIDKL Batch Loop
# ---------------------------
hebo_svgp_batch = HEBO(space, model_name='svgp', rand_sample=4)
last_hebo_svgp_batch_rec = None
start_time = time.time()
for i in range(16):
    rec_x = hebo_svgp_batch.suggest(n_suggestions=8)
    last_hebo_svgp_batch_rec = rec_x.copy()
    precomputed_list = []
    for j in range(len(rec_x)):
        single_x = rec_x.iloc[[j]]
        precomputed = branin_objective(single_x)
        precomputed_list.append(precomputed)
    precomputed_array = np.vstack(precomputed_list)
    hebo_svgp_batch.observe(rec_x, precomputed_array)
    print('HEBO SVGP Batch Iter %d, best_y = %.2f' % (i, hebo_svgp_batch.y.min()))
print('HEBO SVGP Batch time: %.2f seconds' % (time.time() - start_time))


# ---------------------------
# Compute the cumulative minimum (regret) over iterations
# ---------------------------   
conv_bo_seq         = np.minimum.accumulate(bo.y)
conv_hebo_seq       = np.minimum.accumulate(hebo_seq.y)
conv_hebo_batch     = np.minimum.accumulate(hebo_batch.y)
conv_hebo_rf_batch  = np.minimum.accumulate(hebo_rf_batch.y)
conv_hebo_svgp_batch = np.minimum.accumulate(hebo_svgp_batch.y)

ideal_pt = 0.397887

# Convergence Plot 
fig1 = plt.figure(figsize=(8, 6))
plt.semilogy(conv_hebo_svgp_batch[::8] - ideal_pt, 'x-', label='HEBO, Parallel, SVGP, Batch = 8')
plt.semilogy(conv_hebo_rf_batch[::8] - ideal_pt, 'x-', label='HEBO, Parallel, RF, Batch = 8')
plt.semilogy(conv_hebo_batch[::8] - ideal_pt, 'x-', label='HEBO, Parallel, GP, Batch = 8')
plt.semilogy(conv_hebo_seq - ideal_pt, 'x-', label='HEBO, Sequential')
plt.semilogy(conv_bo_seq - ideal_pt, 'x-', label='BO, LCB')
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.legend()
plt.title('Convergence Plot')


# ---------------------------
# 3D Plot of the Branin Function with Last Suggested Points
# ---------------------------
def branin(x0, x1):
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    return a * (x1 - b*x0**2 + c*x0 - r)**2 + s*(1-t)*np.cos(x0) + s

x0_range = np.linspace(-5, 10, 200)
x1_range = np.linspace(0, 15, 200)
X_mesh, Y_mesh = np.meshgrid(x0_range, x1_range)
Z_mesh = branin(X_mesh, Y_mesh)

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', edgecolor='none', alpha=0.8)
fig2.colorbar(surf, shrink=0.5, aspect=5)

#extract last suggested point 
def get_last_point(last_rec):
    if last_rec is None or last_rec.shape[0] == 0:
        return None
    return last_rec.iloc[0].values  

models = [
    ('BO', get_last_point(last_bo_rec)),
    ('HEBO Seq', get_last_point(last_hebo_seq_rec)),
    ('HEBO GP Batch', get_last_point(last_hebo_batch_rec)),
    ('HEBO RF Batch', get_last_point(last_hebo_rf_batch_rec)),
    ('HEBO SVGP Batch', get_last_point(last_hebo_svgp_batch_rec))
]

for label, pt in models:
    if pt is None:
        continue
    z_val = branin(pt[0], pt[1])
    ax.scatter(pt[0], pt[1], z_val, marker='^', s=80, label=label)

ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
ax.set_title('Branin function with suggested optima')
ax.legend()
plt.show()
