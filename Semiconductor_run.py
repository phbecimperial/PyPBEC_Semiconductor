from Semiconductor_func import pop_solve
import numpy as np

grid_size = 50.0*1e-6      # meters
grid_delta = 1*1e-6      # meters
L0 = 1.96*1e-6              # meters
q = 10                     # longitudinal mode number
n = 2.4
#n=1.43
n_modes = 50
feature_RoC = 0.1   # meters
feature_depth = 0.279*1e-6 # meters
pump_width = 5 * 1e-6  # meters


all_pops = []

L0 = np.linspace(1.96e-6, 2.1e-6, 6)

for L in L0:
    pop = pop_solve(grid_size=grid_size, grid_delta=grid_delta, L0=L, q=q, n_modes=n_modes, pump_width=pump_width, n=n, pump_value_max = 100000.0)
    all_pops.append(pop)