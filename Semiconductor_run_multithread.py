import numpy as np
import pickle
from multiprocessing import Process, Manager
from Semiconductor_func import pop_solve
import warnings
warnings.filterwarnings("ignore")


def solve_pop(y, all_pops, grid_size, grid_delta, L0, q, n_modes, pump_width, n):
    pop = pop_solve(grid_size=grid_size, grid_delta=grid_delta, L0=L0, q=q, n_modes=n_modes, pump_width=pump_width, n=n,
                    pump_value_max=100000.0, n_pump_values=40, yoffset=y, plot=False)
    all_pops.append(pop)

if __name__ == "__main__":
    grid_size = 60.0*1e-6      # meters
    grid_delta = 1*1e-6      # meters
    L0 = 1.98*1e-6              # meters
    q = 10                     # longitudinal mode number
    n = 2.4
    # n=1.43
    n_modes = 50
    feature_RoC = 0.1   # meters
    feature_depth = 0.279*1e-6 # meters
    pump_width = 10 * 1e-6  # meters
    all_pops = Manager().list()
    yoffsets = np.linspace(0, 10e-6, 5)
    #L0s = np.linspace(1.96e-6, 1.988e-6, 6)
    processes = []
    for y in yoffsets:
        process = Process(target=solve_pop, args=(y, all_pops, grid_size, grid_delta, L0, q, n_modes, pump_width, n))
        process.start()
        processes.append(process)

    # for L in L0s:
    #     process = Process(target=solve_pop, args=(0, all_pops, grid_size, grid_delta, L, q, n_modes, pump_width, n))
    #     process.start()
    #     processes.append(process)

    for process in processes:
        process.join()

    with open('all_pop.pkl', 'wb') as f:
        # Dump
        pickle.dump(list(all_pops), f)
