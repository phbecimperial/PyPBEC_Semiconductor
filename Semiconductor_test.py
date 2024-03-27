import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from tqdm import tqdm



grid_size = 40.0*1e-6     # meters
grid_delta = 0.5*1e-6      # meters
q = 17                     # longitudinal mode number
lamb0 = 950e-9             #meters
L0 = lamb0 * q / 2         #cavity length

n = 1
n_modes = 100
feature_RoC = 100*1e-6   # This is 0.1 meters. Who knows why
feature_depth = 1 # meters

from PyPBEC.Cavity import Modes
cavity_modes = Modes(grid_size=grid_size, grid_delta=grid_delta, L0=L0, q=q, n=n, n_modes=n_modes)
cavity_modes.set_geometry_elliptical(RoC=feature_RoC, depth=feature_depth, anistropy_factor = 1.001)
lambdas, modes = cavity_modes.compute_cavity_modes()
g = cavity_modes.get_coupling_matrix()
print(lambdas)
#%%
#Define pump laser, and compare to cavity modes
pump_width = 2.5*1e-6       # meters

X, Y = cavity_modes.get_cavity_grid()
pump_base = np.exp(-((X)**2+Y**2) / pump_width**2)
pump = 1*(pump_base/np.sum(pump_base))
cavity_modes.load_pump(pump=pump)
cavity_modes.plot_cavity(start_mode=0, plot=False)

#If you want to see more modes
for i in tqdm(range(1, 12)):
    cavity_modes.plot_cavity(start_mode=i*8, plot=False)

from PyPBEC.OpticalMedium import OpticalMedium
QW = OpticalMedium(optical_medium="InGaAs_QW")
absorption_rates, emission_rates = QW.get_rates(lambdas=lambdas, mode=17)


