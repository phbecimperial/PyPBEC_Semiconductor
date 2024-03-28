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

n = 1*2.5
L0 = lamb0 * q / (2*n)       #cavity length, reconstructed to give us correct wavelengths.

n_modes = 50
feature_RoC = 100*1e-6   # This is 0.1 meters. Who knows why
feature_depth = 0.1 # meters

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
cavity_modes.plot_cavity(start_mode=0, plot=False) #If plot=False, will save instead.

#If you want to see more modes
# for i in tqdm(range(1, 12)):
#     cavity_modes.plot_cavity(start_mode=i*8, plot=False)

#Set up optical medium
from PyPBEC.OpticalMedium import OpticalMedium
QW = OpticalMedium(optical_medium="InGaAs_QW")
absorption_rates, emission_rates = QW.get_rates(lambdas=lambdas, mode=17)

#Normalise rates to improve performance
cavity_loss_rate = 1.0/(10*1e-12)     # in s^-1

cavity_loss_rates = np.ones(n_modes)
emission_rates = emission_rates / cavity_loss_rate
absorption_rates = absorption_rates / cavity_loss_rate

#Some decay rate
Gamma_down = 100.0

# Properties of the photonic modes
from PyPBEC.Cavity import Cavity
cavity = Cavity(M=n_modes, J=g.shape[1])
cavity.set_cavity_loss_rates(rates=cavity_loss_rates)
cavity.set_cavity_emission_rates(rates=emission_rates)
cavity.set_cavity_absorption_rates(rates=absorption_rates)

# Properties of the molecular modes
cavity.set_reservoir_decay_rates(rates=Gamma_down*np.ones(g.shape[1]))
cavity.set_reservoir_pump_rates(rates=np.reshape(pump, [pump.shape[0]*pump.shape[1]]))
cavity.set_reservoir_population(population=np.ones(cavity.J))


# Coupling between photonic and molecular modes
cavity.set_coupling_terms(coupling_terms=g)

#Set pump values
pump_value_min = 100.0
pump_value_max = 1000000.0
n_pump_values = 40

delta_p = (pump_value_max/pump_value_min)**(1/n_pump_values)-1
pumps = [(1+delta_p)**i*pump_value_min for i in range(0, n_pump_values)]


#Solve the thing
from PyPBEC.Solver import SteadyState
from tqdm import tqdm

populations = list()
for value in tqdm(pumps):
    # loads the pump onto the cavity object
    pump = value * (pump_base / np.sum(pump_base))
    cavity.set_reservoir_pump_rates(rates=np.reshape(pump, [pump.shape[0] * pump.shape[1]]))

    # Steady-state solver
    solver_steadystate = SteadyState(cavity_obj=cavity, ANNEALING=False)
    # Defines the initial guess
    if value == pumps[0]:
        solver_steadystate.set_initial_photons(initial_photons=np.array(np.zeros(cavity.M)))
        solver_steadystate.set_initial_excited_molecules(initial_excited_molecules=np.array(np.zeros(cavity.J)))
    else:
        solver_steadystate.set_initial_photons(initial_photons=np.array(solved_cavity_steadystate.photons[-1]))
        solver_steadystate.set_initial_excited_molecules(
            initial_excited_molecules=np.array(solved_cavity_steadystate.emols[-1]))

        # Calls the solver
    solved_cavity_steadystate = solver_steadystate.solve()

    # Sums the populations over the mode degeneracy, g=n+1, with n=0,1,2,...
    mode_degeneracy = np.array([j for j in range(0, n_modes) for i in range(0, j)][0:n_modes])
    steady_state_photon_population = [
        np.sum(solved_cavity_steadystate.photons[:, np.where(mode_degeneracy == mode_number)[0]], 1)
        for mode_number in list(set(list(mode_degeneracy)))]
    steady_state_photon_population = np.transpose(np.array(steady_state_photon_population))

    # Appends
    populations.append(steady_state_photon_population)
populations = np.array(populations)