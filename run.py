import numpy as np
from Crypt import *
import time

# parameters of the differential equation
D1 = 0.25
D2 = 0.25
a = 1 

phi0 = 0.1
f0 = 0.5  
n = 2



l = 1e-4 
g = l
k = l
v0 = 5*l

# simulation parameters
X = 128
dt = 1e-3
n_batches = 100
phi_init = 0.5 
f_init = 0.5


T = 3e5
solver = TimeEvolution(D1, D2, l, g, k, a, v0, phi0, f0, n)
solver.initialise(X, T, dt, n_batches, phi_init, f_init)
# solver.one_time_step()
solver.evolve(verbose=True)
solver.plot_phi_evol('test')
solver.plot_f_evol('test')