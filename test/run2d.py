import numpy as np
from Crypt2D import *
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

epsilon = 1e-3

# simulation parameters
X = 128
dt = 1e-3
n_batches = 100
phi_init = 0.5 
f_init = 0.5


T = 1e1
solver = StoEvolution2D(epsilon, D1, D2, l, g, k, a, v0, phi0, f0, n)
solver.initialise(X, T, dt, n_batches, phi_init, f_init)
solver.evolve(verbose=True)