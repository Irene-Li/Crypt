import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Source.Crypt import *
import time

# parameters of the differential equation
D1 = 0.1
a = 1 

phi0 = 0.1 
n = 2

l = 1e-4 
g = l
k = l
v0 = 10*l

# simulation parameters
X = 128
dt = 1e-3
n_batches = 100
phi_init = 0.5 
f_init = 1


T = 3e5

for D in [0.01, 0.1, 1, 4, 10]:
	for x in [2, 5, 10]: 
		v0 = x*l 
		label = 'D={}_v0={}'.format(D, v0)

		solver = TimeEvolution(D1, D*D1, l, g, k, a, v0, phi0, n)
		solver.initialise(X, T, dt, n_batches, phi_init, f_init)
		solver.evolve(verbose=True)
		solver.save(label)
		solver.plot_phi_evol(label)
		solver.plot_f_evol(label)