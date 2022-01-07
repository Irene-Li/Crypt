import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Source.Crypt2D import *
import time

# parameters of the differential equation
D1 = 0.25
D2 = 0.25
a = 1 

phi0 = 0.1
n = 4

l = 1e-4 
g = l
k = l
v0 = 10*l

epsilon = 1e-3

# simulation parameters
X = 64
dt = 1e-3
n_batches = 100
phi_init = 0.5 
f_init = 1


T = 1e4
solver = StoEvolution2D(epsilon, D1, D2, l, g, k, a, v0, phi0, n)
solver.initialise(X, T, dt, n_batches, phi_init, f_init)
start_time = time.time()
solver.evolve(verbose=True, cython=True)
end_time = time.time() 
print('time taken: ', end_time - start_time)
solver.save('test')




