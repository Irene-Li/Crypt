import numpy as np
import sys, os 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Source.CryptSto1D import *
import time

# parameters of the differential equation
D1 = 0.25
D2 = 0.25
a = 1/D1 # want the fourth order derivative term to have coeff of 1 

phi0 = 0.1
n = 4

l = 3e-5
g = l
k = l
v0 = 10*l

epsilon = 1e-2

# simulation parameters
X = 128
dt = 1e-3
n_batches = 100
phi_init = 0.5
f_init = 1

T = 4e5
solver = StoEvolution1D(epsilon, D1, D2, l, g, k, a, v0, phi0, n)
solver.initialise(X, T, dt, n_batches, phi_init, f_init)

start_time = time.time()
solver.evolve(verbose=True)
end_time = time.time() 
print('time taken: ', end_time - start_time)
solver.save('lbda={}_epsilon={}'.format(l, epsilon))
