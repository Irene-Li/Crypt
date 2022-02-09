import time
import json
from .Crypt import TimeEvolution
import mkl_fft
from . import pseudospectral as ps 

class StoEvolution1D(TimeEvolution): 

	def __init__(self, epsilon=None, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, n=None): 
		super().__init__(D1, D2, l, g, k, a, v0, phi0, n) 
		self.epsilon = epsilon 

	def evolve(self, verbose=False): 
		phi_init = mkl_fft.rfft(self.initial_state[:self.X])
		f_init = mkl_fft.rfft(self.initial_state[self.X:])
		params = self._collect_params() 
		self.y = ps.evolve_sto_ps_1d(phi_init, f_init, params).reshape((self.n_batches, self.X*2))

	def _collect_params(self): 
		params = super()._collect_params()
		params['epsilon'] = self.epsilon 
		return params 

	def _load_params(self, params):
		super()._load_params(params)
		self.epsilon = params['epsilon']











