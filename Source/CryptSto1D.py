import time
import json
from .Crypt import TimeEvolution
import mkl_fft
from . import pseudospectral as ps 
import numpy as np 

class StoEvolution1D(TimeEvolution): 

	def __init__(self, epsilon=None, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, n=None): 
		super().__init__(D1, D2, l, g, k, a, v0, phi0, n) 
		self.epsilon = epsilon 

	def evolve(self, verbose=False, cython=True): 
		phi_init = mkl_fft.rfft(self.initial_state[:self.X])
		f_init = mkl_fft.rfft(self.initial_state[self.X:])

		if cython: 
			params = self._collect_params()
			self.y = ps.evolve_sto_ps_1d(phi_init, f_init, params).reshape((self.n_batches, self.X*2))
		else:
			self.y = np.empty((self.n_batches, 2*self.X))
			self.evolve_python(phi_init, f_init, verbose)

	def _collect_params(self): 
		params = super()._collect_params()
		params['epsilon'] = self.epsilon 
		return params 

	def _load_params(self, params):
		super()._load_params(params)
		self.epsilon = params['epsilon']

	def evolve_python(self, phi_init, f_init, verbose): 
		self._make_kgrid() 

		n = 0
		y_k = np.concatenate((phi_init, f_init))

		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.y[n, :self.X] = mkl_fft.irfft(y_k[:self.X])
				self.y[n, self.X:] = mkl_fft.irfft(y_k[self.X:])
				if verbose:
					print('iteration: {}	phi mean: {}'.format(n, y_k[0]/self.X))
				n += 1
			delta = self._delta_y(0, y_k)*self.dt 
			noisy_delta = self._noisy_delta() 
			y_k += delta + noisy_delta 


	def _noisy_delta(self):
		dW_phi = np.random.normal(size=(self.X))
		dW_f = np.random.normal(size=(self.X))
		dW_phi *= np.sqrt(2*self.X*(self.l+self.ksq*self.D1)*self.epsilon*self.dt)
		dW_f *= np.sqrt(2*self.X*(self.g+self.ksq*self.D2)*self.epsilon*self.dt) 
		return np.concatenate((dW_phi, dW_f))












