import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import json
from .Crypt import TimeEvolution
import mkl_fft
from . import pseudospectral as ps 

class StoEvolution2D(TimeEvolution): 

	def __init__(self, epsilon=None, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, n=None): 
		super().__init__(D1, D2, l, g, k, a, v0, phi0, n) 
		self.epsilon = epsilon 

	def evolve(self, verbose=False, cython=True): 
		phi_init = mkl_fft.fft2(self.initial_state[0])
		f_init = mkl_fft.fft2(self.initial_state[1])
		if cython: 
			params = self._collect_params() 
			self.y = ps.evolve_sto_ps2(phi_init, f_init, params)
		else:  
			self.y  = np.empty((self.n_batches, 2, self.X, self.X))
			self.evolve_python(phi_init, f_init, verbose)



	def evolve_python(self, phi_init, f_init, verbose): 
		self._make_kgrid() 

		n = 0
		y_k = np.empty((2, self.X, self.X), dtype=np.complex128)
		y_k[0] = phi_init
		y_k[1] = f_init 

		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.y[n, 0] = np.real(mkl_fft.ifft2(y_k[0]))
				self.y[n, 1] = np.real(mkl_fft.ifft2(y_k[1]))
				if verbose:
					print('iteration: {}	phi mean: {}'.format(n, np.real(y_k[0, 0, 0])/(self.X*self.X)))
				n += 1
			delta = self._delta_y(y_k)*self.dt 
			noisy_delta = self._noisy_delta() 
			y_k += delta + noisy_delta 

	def time_one_step(self): 
		self._make_kgrid() 
		y_k = np.empty((2, self.X, self.X), dtype=np.complex128)
		y_k[0] = mkl_fft.fft2(self.initial_state[0])
		y_k[1] = mkl_fft.fft2(self.initial_state[1])

		start_time = time.time() 
		delta = self._delta_y(y_k)*self.dt 
		end_time = time.time() 
		print('det step: ', end_time - start_time)

		start_time = time.time() 
		noisy_delta = self._noisy_delta() 
		end_time = time.time() 
		print('sto step: ', end_time - start_time)



	def _collect_params(self): 
		params = super()._collect_params()
		params['epsilon'] = self.epsilon 
		return params 

	def _load_params(self, params):
		super()._load_params(params)
		self.epsilon = params['epsilon']


	def _uniform_init(self, phi_init, f_init):
		self.initial_state = np.zeros((2, self.X, self.X))
		self.initial_state[0] += phi_init
		self.initial_state[1] += f_init  

	def _make_kgrid(self):
		kx = np.array([min(i, self.X-i) for i in range(self.X)])*np.pi*2/self.X
		kx = kx.reshape((self.X, 1))
		ky = kx.T 
		self.ksq = kx*kx+ky*ky 
		kmax = np.pi
		self.kmax_half_mask = (kx < kmax/2) & (ky < kmax/2)
		self.kmax_two_thirds_mask = (kx < kmax*2/3) & (ky < kmax*2/3) 

	def _delta_y(self, y): 
		phi_k = y[0]
		f_k = y[1] 

		phi = mkl_fft.ifft2(phi_k)
		f = mkl_fft.ifft2(f_k)
		phi_sq_k = mkl_fft.fft2(phi*phi) 
		phi_cb_k = mkl_fft.fft2(phi*phi*phi)
		np.putmask(phi_sq_k, self.kmax_two_thirds_mask, 0)
		np.putmask(phi_cb_k, self.kmax_half_mask, 0)

		h = self._hill_function(f)
		nu = self._nu(phi)
		mu = (1./2. + self.a*self.ksq)*phi_k -3/2*phi_sq_k + phi_cb_k

		delta_phi = mkl_fft.fft2(self.l*(2*h-1)*phi)-self.D1*self.ksq*mu
		delta_f = mkl_fft.fft2(-self.g*h*phi + nu)-self.D2*self.ksq*f_k-self.k*f_k

		return np.stack((delta_phi, delta_f))

	def _noisy_delta(self):
		dW_phi = mkl_fft.fft2(np.random.normal(size=(self.X, self.X)))
		dW_f = mkl_fft.fft2(np.random.normal(size=(self.X, self.X)))
		dW_phi *= np.sqrt(2*(self.l+self.ksq*self.D1)*self.epsilon*self.dt)
		dW_f *= np.sqrt(2*(self.g+self.ksq*self.D2)*self.epsilon*self.dt) 
		return np.stack((dW_phi, dW_f))











