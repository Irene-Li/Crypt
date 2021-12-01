import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import json
from Crypt import TimeEvolution
import mkl_fft

class StoEvolution2D(TimeEvolution): 

	def __init__(self, epsilon=None, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, n=None): 
		super().__init__(D1, D2, l, g, k, a, v0, phi0, n) 
		self.epsilon = epsilon 



	def evolve(self, verbose=False): 
		self.y  = np.empty((self.n_batches, 2, self.X, self.X))
		self._make_kgrid() 

		n = 0
		y = self.initial_state

		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.y[n] = y
				if verbose:
					print('iteration: {}	phi mean: {}, f mean: {}'.format(n, np.mean(y[0]), np.mean(y[1])))
				n += 1
			delta = self._delta_y(y)*self.dt 
			noisy_delta = self._noisy_delta() 
			y += delta + noisy_delta 


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
		self.kmax_two_thirds_mask = (kx < kmax*2/3) & (ky < kmax/2) 

	def _delta_y(self, y): 
		phi = y[0]
		f = y[1] 
		phi_k = mkl_fft.fft2(phi)
		phi_sq_k = mkl_fft.fft2(phi*phi) 
		phi_cb_k = mkl_fft.fft2(phi**3)
		f_k = mkl_fft.fft2(f)

		h = self._hill_function(f)
		nu = self._nu(phi)
		mu = (1./2. + self.a*self.ksq)*phi_k 

		mu[self.kmax_two_thirds_mask] += -3./2.*phi_sq_k[self.kmax_two_thirds_mask] 
		mu[self.kmax_half_mask] += phi_cb_k[self.kmax_half_mask] 

		delta_phi = self.l*(2*h-1)*phi
		delta_f = -self.g*h*phi + nu - self.k*f 
		delta_phi += np.real(mkl_fft.ifft2(-self.D1*self.ksq*mu))
		delta_f += np.real(mkl_fft.ifft2(-self.D2*self.ksq*f_k))

		return np.stack((delta_phi, delta_f))


	def _noisy_delta(self):
		dW_phi = mkl_fft.fft2(np.random.normal(size=(self.X, self.X)))
		dW_f = mkl_fft.fft2(np.random.normal(size=(self.X, self.X)))
		dW_phi *= np.sqrt(2*(self.l+self.ksq*self.D1)*self.epsilon*self.dt)
		dW_f *= np.sqrt(2*(self.g+self.ksq*self.D2)*self.epsilon*self.dt) 
		dW_phi = np.real(mkl_fft.ifft2(dW_phi))
		dW_f = np.real(mkl_fft.ifft2(dW_f))
		return np.stack((dW_phi, dW_f))











