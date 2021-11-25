import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode
import json
import mkl_fft 

class TimeEvolution: 

	def __init__(self, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, f0=None, n=None):
		self.D1 = D1 
		self.D2 = D2 
		self.l = l   
		self.g = g   
		self.k = k   
		self.v0 = v0  
		self.phi0 = phi0 
		self.f0 = f0   
		self.n = n
		self.a = a

	def initialise(self, X, T, dt, n_batches, phi_init, f_init): # default dx=1 
		self.X = int(X) 
		self.T = T 
		self.dt = dt 
		self.size = self.X*2 
		self.n_batches = int(n_batches)
		self.step_size = T/(self.n_batches-1)
		self.batch_size = int(np.floor(self.step_size/self.dt))
		self.initial_state = np.zeros(self.size) + np.random.normal(size=self.size)/1e4
		self.initial_state[:self.X] += phi_init
		self.initial_state[self.X:] += f_init 
		
	def compute_Turing(self): 
		phi = np.mean(self.y[-2, :self.X])
		f = np.mean(self.y[-2, self.X])

		h = self._hill_function(f)
		h_prime = self._hill_prime(f)
		nu = self._nu(phi)
		J11 = self.l*(2*h-1) 
		J22 = -self.g*h_prime*phi-self.k 
		J12 = self.l*phi*(2*h_prime)
		J21 = -self.g*h+nu*(-phi/self.phi0**2)

		det = J11*J22 - J12*J21
		print(det)
		print(J11+J22)
		lhs = self.D1*J22+self.D2*J11 
		rhs = 2*np.sqrt(self.D1*self.D2*det)

		return lhs, rhs 

	def evolve(self, verbose=False): 
		self.y  = np.zeros((self.n_batches, self.size))
		k_array = np.array([min(i, self.X-i) for i in range(self.X)])*np.pi*2/self.X
		self.ksq = k_array*k_array
		kmax = np.pi
		self.kmax_half_mask = (k_array < kmax/2)
		self.kmax_two_thirds_mask = (k_array < kmax*2/3)

		small_batch = self.batch_size
		while small_batch > 1000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._delta_y).set_integrator('lsoda', atol=1e-8, nsteps=small_batch)
		r.set_initial_value(self.initial_state, 0)

		n = 0
		y = self.initial_state

		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					self.y[n] = y
					if verbose:
						print('iteration: {}	phi mean: {}, f mean: {}'.format(n, np.mean(y[:self.X]), np.mean(y[self.X:])))
					n += 1
				y = r.integrate(r.t+self.dt*small_batch)

	def plot_phi_evol(self, label): 
		plt.imshow(self.y[:, :self.X], origin='lower')
		plt.xlabel('x')
		plt.ylabel('t')
		plt.title('phi')
		plt.colorbar() 
		plt.savefig('{}_phi.pdf'.format(label))
		plt.close() 


	def plot_f_evol(self, label): 
		plt.imshow(self.y[:, self.X:], origin='lower')
		plt.xlabel('x')
		plt.ylabel('t')
		plt.title('f')
		plt.colorbar() 
		plt.savefig('{}_f.pdf'.format(label))
		plt.close() 



	def _hill_prime(self, f): 
		x = f/self.f0 
		xn = x**self.n 
		return (1/(1+xn)**2)*(x**(self.n-1))*self.n/self.f0 

	def _hill_function(self, f): 
		x = (f/self.f0)**self.n 
		return x/(1+x)

	def _nu(self, phi): 
		return self.v0*np.exp(-phi**2/(2*self.phi0**2))

	def _delta_y(self, t, y): 

		phi = y[:self.X]
		f = y[self.X:]
		phi_k = mkl_fft.fft(phi)
		phi_sq_k = mkl_fft.fft(phi*phi)
		phi_cb_k = mkl_fft.fft(phi*phi*phi)
		f_k = mkl_fft.fft(f) 

		h = self._hill_function(f)
		nu = self._nu(phi)
		mu = (1./2. + self.a*self.ksq)*phi_k 
		mu[self.kmax_two_thirds_mask] += -3./2.*phi_sq_k[self.kmax_two_thirds_mask] 
		mu[self.kmax_half_mask] += phi_cb_k[self.kmax_half_mask] 

		delta_phi = self.l*(2*h-1)*phi
		delta_f = -self.g*h*phi + nu - self.k*f 
		delta_phi += np.real(mkl_fft.ifft(-self.D1*self.ksq*mu))
		delta_f += np.real(mkl_fft.ifft(-self.D2*self.ksq*f_k))

		return np.concatenate((delta_phi, delta_f))

		



