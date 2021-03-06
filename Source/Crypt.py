import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode 
from scipy.fftpack import rfftfreq 
import json
import mkl_fft 
from scipy.optimize import root_scalar

class TimeEvolution: 

	def __init__(self, D1=None, D2=None, l=None, g=None, k=None, a=None, v0=None, phi0=None, n=None):
		self.D1 = D1 
		self.D2 = D2 
		self.l = l   
		self.g = g   
		self.k = k   
		self.v0 = v0  
		self.phi0 = phi0 
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
		self._uniform_init(phi_init, f_init)


	def save(self, label): 
		np.save("{}_data.npy".format(label), self.y)

		params = self._collect_params() 

		with open('{}_params.json'.format(label), 'w') as f:
			json.dump(params, f)

	def _collect_params(self):
		params = {
			'T': self.T,
			'dt': self.dt,
			'X': self.X,
			'size': self.size, 
			'n_batches': self.n_batches,
			'step_size': self.step_size,
			'D1': self.D1,
			'D2': self.D2,
			'l': self.l,
			'g': self.g, 
			'k': self.k, 
			'v0': self.v0, 
			'phi0': self.phi0, 
			'n': self.n, 
			'a': self.a 
		}
		return params  

	def load(self, label):
		self.y = np.load('{}_data.npy'.format(label))

		with open('{}_params.json'.format(label), 'r') as f:
			params = json.load(f)

		self._load_params(params) 

	def _load_params(self, params):
		self.D1 = params['D1']
		self.D2 = params['D2']
		self.l = params['l']
		self.g = params['g']
		self.k = params['k']
		self.v0 = params['v0']
		self.phi0 = params['phi0']
		self.n = params['n']
		self.a = params['a']

		self.X = params['X']
		self.T = params['T']
		self.dt = params['dt']
		self.size = params['size']
		self.n_batches = params['n_batches']
		self.step_size = params['step_size']  

	def compute_uniform_ss(self): 
		ss1 = (0, self.v0/self.k)
		phi_bar = self._solve_phi_bar()
		ss2 = (phi_bar, 1)
		return ss1, ss2


	def _solve_phi_bar(self): 

		def f(phi):
			return self._nu(phi) - self.k - self.g/2*phi
		def fprime(phi):
			return self._nu(phi)**(-phi/self.phi0**2) - self.g/2
		sol = root_scalar(f, bracket=[0, 1], fprime=fprime)
		return sol.root 
		
	def compute_eigvals(self, phi, f): 

		h = self._hill_function(f)
		h_prime = self._hill_prime(f)
		nu = self._nu(phi)
		J11 = self.l*(2*h-1) 
		J22 = -self.g*h_prime*phi-self.k 
		J12 = self.l*phi*(2*h_prime)
		J21 = -self.g*h+nu*(-phi/self.phi0**2)

		return np.linalg.eigvals([[J11, J12], [J21, J22]])

	def evolve(self, verbose=False): 
		self.y  = np.zeros((self.n_batches, self.size))
		self._make_kgrid() 

		small_batch = self.batch_size
		while small_batch > 1000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._delta_y).set_integrator('lsoda', rtol=1e-5, nsteps=small_batch)


		phi_init = self.initial_state[:self.X]
		f_init = self.initial_state[self.X:]
		y_k = np.concatenate([mkl_fft.rfft(phi_init), mkl_fft.rfft(f_init)])
		r.set_initial_value(y_k, 0)
		n = 0

		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					self.y[n, :self.X] = np.real(mkl_fft.irfft(y_k[:self.X]))
					self.y[n, self.X:] = np.real(mkl_fft.irfft(y_k[self.X:]))
					if verbose:
						print('time step: {}	phi mean: {}'.format(n, np.real(y_k[0])/self.X))
					n += 1
				y_k = r.integrate(r.t+self.dt*small_batch)

	def plot_phi_evol(self, label, show=False): 
		plt.imshow(self.y[:, :self.X], origin='lower')
		plt.xlabel('x')
		plt.ylabel('t')
		plt.title('phi')
		plt.colorbar() 
		plt.savefig('{}_phi.pdf'.format(label))
		if show: 
			plt.show() 
		plt.close() 


	def plot_f_evol(self, label, show=False): 
		plt.imshow(self.y[:, self.X:], origin='lower')
		plt.xlabel('x')
		plt.ylabel('t')
		plt.title('f')
		plt.colorbar() 
		plt.savefig('{}_f.pdf'.format(label))
		if show: 
			plt.show() 
		plt.close() 


	def _uniform_init(self, phi_init, f_init):
		noise1 = np.random.normal(size=self.X)/1e4
		noise2 = np.random.normal(size=self.X)/1e4
		self.initial_state = np.concatenate([phi_init+noise1, f_init+noise2])

	def _hill_prime(self, f): 
		x = f
		xn = x**self.n 
		return (1/(1+xn)**2)*(x**(self.n-1))*self.n

	def _hill_function(self, f): 
		x = (f)**self.n 
		return x/(1+x)

	def _nu(self, phi): 
		return self.v0*np.exp(-phi**2/(2*self.phi0**2))

	def _delta_y(self, t, y): 

		phi_k = y[:self.X]
		f_k = y[self.X:]

		phi = mkl_fft.irfft(phi_k)
		f = mkl_fft.irfft(f_k)
		phi_sq_k = mkl_fft.rfft(phi*phi) 
		phi_cb_k = mkl_fft.rfft(phi*phi*phi)
		np.putmask(phi_sq_k, self.kmax_two_thirds_mask, 0)
		np.putmask(phi_cb_k, self.kmax_half_mask, 0)

		h = self._hill_function(f)
		nu = self._nu(phi)
		mu = (1./2. + self.a*self.ksq)*phi_k -(3/2)*phi_sq_k + phi_cb_k

		delta_phi = mkl_fft.rfft(self.l*(2*h-1)*phi) -self.D1*self.ksq*mu
		delta_f = mkl_fft.rfft(-self.g*h*phi + nu) - (self.k+self.D2*self.ksq)*f_k 

		return np.concatenate((delta_phi, delta_f))


	def _make_kgrid(self):
		k_array = rfftfreq(self.X)*np.pi*2
		self.ksq = k_array*k_array
		kmax = np.pi
		self.kmax_half_mask = (k_array >= kmax/2)
		self.kmax_two_thirds_mask = (k_array >= kmax*2/3) 






		



