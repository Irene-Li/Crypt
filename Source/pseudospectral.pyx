import numpy as np
import mkl_fft
cimport numpy as np
cimport cython
from cython.view cimport array
from libc.math cimport sqrt, fmin, M_PI, pow, exp 
from cython.parallel import prange

@cython.cdivision(True) 
@cython.nonecheck(False)
cdef double _hill_function(double f, int n) nogil: 
	cdef double temp 
	temp = pow(f, n) 
	return temp/(1+temp)

@cython.cdivision(True) 
@cython.nonecheck(False)
cdef double _nu(double phi, double v0, double phi0) nogil: 
	return v0*exp(-phi*phi/(2*phi0*phi0))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def evolve_sto_ps(np.ndarray[np.complex128_t, ndim=2] phi_init, np.ndarray[np.complex128_t, ndim=2] f_init, params):
	cdef np.ndarray[np.float64_t, ndim=4] y_evol
	cdef np.ndarray[np.complex128_t, ndim=2] phi, phi_cube, phi_sq, dW_phi
	cdef np.ndarray[np.complex128_t, ndim=2] f, dW_f 
	cdef np.ndarray[np.complex128_t, ndim=2] phi_x, phi_x_cube, phi_x_sq, phi_nl, f_nl 
	cdef np.ndarray[np.complex128_t, ndim=2] f_x 

	cdef Py_ssize_t n, i, j, m, batch_size, N=params['n']
	cdef double D1=params['D1'], D2=params['D2'], epsilon=params['epsilon']
	cdef double a=params['a'], l=params['l'], g=params['g'], k=params['k'] 
	cdef double v0=params['v0'], phi0=params['phi0']

	cdef Py_ssize_t X=params['X'], n_batches=params['n_batches']
	cdef double T=params['T'], dt=params['dt']
	cdef double kmax_half, kmax_two_thirds, kx, ky, ksq, factor
	cdef np.complex128_t temp, mu, h, nu 

	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	factor = M_PI*2.0/X

	phi_x_sq = np.empty((X, X), dtype='complex128')
	phi_x_cube = np.empty((X, X), dtype='complex128')
	phi_nl = np.empty((X, X), dtype='complex128')
	f_nl = np.empty((X, X), dtype='complex128')


	phi = phi_init 
	f = f_init 
	nitr = int(T/dt) 
	batch_size = int(nitr/n_batches)
	y_evol = np.empty((n_batches, 2, X, X), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_x = mkl_fft.ifft2(phi)
		f_x = mkl_fft.ifft2(f)

		if i % batch_size == 0:
			for j in xrange(X):
				for m in xrange(X):
					y_evol[n, 0, j, m] = phi_x[j, m].real
					y_evol[n, 1, j, m] = f_x[j, m].real 
			print('iteration: {},  phi mean: {}'.format(n, phi[0,0].real/(X*X)))
			n += 1

		for j in prange(X, nogil=True):
			for m in prange(X):
				temp = phi_x[j,m]
				phi_x_sq[j,m] = temp*temp
				phi_x_cube[j,m] = temp*temp*temp

				h = _hill_function(f_x[j,m].real, N)
				nu = _nu(temp.real, v0, phi0)
				phi_nl[j,m] = l*(2*h-1)*temp.real
				f_nl[j,m] = -g*h*temp.real + nu

		phi_cube = mkl_fft.fft2(phi_x_cube)
		phi_sq = mkl_fft.fft2(phi_x_sq)
		phi_nl = mkl_fft.fft2(phi_nl)
		f_nl = mkl_fft.fft2(f_nl)
		dW_phi = mkl_fft.fft2(np.random.normal(size=(X, X)))
		dW_f = mkl_fft.fft2(np.random.normal(size=(X, X)))

		for j in prange(X, nogil=True):
			for m in prange(X):
				kx = fmin(j, X-j)*factor
				ky = fmin(m, X-m)*factor
				ksq = kx*kx + ky*ky
				temp = phi[j,m]
				mu = (1/2 + a*ksq)*temp 
				if (kx<kmax_half) and (ky<kmax_half): 
					mu = mu + phi_cube[j,m]
				if (kx<kmax_two_thirds) and (ky<kmax_two_thirds):
					mu = mu - 3/2*phi_sq[j, m]

				dW_phi[j,m] = sqrt(2*(l+ksq*D1)*epsilon*dt)*dW_phi[j,m]
				dW_f[j,m] = sqrt(2*(g+ksq*D2)*epsilon*dt)*dW_f[j,m]

				phi[j,m] = dt*(phi_nl[j,m]-D1*ksq*mu) + dW_phi[j,m] + phi[j,m]
				f[j,m] =  dt*(f_nl[j,m]-D2*ksq*f[j,m]-k*f[j,m]) + dW_f[j,m] + f[j,m]

	return y_evol


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def evolve_sto_ps2(np.ndarray[np.complex128_t, ndim=2] phi_init, np.ndarray[np.complex128_t, ndim=2] f_init, params):
	cdef np.ndarray[np.float64_t, ndim=4] y_evol
	cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] phi, phi_cube, phi_sq
	cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] f, f_x
	cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] phi_x, phi_x_cube, phi_x_sq, phi_nl, f_nl 
	cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dW_phi, dW_f   

	cdef np.complex128_t *phi_ptr, *phi_cube_ptr, *phi_sq_ptr
	cdef np.complex128_t *f_ptr, *phi_nl_ptr, *f_nl_ptr 
	cdef np.float64_t *dW_phi_ptr, *dW_f_ptr

	cdef Py_ssize_t n, i, j, m, index, batch_size, N=params['n']
	cdef double D1=params['D1'], D2=params['D2'], epsilon=params['epsilon']
	cdef double a=params['a'], l=params['l'], g=params['g'], k=params['k'] 
	cdef double v0=params['v0'], phi0=params['phi0']

	cdef Py_ssize_t X=params['X'], n_batches=params['n_batches']
	cdef double T=params['T'], dt=params['dt']
	cdef double kmax_half, kmax_two_thirds, kx, ky, ksq, factor, phi_factor, f_factor 
	cdef np.complex128_t temp, mu, h, nu 

	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	factor = M_PI*2.0/X

	phi_x_sq = np.empty((X, X), dtype='complex128')
	phi_x_cube = np.empty((X, X), dtype='complex128')
	phi_nl = np.empty((X, X), dtype='complex128')
	f_nl = np.empty((X, X), dtype='complex128')

	phi = np.ascontiguousarray(phi_init)
	f = np.ascontiguousarray(f_init)

	nitr = int(T/dt) 
	batch_size = int(nitr/n_batches)
	y_evol = np.empty((n_batches, 2, X, X), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_x = mkl_fft.ifft2(phi)
		f_x = mkl_fft.ifft2(f)

		phi_ptr = &phi_x[0,0] 
		f_ptr = &f_x[0,0]
		phi_sq_ptr = &phi_x_sq[0,0]
		phi_cube_ptr = &phi_x_cube[0,0]
		phi_nl_ptr = &phi_nl[0,0] 
		f_nl_ptr = &f_nl[0,0]

		if i % batch_size == 0:
			for j in xrange(X):
				for m in xrange(X):
					index = j*X+m
					y_evol[n, 0, j, m] = phi_ptr[index].real
					y_evol[n, 1, j, m] = f_ptr[index].real 
			print('iteration: {},  phi mean: {}'.format(n, phi[0,0].real/(X*X)))
			n += 1

		for j in prange(X, nogil=True):
			for m in prange(X):
				index = j*X+m
				temp = phi_ptr[index]
				phi_sq_ptr[index] = temp*temp
				phi_cube_ptr[index]= temp*temp*temp

				h = _hill_function(f_ptr[index].real, N)
				nu = _nu(temp.real, v0, phi0)
				phi_nl_ptr[index] = l*(2*h-1)*temp.real
				f_nl_ptr[index] = -g*h*temp.real + nu

		phi_cube = mkl_fft.fft2(phi_x_cube)
		phi_sq = mkl_fft.fft2(phi_x_sq)
		phi_nl = mkl_fft.fft2(phi_nl)
		f_nl = mkl_fft.fft2(f_nl)
		dW_phi = np.random.normal(size=(X*X))
		dW_f = np.random.normal(size=(X*X))

		phi_ptr = &phi[0,0] 
		f_ptr = &f[0,0]
		phi_sq_ptr = &phi_sq[0,0]
		phi_cube_ptr = &phi_cube[0,0]
		dW_phi_ptr = &dW_phi[0]
		dW_f_ptr = &dW_f[0]

		for j in prange(X, nogil=True):
			for m in prange(X):
				index = j*X+m
				kx = fmin(j, X-j)*factor
				ky = fmin(m, X-m)*factor
				ksq = kx*kx + ky*ky
				temp = phi_ptr[index]
				mu = (0.5 + a*ksq)*temp 
				if (kx<kmax_half) and (ky<kmax_half): 
					mu = mu + phi_cube_ptr[index]
				if (kx<kmax_two_thirds) and (ky<kmax_two_thirds):
					mu = mu - 1.5*phi_sq_ptr[index]

				phi_ptr[index] += dt*(phi_nl_ptr[index]-D1*ksq*mu) 
				f_ptr[index] +=  dt*(f_nl_ptr[index]-(D2*ksq+k)*f_ptr[index]) 

				phi_factor = sqrt(X*(l+ksq*D1)*epsilon*dt) # noise prefactor as a result of FT 
				f_factor = sqrt(X*(g+ksq*D2)*epsilon*dt)

				if m == 0 or m == X/2: # zero imag part 
					phi_ptr[index] += phi_factor*dW_phi_ptr[index]
					f_ptr[index] += f_factor*dW_f_ptr[index]
				elif m < X - m: 
					phi_ptr[index] += phi_factor*(dW_phi_ptr[index]+1j*dW_phi_ptr[index+X/2])
					f_ptr[index] += f_factor*(dW_f_ptr[index]+1j*dW_f_ptr[index+X/2])
				else: # X-m < m 
					phi_ptr[index] += phi_factor*(dW_phi_ptr[j*X+X-m]-1j*dW_phi_ptr[j*X+X-m+X/2])
					f_ptr[index] += f_factor*(dW_f_ptr[j*X+X-m]-1j*dW_f_ptr[j*X+X-m+X/2])

	return y_evol