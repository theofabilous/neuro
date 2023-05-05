#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cstdlib>
#include <iostream>
#include <utility>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Python.h>
#include <tuple>

extern "C"
{
	#include <numpy/arrayobject.h>
	#include <numpy/npy_math.h>
}

// Math
#include <cmath>
#include <numbers> // constants
#include <numeric> // accumulate, reduce. iota, etc.
#include <algorithm> // transform, ranges stuff, min, max, etc.

// Concurrency
#include <atomic>
#include <mutex>
#include <thread>


#define TYPE_CHECK 		0
#define STATIC_SIZES 	0


#if STATIC_SIZES
	#define maybe_constexpr constexpr
	#define Q N
#else
	#define maybe_constexpr
	#define Q Eigen::Dynamic
#endif

static maybe_constexpr double g_Na = 120., g_K = 36., g_L = 0.3;
static maybe_constexpr double E_Na = 115., E_K = -12., E_L = 10.613;
static maybe_constexpr double C = 1., A = 0.0238, R = 34.5;
static maybe_constexpr double _dx = 0.004;
static maybe_constexpr double D = 0.5;
static maybe_constexpr size_t N = D/_dx;
static maybe_constexpr double dx = D/N;
static maybe_constexpr int forward_step = 0;
static maybe_constexpr int correction   = 1;
static maybe_constexpr int verbosity   	= 0;

using array_ref = Eigen::Map<Eigen::Array<double, Q, 1>>;
using array 	= Eigen::Array<double, Q, 1>;
using matrix 	= Eigen::Matrix<double, Q, Q>;
using vector 	= Eigen::Vector<double, Q>;
using diagonal 	= Eigen::DiagonalMatrix<double, Q>;

#undef value_t
#undef Q

static inline
#if STATIC_SIZES
constexpr 
#endif
double get_Sigma(double dt, double _dx=dx)
{
	return (A*dt)/(2.*R*C * (_dx*_dx));
}

static inline matrix get_M_pre(double dt, double _dx=dx)
{
	matrix M = matrix::Zero(N, N);
	auto sigma = get_Sigma(dt, _dx);

	M += diagonal::Identity(N) * (1. + 2. * sigma);

	for (size_t i = 0; i < N-1; ++i)
	{
		M(i, i+1) = -sigma;
		M(i+1, i) = -sigma;
	}
	M(0,1) = -2.*sigma;
	M(N-1, N-2) = -2.*sigma;
	return M;
}



static inline auto A_m(auto& array)
{
	auto c = (array*-1. + 25.)*0.1;
	return c * (c.exp()-1.).inverse();
}
static inline auto B_m(auto& array)
{
	return (-1*array/18.).exp()*4.;
}

static inline auto A_n(auto& array)
{
	auto c = (array*-1. + 10.)*0.01;
	return c * (((c*10.).exp())-1.).inverse();
}
static inline auto B_n(auto& array)
{
	return ((array*-0.0125).exp())*0.125;
}

static inline auto A_h(auto& array)
{
	return ((array*-0.05).exp())*0.07;
}
static inline auto B_h(auto& array)
{
	return ( ((array*-1.+30.)*0.1).exp()+1. ).inverse();
}

template<typename ...T>
static bool check_dims(T... args)
{
	return (
		( PyArray_NDIM((PyArrayObject*)args) == 1 && 
		  PyArray_DIMS((PyArrayObject*)args)[0] == N ) 
		&& ...
	);
}

struct Gate
{
	static constexpr bool alpha = true, beta = false;
	static constexpr int M = 0, N = 1, H = 2;

	template<int gate>
	static inline auto A(auto& V_array)
	{
		if constexpr (gate == M)
			return A_m(V_array);
		else if constexpr (gate == N)
			return A_n(V_array);
		else if constexpr (gate == H)
			return A_h(V_array);
	}

	template<int gate>
	static inline auto B(auto& V_array)
	{
		if constexpr (gate == M)
			return B_m(V_array);
		else if constexpr (gate == N)
			return B_n(V_array);
		else if constexpr (gate == H)
			return B_h(V_array);
	}

	template<bool alpha_beta, int gate>
	static inline auto gating_function(auto& V_array)
	{
		if constexpr (alpha_beta == alpha)
			return A<gate>(V_array);
		else if constexpr (alpha_beta == beta)
			return B<gate>(V_array);
	}

	template<int gate>
	static inline auto implicit_step(auto& __restrict V_array, auto& __restrict W_array, double dt)
	{
		auto A_w = A<gate>(V_array);
		auto B_w = B<gate>(V_array);

		return (W_array + A_w*dt) * ((A_w+B_w)*dt + 1.).inverse();
	}

	template<int gate>
	static inline auto explicit_step(auto& __restrict V_array, auto& __restrict W_array, double dt)
	{
		auto A_w = A<gate>(V_array);
		auto B_w = B<gate>(V_array);

		return dt*(1.-W_array)*A_w + (1. - dt*B_w)*W_array;
	}

	template<typename T>
	static inline auto implicit_step(
		auto& __restrict V_array,
		T& 	  __restrict M_array,
		T& 	  __restrict N_array,
		T& 	  __restrict H_array,
		double dt
	)
	{
		return std::tuple{
			implicit_step<Gate::M>(V_array, M_array, dt),
			implicit_step<Gate::N>(V_array, N_array, dt),
			implicit_step<Gate::H>(V_array, H_array, dt)
		};
	}

	template<typename T>
	static inline auto explicit_step(
		auto& __restrict V_array,
		T& 	  __restrict M_array,
		T& 	  __restrict N_array,
		T& 	  __restrict H_array,
		double dt
	)
	{
		return std::tuple{
			explicit_step<Gate::M>(V_array, M_array, dt),
			explicit_step<Gate::N>(V_array, N_array, dt),
			explicit_step<Gate::H>(V_array, H_array, dt)
		};
	}
};

	
static inline double
get_next_V(
	PyArrayObject* __restrict npy_V,
	PyArrayObject* __restrict npy_M,
	PyArrayObject* __restrict npy_N,
	PyArrayObject* __restrict npy_H,
	double tolerance,
	double I,
	double t, 
	double dt
)
{
	array_ref 
		eig_V( static_cast<double*>( PyArray_DATA(npy_V) ), N ),
		eig_M( static_cast<double*>( PyArray_DATA(npy_M) ), N ),
		eig_N( static_cast<double*>( PyArray_DATA(npy_N) ), N ),
		eig_H( static_cast<double*>( PyArray_DATA(npy_H) ), N );

	double err = NPY_INFINITY;
	double prev_err = err;
	double prev_dt = dt;
	unsigned count = 0;

	double V_x0, M_x0, N_x0, H_x0;


	matrix 	M_pre 		= get_M_pre(dt),
			M 	 		= matrix::Zero(N, N);

	array 	Na 	 		= array::Zero(N),
		  	K 	 		= array::Zero(N),
		  	V_star 		= array::Zero(N),
		  	M_star 		= array::Zero(N),
		  	N_star 		= array::Zero(N),
		  	H_star 		= array::Zero(N)
				// ,
		  // 	A_w 		= array::Zero(N),
		  // 	B_w 		= array::Zero(N)
		;

	vector B_star 		= vector::Zero(N),
		   V_dist 		= vector::Zero(N),
		   M_dist 		= vector::Zero(N),
		   N_dist 		= vector::Zero(N),
		   H_dist 		= vector::Zero(N);
	do {
		Na = g_Na*eig_M.pow(3)*eig_H;
		K = g_K*eig_N.pow(4);

		M = M_pre;
		M += diagonal(Na+K+g_L)*(dt/C);

		B_star = eig_V + (dt/C) *(Na*E_Na +K*E_K + g_L*E_L);
		B_star(0) += I*(dt/(
			std::numbers::pi_v<double>*A*C*dx
		));
		V_star = M.inverse() * B_star;

		if (forward_step)
		{
			M_star = Gate::explicit_step<Gate::M>(V_star, eig_M, dt);
			N_star = Gate::explicit_step<Gate::N>(V_star, eig_N, dt);
			H_star = Gate::explicit_step<Gate::H>(V_star, eig_H, dt);
			// tie( M_star, N_star, H_star ) = 
			// 	// Gate::explicit_step(eig_V, eig_M, eig_N, eig_H, dt);
			// 	Gate::explicit_step(V_star, eig_M, eig_N, eig_H, dt);
		}
		else
		{
			M_star = Gate::implicit_step<Gate::M>(V_star, eig_M, dt);
			N_star = Gate::implicit_step<Gate::N>(V_star, eig_N, dt);
			H_star = Gate::implicit_step<Gate::H>(V_star, eig_H, dt);
			// tie( M_star, N_star, H_star ) = 
			// 	// Gate::implicit_step(eig_V, eig_M, eig_N, eig_H, dt);
			// 	Gate::implicit_step(V_star, eig_M, eig_N, eig_H, dt);
		}

		if (count == 0)
		{
			V_x0 = V_star(0);
			// M_x0 = M_star(0);
			// N_x0 = N_star(0);
			// H_x0 = H_star(0);
		}
		else
		{
			V_star(0) = V_x0;
			// M_star(0) = M_x0;
			// N_star(0) = N_x0;
			// H_star(0) = H_x0;
		}


		if (correction)
		{
			V_dist = V_star - eig_V;
			M_dist = M_star - eig_M;
			N_dist = N_star - eig_N;
			H_dist = H_star - eig_H;

			V_dist(0) = 0.;
			// M_dist(0) = 0.;
			// N_dist(0) = 0.;
			// H_dist(0) = 0.;

			err = std::max({
				V_dist.norm(),
				M_dist.norm(),
				N_dist.norm(),
				H_dist.norm()
			});

			if (err >= prev_err)
			{
				dt *= 0.8;
				M_pre = get_M_pre(dt);
				if (verbosity > 1)
					std::cout << "dt: " << dt << std::endl;
			}
			else if (err <= 0.8*tolerance and count < 2)
			{
				dt *= 1.25;
				M_pre = get_M_pre(dt);
			}
			count++;

			if (count % 1'000 == 0 && verbosity)
			{
				std::cout << "t: " << t << ", count: " << count << ", err: " << 
					  err << ", dt: " << dt << std::endl;
			}

			prev_err = err;
		}
		else err = -1.;

		eig_V = V_star;
		eig_M = M_star;
		eig_N = N_star;
		eig_H = H_star;

		if (count > 5 && verbosity > 1)
		{
			std::cout <<
				"Err: " 		<< err 		<< ", " <<
				"Prev err: " 	<< prev_err << ", " <<
				"dt: " 			<< dt 		<<
			std::endl;
		}

	} while (err > tolerance);

	return dt;
}


template<bool alpha_beta, int gate>
static inline PyObject* call_gating_function_impl(PyObject* self, PyObject* V)
{
	PyArrayObject *V_array = (PyArrayObject*)V;
#if TYPE_CHECK
	if (
		!PyArray_Check(V_array) || PyArray_TYPE(V_array) != NPY_DOUBLE || 
		!PyArray_IS_C_CONTIGUOUS(V_array) || PyArray_NDIM(V_array) != 1
	)
	{
		PyErr_SetString(PyExc_TypeError, "Argument is not a C-contiguous numpy array!");
		Py_RETURN_NONE;
	}
#endif

	// array eig_V(
	array_ref eig_V(
		static_cast<double*>( PyArray_DATA(V_array) ),
		PyArray_DIM(V_array, 0)
	);

	PyArrayObject *npy_result = (PyArrayObject*)PyArray_SimpleNew(
		PyArray_NDIM(V_array), PyArray_DIMS(V_array), NPY_DOUBLE
	);

	array_ref eig_result(
		static_cast<double*>( PyArray_DATA(npy_result) ),
		PyArray_DIM(npy_result, 0)
	);

	eig_result = Gate::gating_function<alpha_beta, gate>(eig_V);

	return (PyObject*) npy_result;
}


extern "C"
{
	
	static PyObject* call_A_m(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::alpha, Gate::M>(self, V);
	}
	static PyObject* call_B_m(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::beta, Gate::M>(self, V);
	}

	static PyObject* call_A_n(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::alpha, Gate::N>(self, V);
	}
	static PyObject* call_B_n(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::beta, Gate::N>(self, V);
	}

	static PyObject* call_A_h(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::alpha, Gate::H>(self, V);
	}
	static PyObject* call_B_h(PyObject* self, PyObject* V)
	{
		return call_gating_function_impl<Gate::beta, Gate::H>(self, V);
	}



	static PyObject* set_params(PyObject* self, PyObject *args, PyObject *kwargs)
	{
#if STATIC_SIZES
		PyErr_SetString(PyExc_TypeError, "Extension compiled with static parameters! Cannot set parameters.");
		Py_RETURN_NONE;
#else


		static char *kwlist[] = {
			(char*)"dx", 	(char*)"D", 	(char*)"C",   (char*)"A",
			(char*)"g_Na", 	(char*)"g_K", 	(char*)"g_L",
			(char*)"E_Na", 	(char*)"E_K", 	(char*)"E_L",
			(char*)"explicit", (char*)"correction", (char*)"verbosity",
			nullptr
		};

		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ddddddddddppi", kwlist,
			&_dx, &D, &C, &A, &g_Na, &g_K, &g_L, &E_Na, &E_K, &E_L, &forward_step,
			&correction, &verbosity))
		{
			PyErr_SetString(PyExc_TypeError, "Error parsing arguments!");
			Py_RETURN_NONE;
		}

		N = D/_dx;
		dx = D/N;

		PyObject *tuple = PyTuple_New(2);
		PyTuple_SetItem(tuple, 0, PyLong_FromLong(N));
		PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(dx));
		return tuple;
#endif
	}

	static PyObject* call_get_next_V(PyObject* self, PyObject *const *args, Py_ssize_t nargs)
	{
#if TYPE_CHECK
		if (nargs != 8)
		{
			PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments!");
			Py_RETURN_NONE;
		}
		PyObject* curr = nullptr;
		for(Py_ssize_t i=0; i<4; i++)
		{
			curr = args[i];
			if (!PyArray_Check(curr) || !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)curr))
			{
				PyErr_SetString(PyExc_TypeError, "Expected a C-contiguous array");
				Py_RETURN_NONE;
			}
		}
#endif
		PyArrayObject *V = (PyArrayObject*)args[0],
					  *m = (PyArrayObject*)args[1],
					  *n = (PyArrayObject*)args[2],
					  *h = (PyArrayObject*)args[3];
#if TYPE_CHECK
		for(int i=4; i<8; i++)
		{
			if (!PyFloat_Check(args[i]))
			{
				PyErr_SetString(PyExc_TypeError, "Expected a float for argument!");
				Py_RETURN_NONE;
			}
		}
#endif
		if(!check_dims(V, m, n, h))
		{
			PyErr_SetString(PyExc_TypeError, "Expected arrays of same size!");
			Py_RETURN_NONE;
		}

		double tolerance 	= PyFloat_AsDouble(args[4]),
			   I 			= PyFloat_AsDouble(args[5]),
			   t 			= PyFloat_AsDouble(args[6]),
			   dt 			= PyFloat_AsDouble(args[7]);

#if !STATIC_SIZES
		N = PyArray_DIM(V, 0);
#endif
		dt = get_next_V(V, m, n, h, tolerance, I, t, dt);

		return PyFloat_FromDouble(dt);
	}

	static PyObject* call_get_next_V_n(PyObject* self, PyObject *const *args, Py_ssize_t nargs)
	{
#if TYPE_CHECK
		if (nargs != 9 && nargs != 10)
		{
			PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments!");
			Py_RETURN_NONE;
		}
		PyObject* curr = nullptr;
		for(Py_ssize_t i=0; i<4; i++)
		{
			curr = args[i];
			if (!PyArray_Check(curr) || !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)curr))
			{
				PyErr_SetString(PyExc_TypeError, "Expected a C-contiguous array");
				Py_RETURN_NONE;
			}
		}
#endif
		PyArrayObject *V = (PyArrayObject*)args[0],
					  *m = (PyArrayObject*)args[1],
					  *n = (PyArrayObject*)args[2],
					  *h = (PyArrayObject*)args[3];
#if TYPE_CHECK
		for(int i=4; i<9; i++)
		{
			if (!PyFloat_Check(args[i]))
			{
				PyErr_SetString(PyExc_TypeError, "Expected a float for argument!");
				Py_RETURN_NONE;
			}
		}
		if (nargs == 10 && !PyBool_Check(args[9]))
		{
			PyErr_SetString(PyExc_TypeError, "Expected a boolean for argument #9!");
			Py_RETURN_NONE;
		}
#endif
		if(!check_dims(V, m, n, h))
		{
			PyErr_SetString(PyExc_TypeError, "Expected arrays of same size!");
			Py_RETURN_NONE;
		}

		double tolerance 	= PyFloat_AsDouble(args[4]),
			   I 		/*	= PyFloat_AsDouble(args[5])  */,
			   t 			= PyFloat_AsDouble(args[6]),
			   dt 			= PyFloat_AsDouble(args[7]),
			   t_int 		= PyFloat_AsDouble(args[8]);

		int get_dt_list = 0;

		if (nargs == 10)
			get_dt_list = PyObject_IsTrue(args[9]);

		PyObject* dt_list = nullptr;

		if (get_dt_list)
			dt_list = PyList_New(0);

		bool is_I_callable = PyCallable_Check(args[5]);
		PyObject *callable = nullptr;

		if (is_I_callable)
			callable = args[5];
		else
			I = PyFloat_AsDouble(args[5]);

		double dist = 0;
		while(dist < t_int)
		{
			if (is_I_callable)
				I = PyFloat_AsDouble (
					PyObject_CallOneArg(callable, PyFloat_FromDouble(t))
				);

			dt = get_next_V(V, m, n, h, tolerance, I, t, dt);
			if (get_dt_list)
				PyList_Append(dt_list, PyFloat_FromDouble(dt));
			dist += dt;
		}

		PyObject *tuple_return = PyTuple_New(get_dt_list ? 3 : 2);
		PyTuple_SetItem(tuple_return, 0, PyFloat_FromDouble(dt));
		PyTuple_SetItem(tuple_return, 1, PyFloat_FromDouble(dist));
		if (get_dt_list)
			PyTuple_SetItem(tuple_return, 2, dt_list);
		return tuple_return;
	}

	static PyMethodDef neuro_methods[] = {
		{"a_m", 		(PyCFunction) call_A_m, 		METH_O, NULL},
		{"b_m", 		(PyCFunction) call_B_m, 		METH_O, NULL},
                		              
		{"a_n", 		(PyCFunction) call_A_n, 		METH_O, NULL},
		{"b_n", 		(PyCFunction) call_B_n, 		METH_O, NULL},

		{"a_h", 		(PyCFunction) call_A_h, 		METH_O, NULL},
		{"b_h", 		(PyCFunction) call_B_h, 		METH_O, NULL},

		{"set_params", 	(PyCFunction)set_params, 		METH_VARARGS | METH_KEYWORDS, NULL},
		{"get_next", 	(PyCFunction)call_get_next_V, 	METH_FASTCALL, NULL},
		{"get_next_n", 	(PyCFunction)call_get_next_V_n, METH_FASTCALL, NULL},

		{NULL, NULL, 0, NULL}
	};

	static PyModuleDef neuro_module = {
		PyModuleDef_HEAD_INIT,
		"neuro",
		"Neuro module",
		-1,
		neuro_methods
	};

	PyMODINIT_FUNC PyInit_neuro(void)
	{
		import_array();
		return PyModule_Create(&neuro_module);
	}

}
