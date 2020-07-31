#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

#include "Matrix.h"
#include "Math.h"
#include <algorithm>

#define MATRIX_FUNCTION(NAME,FUNC)					\
template<typename T>								\
static Matrix<T> ##NAME##(const Matrix<T>& in)		\
{													\
	Matrix<T> result(in);							\
	const int size = result.cols()*result.rows();	\
	for (int i = 0; i < size; ++i)					\
	{												\
		auto& x = in[i];							\
		auto& y = result[i];						\
		##FUNC##;									\
	}												\
	return result;									\
}

#define SIGM(P) 1.0/(1 + exp(-##P##))

MATRIX_FUNCTION(sigmoid, y = SIGM(x))
MATRIX_FUNCTION(sigmoid_d, y = SIGM(x)*(1- SIGM(x)) )

MATRIX_FUNCTION(tanH,	y = tan(x))
MATRIX_FUNCTION(tanH_d, y = 1 + tan(x)*tan(x))

MATRIX_FUNCTION(ReLU,	y = std::max(T(0),x))
MATRIX_FUNCTION(ReLU_d, y = (x<0)?0:1)

MATRIX_FUNCTION(leaky_ReLU, y = (x<0)? 0.01*x : x)
MATRIX_FUNCTION(leaky_ReLU_d, y = (x<0)? 0.01 : 1)

MATRIX_FUNCTION(swish, y = x*SIGM(x))
MATRIX_FUNCTION(swish_d, y = (exp(-x)*(x+1)+1)*pow(SIGM(x),2))



template<typename T>
static Matrix<T> softMax(const Matrix<T>& x)
{
	const auto size = x.cols()*x.rows();
	T max = x[0];
	for (int i = 0; i < size; ++i)
		if (x[i] > max) max = x[i];

	double scale = 0.0;
	for (int i = 0; i < size; ++i)
		scale += exp(x[i] - max);

	Matrix<T> result(x);

	for (int i = 0; i < size; ++i)
		result[i] = exp(x[i] - max) / scale;

	return result;
}
#endif // ! ACTIVATION_FUNCTION

