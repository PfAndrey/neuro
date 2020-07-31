#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include "assert.h"
#include <vector>

template<typename T>
class Matrix
{
public:
	Matrix(const std::initializer_list <std::initializer_list<T>>& list):
		m_height(list.size()),
		m_width(list.begin()->size()),
		m_size(m_height*m_width)
	{
		m_data = new T[m_size];
		int i = 0;
		for (const auto& row : list)
			for (const auto& element : row)
				m_data[i++] = element;
	}

	Matrix(T* data, int width, int height) :
		m_height(height),
		m_width(width),
		m_size(m_height*m_width),
		m_data(data)
	{
	}

	~Matrix()
	{
		delete[] m_data;
	}

	static Matrix<T> fromVector(const std::vector<T>& input)
	{
		Matrix new_matrix(input.size(),1);
		for (int i = 0; i < input.size(); ++i)
			new_matrix.m_data[i] = input[i];
		return new_matrix;
	}

	Matrix(int width = 0, int height = 0, T def_value = T()) :
		m_width(width),
		m_height(height),
		m_size(width*height)
	{
		if (m_size != 0)
		{
			m_data = new T[m_size];
			for (int i = 0; i < m_size; ++i)
				m_data[i] = def_value;
		}
		else
		{
			m_data = nullptr;
		}
	}

	Matrix(const std::pair<int,int>& size, T def_value = T()) :
		Matrix(size.first,size.second, def_value)
	{
	}

	Matrix(const Matrix<T>& other) :
		m_width(other.m_width),
		m_height(other.m_height),
		m_size(other.m_width*other.m_height)
	{
		m_data = new T[m_size];
		for (int i = 0; i < m_size; ++i)
			m_data[i] = other.m_data[i];

	}

	Matrix(Matrix<T>&& other):
		m_width(other.m_width),
		m_height(other.m_height),
		m_size(m_width*m_height)
	{
		m_data = other.m_data;
		other.m_data = nullptr;
	}
	 
	Matrix<T>& operator=(const Matrix<T>& other)
	{
		if (m_width != other.m_width || m_height != other.m_height)
		{
			m_width = other.m_width;
			m_height = other.m_height;
			m_size = m_width * m_height;
			delete[] m_data;
			m_data = new T[m_size];
		}
		for (int i = 0; i < m_size; ++i)
			m_data[i] = other.m_data[i];

		return *this;
	}

	void resize(int width, int height)
	{
		if (width*height != m_width*m_height)
		{
			delete[] m_data;
			m_data = new T[width*height];
		}
		m_width = width;
		m_height = height;
		m_size = m_width * m_height;

		for (int i = 0; i < m_size; ++i)
			m_data[i] = T(0);
	}

	Matrix<T>& resize(const std::pair<int,int>& size, T def_value = T(0))
	{
		if (size.first*size.second != m_width * m_height)
		{
			delete[] m_data;
			m_data = new T[size.first*size.second];
		}
		m_width = size.first;
		m_height = size.second;
		m_size = m_width * m_height;

		for (int i = 0; i < m_size; ++i)
			m_data[i] = def_value;

		return *this;
	}

	Matrix<T>& operator=(Matrix<T>&& other)
	{
		m_width = other.m_width;
		m_height = other.m_height;
		m_size = m_width * m_height;
		delete[] m_data;
		m_data = other.m_data;
		other.m_data = nullptr;
		return *this;
	}

	T& operator()(int x, int y)
	{
		int id = y * m_width + x;
		assert(y < m_height && x < m_width);
		return m_data[id];
	}

	
	const T& operator()(int x, int y) const
	{
		int id = y * m_width + x;
		assert(y < m_height && x < m_width);
		return m_data[id];
	}

	T& operator[](int i)
	{
		assert(i < m_size);
		return m_data[i];
	}


	const T& operator[](int i) const
	{
		assert(i < m_size);
		return m_data[i];
	}

	Matrix<T> operator+(const Matrix<T>& other) const
	{
		assert(m_width == other.m_width && m_height == other.m_height);
		
		Matrix<T> new_matrix(*this);
		
		for (int i = 0; i < m_size; ++i)
			new_matrix.m_data[i] += other.m_data[i];

		return new_matrix;
	}

	Matrix<T> operator-(const Matrix<T>& other) const
	{
		assert(m_width == other.m_width && m_height == other.m_height);
		Matrix<T> new_matrix(*this);
		for (int i = 0; i < m_size; ++i)
			new_matrix.m_data[i] -= other.m_data[i];
		return new_matrix;
	}

	Matrix<T> operator-() const
	{
		Matrix<T> new_matrix(*this);
		for (int i = 0; i < m_size; ++i)
			new_matrix[i] *= -1;
		return new_matrix;
	}

	static void multi_dot(T* A, T* B, T* C, int from, int to, int A_width, int C_width)
	{		

		for (int row = from; row < to; ++row)
		{
			const int K1 = row * A_width;
			const int K2 = row * C_width;
			for (int col = 0; col < C_width; ++col)
			{
				T sum = 0;
				for (int i = 0; i < A_width; ++i)
					sum += B[i + K1] * C[col + i * C_width];
				A[col + K2] = sum;
			}
		}
	}

	Matrix<T> operator*(const Matrix<T>& other) const
	{
		assert(m_width == other.m_height);
		Matrix<T> new_matrix(other.m_width,m_height);
		multi_dot(new_matrix.m_data, m_data, other.m_data, 0, m_height, m_width, other.m_width);
		return new_matrix;
	}

	Matrix<T> operator*(const T& val) const
	{
		Matrix<T> new_matrix(*this);
		for (int i = 0; i < m_size; i++)
			new_matrix.m_data[i] *= val;
		return new_matrix;
	}

	Matrix<T> operator/(const T& val) const
	{
		Matrix<T> new_matrix(*this);
		for (int i = 0; i < m_size; i++)
			new_matrix.m_data[i] /= val;
		return new_matrix;
	}

	void operator+=(const Matrix<T>& other)
	{
		if (m_data == NULL)
		{
			m_data = new T[other.m_size];
			m_size = other.m_size;
			m_width = other.m_width;
			m_height = other.m_height;
			for (int i = 0; i < m_size; i++)
				m_data[i] = other.m_data[i];
			return;
		}
			for (int i = 0; i < m_size; i++)
				m_data[i] += other.m_data[i];
		
	}

	void operator-=(const Matrix<T>& other)
	{
		for (int i = 0; i < m_size; i++)
			m_data[i] -= other.m_data[i];
	}

	T det() const
	{
		assert(m_width == m_height && m_width);
		const auto& self(*this);

		if (m_width == 1)
		{
			return m_data[0];
		}
		else if (m_width == 2)
		{
			return self(0, 0)*self(1, 1) - self(1, 0)*self(0, 1);
		}

		Matrix<T> a = (*this);
		
		const int n = a.m_width;
		for (int i = 0; i < n - 1; i++)
			for (int j = i + 1; j < n; j++)
			{
				const T koef = a(j, i) / a(i, i);
				for (int k = i; k < n; k++)
					a(j, k) -= a(i, k) * koef;
			}

		T res = T(1);
		for (int i = 0; i < n; i++)
			res *= a(i, i);
		return res;
	}

	Matrix<T> transp() const
	{
		Matrix<T> new_matrix(m_height, m_width);

		for (int i = 0; i < m_size; ++i)
		{
			int y = i / m_width;
			int x = i % m_width;
			new_matrix.m_data[y + x*m_height] = m_data[x + y*m_width];
		}
		return new_matrix;
	}

	Matrix<T> minor(int mx, int my) const
	{
		Matrix<T> result(m_width - 1, m_height - 1);
		int dy = 0;
		for (int y = 0; y < m_height; ++y)
		{
			int dx = 0;
			if (my == y)
			{
				dy = -1;
				continue;
			}
			for (int x = 0; x < m_width; ++x)
			{
				if (mx == x)
				{
					dx = -1;
					continue;
				}
				result(x + dx, y + dy) = (*this)(x, y);
			}
		}
		return result;
	}

	Matrix<T> inverse() const
	{
		Matrix<T> new_matrix(m_width, m_height);

		for (int x = 0; x < m_width; ++x)
			for (int y = 0; y < m_height; ++y)
				new_matrix(x, y) = minor(x, y).det()*pow(-1, x + y);

		return new_matrix.transp() * (1.0 / det());
	}

	void print() const
	{
		for (int y = 0; y < m_height; ++y)
		{
			std::cout << "[";
			for (int x = 0; x < m_width; ++x)
			{
				std::cout << m_data[m_width*y + x] << ((x != m_width - 1) ? ',' : ']');
			}
			std::cout << std::endl;
		}
	}

	T module() const
	{
		T s = 0;
		for (int y = 0; y < m_size; ++y)
		{
			s += pow(m_data[y], 2);
		}
		return sqrt(s);
	}

	int cols() const
	{
		return m_width;
	}

	int rows() const
	{
		return m_height;
	}

	std::pair<int,int> size() const
	{
		return { m_width,m_height };
	}

	const T* const begin() const
	{
		return &m_data[0];
	}

	const T* const end() const
	{
		return &m_data[m_size];
	}

	T* begin()
	{
		return &m_data[0];
	}

	T* end()
	{
		return &m_data[m_size];
	}

	void fillRandom(T min, T max)
	{
		for (int x = 0; x < m_size; ++x)
		{
			m_data[x] = (max-min)*rand()/RAND_MAX + min;
		}
	}

	const Matrix<T>& fill(T val)
	{
		for (int x = 0; x < m_size; ++x)
			m_data[x] = val;
		return *this;
	}

	Matrix<T> wiseproduct(const Matrix<T>& second)
	{
		assert(m_width == second.m_width && m_height == second.m_height);
		Matrix<T> result(m_width, m_height);
		for (int i = 0; i < m_size; ++i)
			result.m_data[i] = m_data[i] * second.m_data[i];
		return result;
	}

	Matrix<T> wiseproduct(Matrix<T>&& second)
	{
		assert(m_width == second.m_width && m_height == second.m_height);
		Matrix<T> result(second.m_data, m_width, m_height);
		second.m_data = nullptr;
		for (int i = 0; i < m_size; ++i)
			result.m_data[i] *= m_data[i];
		return result;
	}
	
	int maxVectorIndex() const
	{
		assert(m_height == 1);

		T max = m_data[0];
		int index = 0;

		for (int i = 0; i < m_size; ++i)
			if (m_data[i] > max)
			{
				max = m_data[i];
				index = i;
			}
		return index;
	}

private:
	T* m_data = nullptr;
	int m_width = 0;
	int m_height = 0;
	int m_size = 0;
};

template<typename T, typename M>
static Matrix<T> MatrixFunctionAdapter(const Matrix<T>& matrix)
{
	M func;
	auto result = matrix;
	for (auto& element : result)
		element = (*func)(element);
	return result;
}

template<typename T>
using MLIST = std::vector<Matrix<T>>;

template<typename T>
static MLIST<T> operator+(const MLIST<T>& first, const MLIST<T>& second)
{
	assert(first.size() == second.size());
	MLIST<T> result(first.size());
	for (int i = 0; i < first.size(); ++i)
		result[i] = first[i] + second[i];
	return result;
}

template<typename T>
static void operator+=(MLIST<T>& first, const MLIST<T>& second)
{
	assert(first.size() == second.size());
	std::vector<Matrix<T>> result(first.size());
	for (int i = 0; i < first.size(); ++i)
		first[i] += second[i];
}

#endif // !MATRIX_H
