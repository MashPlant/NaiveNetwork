#pragma once
#include <cmath>
#include "Matrix.h"
namespace nn
{
	struct Sigmoid
	{
		template <typename Decimal, int R, int C>
		Matrix<Decimal, R, C> operator()(const Matrix<Decimal, R, C> &m) const
		{
			auto ret = m;
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[i][j] = 1 / (1 + exp(-ret[i][j]));
			return ret;
		}

		template <typename Decimal>
		Decimal dAtY(Decimal y) const
		{
			return y - y * y;
		}
		template <typename Decimal, int R, int C>
		Matrix<Decimal, R, C> dAtY(const Matrix<Decimal, R, C> &y) const
		{
			auto ret = y;
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[i][j] = dAtY(ret[i][j]);
			return ret;
		}
	};

	struct SoftMax;

}