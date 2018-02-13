#pragma once
#include <array>
#include <iostream>
#include <random>

namespace nn
{
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<float> d{ 0,1 };

	template <typename Decimal,int R,int C>
	class Matrix
	{
	private:
		typedef std::array<Decimal, C> row;
		std::array<row, R> M;
	public:
		Matrix() //array居然不会自动初始化，这很不stl
		{
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					M[i][j] = d(gen);
			//if (!std::is_class<Decimal>::value) //注意到只有基本类型的数组才不会初始化
				//memset(&M, 0, sizeof(M));
		}
		Matrix(const std::initializer_list<std::array<Decimal, C>> &lst)
		{
			for (int i = 0; i < R; ++i)
				M[i] = lst.begin()[i];
		}
		Matrix(const std::array<Decimal,C> &arr )
		{
			static_assert(R == 1, "Wrong Dimen");
			M[0] = arr;
		}
		constexpr static int r() { return R; }
		constexpr static int c() { return C; }
		row &operator[](int n) { return M[n]; } //获取某行
		const row &operator[](int n) const { return M[n]; }
		Matrix& operator+=(const Matrix &rhs) //模板天然保证了形状一致
		{
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					M[i][j] += rhs[i][j];
			return  *this;
		}
		template <typename K>
		Matrix operator+(K &&rhs) const  //只是为了触发universal reference
		{
			Matrix ret(rhs);
			return ret += *this;
		}
		Matrix operator-() const
		{
			Matrix ret(*this);
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[i][j] = -ret[i][j];
			return ret;
		}
		Matrix& operator-=(const Matrix &rhs) { return *this += (-rhs); }
		Matrix operator-(const Matrix &rhs) const { return *this + (-rhs); }
		template <int RC>
		Matrix<Decimal, R, RC> operator*(const Matrix<Decimal,C,RC> &rhs) const
		{
			Matrix<Decimal, R, RC> ret;
			//听说这样写对cache更友好?
			for (int k = 0; k < C; ++k)
				for (int i = 0; i < R; ++i)
				{
					Decimal tmp = M[i][k];
					const Decimal *p = rhs[k].data();
					for (int j = 0; j < RC; ++j)
						ret[i][j] += tmp * p[j];
				}
			return ret;
		}
		Matrix operator*(Decimal x) const
		{
			Matrix ret(*this);
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[i][j] *= x;
			return ret;
		}
		Matrix<Decimal, C, R> T() const
		{
			Matrix<Decimal, C, R> ret;
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[j][i] = M[i][j];
			return ret;
		}
		void fill(Decimal x)
		{
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					M[i][j] = x;
		}
		friend std::ostream& operator<<(std::ostream &os,const Matrix &rhs)
		{
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					os << rhs[i][j] << " \n"[j == C - 1];
			return os;
		}
	};

	template <typename Decimal, int R, int C>
	Matrix<Decimal, R, C> operator*(Decimal x, const Matrix<Decimal, R, C> &m) { return m * x; }

	template <typename Decimal, int R, int C>
	Matrix<Decimal, R, C> dot(const Matrix<Decimal, R, C> &lhs, const Matrix<Decimal, R, C> &rhs)
	{
		auto ret(lhs);
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < C; ++j)
				ret[i][j] *= rhs[i][j];
		return ret;
	}
	//template <typename Decimal, int R, int C>
	//Matrix<Decimal, R, C> operator+(Matrix<Decimal, R, C> &&lhs, const Matrix<Decimal, R, C> &rhs) { return rhs += lhs; }
}
