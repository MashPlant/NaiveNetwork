#pragma once
#include <array>
#include <iostream>
#include <random>

namespace nn
{
	template <typename Decimal,int R,int C>
	class Matrix
	{
	private:
		typedef std::array<Decimal, C> row;
		std::array<row, R> M;
	public:
		Matrix() //array居然不会自动初始化，这很不stl
		{
			if (!std::is_class<Decimal>::value)
				memset(&M, 0, sizeof(M));
		}
		explicit Matrix(Decimal xaiverRange)
		{
			static std::random_device rd;
			static std::mt19937 gen(rd()); //这一步是设置种子，利用rd产生的一个int为种子
			std::uniform_real_distribution<Decimal> d(-xaiverRange, xaiverRange); //均匀分布发生器，但它只是一个壳子，需要随机数引擎gen才能发挥作用
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					M[i][j] = d(gen);
		}
		Matrix(const std::initializer_list<std::array<Decimal, C>> &lst)
		{
			for (int i = 0; i < R; ++i)
				M[i] = lst.begin()[i];
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
		template <typename Pred>
		Decimal reduce(Pred pred ,Decimal origin = 0) const
		{
			Decimal ret = origin;
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret = pred(ret, M[i][j]);
			return ret;
		}
		template <typename Pred>
		Matrix map(Pred pred) const
		{
			Matrix ret(*this);
			for (int i = 0; i < R; ++i)
				for (int j = 0; j < C; ++j)
					ret[i][j] = pred(ret[i][j]);
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
