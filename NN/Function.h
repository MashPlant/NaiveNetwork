#pragma once
#include <cmath>
#include <stdexcept>
#include "Matrix.h"
#include <functional>

namespace nn
{
	struct Sigmoid
	{
		//��������������Ǿ�̬��������C#ǡ���෴(������
		template <typename Decimal, int R, int C>
		Matrix<Decimal, R, C> operator()(const Matrix<Decimal, R, C> &y) const
		{
			return y.map([](Decimal d) {return 1 / (1 + std::exp(-d)); });
		}

		template <typename Decimal, int R, int C>
		static Matrix<Decimal, R, C> dAtY(const Matrix<Decimal, R, C> &y)
		{
			return y.map([](Decimal d) {return d - d * d; });
		}
	};

	struct ReLu
	{
		template <typename Decimal, int R, int C>
		Matrix<Decimal, R, C> operator()(const Matrix<Decimal, R, C> &y) const
		{
			return y.map([](Decimal d) {return d >= 0 ? d : 0; });
		}

		template <typename Decimal, int R, int C>
		static Matrix<Decimal, R, C> dAtY(const Matrix<Decimal, R, C> &y)
		{
			return y.map([](Decimal d) {return d >= 0; });
		}
	};

	struct Tanh
	{
		template <typename Decimal, int R, int C>
		Matrix<Decimal, R, C> operator()(const Matrix<Decimal, R, C> &y) const
		{
			//return y.map(std::tanh(Decimal)); 
			return y.map([](Decimal d) {return std::tanh(d); });
		}

		template <typename Decimal, int R, int C>
		static Matrix<Decimal, R, C> dAtY(const Matrix<Decimal, R, C> &y)
		{
			return y.map([](Decimal d) {return 1 - d*d; });
		}
	};

	struct SoftMax
	{
		//softmaxֻ��������ļ����
		//���ĵ�������˵ĵ������岻��ȫ��ͬ����Ҫֱ�ӵõ�������Ϣ
		//��softmax��:ֱ����Ԥ����-�ڴ����
		template <typename Decimal, int R>
		Matrix<Decimal, R, 1> operator()(const Matrix<Decimal, R, 1> &m) const
		{
			auto ret = m;
			Decimal s = 0;
			for (int i = 0; i < R; ++i)
				s += (ret[i][0] = std::exp(ret[i][0]));
			for (int i = 0; i < R; ++i)
				ret[i][0] /= s;
			return ret;
		}
	};
	struct Activation
	{
		enum Type { none, sigmoid, relu, softmax, tanh };
		int type = sigmoid;
		Activation(int type = sigmoid) :type(type) {}
		template <typename Decimal, int R>
		Matrix<Decimal, R, 1> operator()(const Matrix<Decimal, R, 1> &m) const
		{
			//ģ�庯���������麯���������ֹ���һ���麯����
			//��Ȼ�ˣ���ʹģ�庯���������麯����Ҳ�����õģ���������
			switch (type)
			{
			case sigmoid:
				return Sigmoid()(m);
			case relu:
				return ReLu()(m);
			case softmax:
				return SoftMax()(m);
			case none:
				return m;
			case tanh:
				return Tanh()(m);
			default:
				throw std::invalid_argument("No Such Type");
			}
		}
		template <typename Decimal, int R>
		Matrix<Decimal, R, 1> dAtY(const Matrix<Decimal, R, 1> &m) const
		{
			//SoftMax������Ϊ���ز�ļ����(�ҵ����)����������û����
			switch (type)
			{
			case sigmoid:
				return Sigmoid::dAtY(m);
			case relu:
				return ReLu::dAtY(m);
			case none: 
				return m.map([](Decimal) {return 1; });
			case tanh:
				return Tanh::dAtY(m);
			default:
				throw std::invalid_argument("no such activation type");
			}
		}
	};

	struct Loss
	{
		enum Type { mean_square, cross_entropy };
		int type = mean_square;
		Loss(int type = mean_square) :type(type) {}
		template <typename Decimal, int R>
		Decimal loss(const Matrix<Decimal, R, 1> &predict, const Matrix<Decimal, R, 1> &train) const
		{
			switch (type)
			{
			case mean_square:
				return (predict - train).reduce([](Decimal l, Decimal r) {return l + r * r; });
			case cross_entropy:
				return -dot(train, predict.map(std::log)).reduce(std::plus<>()); //log0�����Լ�����
			default:
				throw std::invalid_argument("no such loss type");
			}
		}
		template <typename Decimal, int R>
		Matrix<Decimal, R, 1> grad(const Matrix<Decimal, R, 1> &predict, const Matrix<Decimal, R, 1> &train,Activation ac) const
		{
			if (ac.type == Activation::softmax)
			{
				if (type != cross_entropy)
					throw std::invalid_argument("softmax must work with cross_entropy"); //�����Ҹ��˵�һ���ǳ�ļ���
				return predict - train; //����mean_square����������û�к����һ��Activation::dAtY(predict)
			}
			//predict�Ǿ���������ټ�һ�㼤����õ���(Ҳ���Բ���)
			Matrix<Decimal, R, 1> ret;
			switch (type)
			{
			case mean_square:
				ret = (predict - train) * 2;
				break;
			case cross_entropy:
				ret = -dot(train, predict.map([](Decimal d) {return 1 / d; }));
				break;
			default:
				throw std::invalid_argument("no such loss type");
			}
			return dot(ret, ac.dAtY(predict));
		}
	};
	
}
