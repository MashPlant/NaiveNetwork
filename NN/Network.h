#pragma once
#include "Matrix.h"
#include <string>
#include <iostream>
#include "Function.h"

namespace nn
{
	template <int Where, typename K, K ...Args>
	constexpr K fromArgs()
	{
		//����C++14���ԣ��Ҳ�֪����C++11����ôд
		constexpr std::array<K, sizeof...(Args)> tmp({ Args... });
		return std::get<Where>(tmp);
	}
	template<typename Decimal, int ...Shape>
	class Network
	{
	private:
		template <int Pre, int ...Nxt>
		struct Layer
		{
			//w,b�����������ɱ���ģ�Ҳ����˵w�����ϲ���������
			//Nxt�ĵ�0��Ԫ���Ǳ���Ĵ�С
			const static int Cur = fromArgs<0, int, Nxt...>();
			Matrix<Decimal, Cur, Pre> w[3]; // ��ǰ/��ǰ�ݶ�/�ۼ��ݶ�
			Matrix<Decimal, Cur, 1> b[3];
			Matrix<Decimal, Cur, 1> a[2];

			Layer<Nxt...> next;
			constexpr static int num() { return Cur; }
		};
		template <int ...Nxt>
		struct Layer<0, Nxt...> //����������0��
		{
			const static int Cur = fromArgs<0, int, Nxt...>();
			Matrix<Decimal, Cur, 1> a[2]; //��ʵ����Ҫ[2]��ֻ��Ϊ��д�������
			Layer<Nxt...> next;
			constexpr static int num() { return Cur; }
		};
		template <int Pre,int Cur>
		struct Layer<Pre, Cur> //�����������һ��
		{
			Matrix<Decimal, Cur, Pre> w[3];
			Matrix<Decimal, Cur, 1> b[3];
			Matrix<Decimal, Cur, 1> a[2];
			constexpr static int num() { return Cur; }
		}; 
		Layer <0, Shape... > first;
		Sigmoid f;
		const static int InputLayer = fromArgs<0, int, Shape...>();
		const static int OutputLayer = fromArgs<sizeof...(Shape)-1, int, Shape...>();
		template <int ...Args>
		Matrix<Decimal, OutputLayer, 1> forwardProp(Layer<Args ...> &layer)
		{
			auto &now = layer.next;
			now.a[0] = f(now.b[0] + now.w[0] * layer.a[0]);
			return forwardProp(now);
		}
		template <int Pre,int Cur>
		Matrix<Decimal, OutputLayer, 1> forwardProp(Layer<Pre, Cur> &layer) { return layer.a[0]; }

		template <int ...Args>
		void backProp(Layer<Args...> &layer,const Matrix<Decimal, OutputLayer, 1> &train)
		{
			backProp(layer.next, train);
			auto &next = layer.next;
			next.b[1] = dot(f.dAtY(next.a[0]), next.a[1]); // dC/dbi = dC/dai * dai/dbi = a[1][i] * dAtY(a[0][i])
			next.w[1] = next.b[1] * layer.a[0].T(); //ע�⵽w��ά���Ǳ���xǰһ��,dC/dwij = dC/dai(����) * dai/dwij = a[1][i] * dAtY(a[0][i]) * prev.a[0][j]
			layer.a[1] = next.w[0].T() * next.b[1]; //(ǰһ���)dC/dai = sum(dC/daj(��һ���) * daj/dai) = sum(w[j][i] * a[1][j] * dAtY(a[0][j]))

			next.b[2] += next.b[1];
			next.w[2] += next.w[1];

			/*for (int j = 0; j < next.num(); ++j)//j ��next��ڵ���
			{
				for (int k = 0; k < layer.num(); ++k)//k ��next-1��ڵ���
					next.w[1][j][k] = layer.a[0][k][0] * f.dAtY(next.a[0][j][0]) *next.a[1][j][0];
				next.b[1][j][0] = f.dAtY(next.a[0][j][0]) *next.a[1][j][0];
			}
			for (int j = 0; j < layer.num(); ++j)//��ǰ���C������һ�㼤��ֵ��ƫ����
			{
				layer.a[1][j][0] = 0;
				for (int k = 0; k < next.num(); ++k) //���ﷴ������j ��next-1��ڵ�����k ��next��ڵ���
					layer.a[1][j][0] += next.w[0][k][j] * f.dAtY(next.a[0][k][0]) * next.a[1][k][0];
			}*/
		}
		template <int Pre, int Cur>
		void backProp(Layer<Pre,Cur> &layer, const Matrix<Decimal, OutputLayer, 1> &train)
		{
			layer.a[1] = (layer.a[0] - train) * 2;
		}

		template <int ...Args>
		void update(Layer<Args...> &layer,Decimal rate)
		{
			layer.w[0] -= layer.w[2] * rate;
			layer.b[0] -= layer.b[2] * rate;
			layer.w[2].fill(0), layer.b[2].fill(0);
			update(layer.next, rate);
		}
		template <int Pre, int Cur>
		void update(Layer<Pre, Cur> &layer, Decimal rate)
		{
			layer.w[0] -= layer.w[2] * rate;
			layer.b[0] -= layer.b[2] * rate;
			layer.w[2].fill(0), layer.b[2].fill(0);
		}
	public:
		Matrix<Decimal, OutputLayer, 1> forwardProp(const Matrix<Decimal, InputLayer, 1> &input)
		{
			first.a[0] = input;
			return forwardProp(first);
		}
		void backProp(const Matrix<Decimal, OutputLayer, 1> &train) { backProp(first, train); }
		void update(Decimal rate) { update(first.next, rate); }
	};
}
