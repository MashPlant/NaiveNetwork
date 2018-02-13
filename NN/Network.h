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
		//这是C++14特性，在C++11下可以利用下面的IF配合递归来写，有点丑
		constexpr std::array<K, sizeof...(Args)> tmp({ Args... });
		return std::get<Where>(tmp);
	}

	template <bool,typename OnTrue,typename OnFalse>
	struct IF { typedef OnFalse type; };
	template<typename OnTrue, typename OnFalse>
	struct IF<true, OnTrue, OnFalse> { typedef OnTrue type; };

	template<typename Decimal, int ...Shape>
	class Network
	{
	private:
		template <int Pre, int ...Nxt>
		struct Layer
		{
			//w,b都是用来生成本层的，也就是说w是由上层连出来的
			//Nxt的第0个元素是本层的大小
			const static int Cur = fromArgs<0, int, Nxt...>();
			Matrix<Decimal, Cur, Pre> w[3]; // 当前/当前梯度/累计梯度
			Matrix<Decimal, Cur, 1> b[3];
			Matrix<Decimal, Cur, 1> a[2];
			Layer<Nxt...> next;
			Activation f;
			Layer()
			{
				w[0] = Matrix<Decimal, Cur, Pre>(std::sqrt(6 / (InputLayer + OutputLayer)));
				b[0] = Matrix<Decimal, Cur, 1>(std::sqrt(6 / (InputLayer + OutputLayer)));
			}
		};
		template <int ...Nxt>
		struct Layer<0, Nxt...> //特例化，第0层
		{
			const static int Cur = fromArgs<0, int, Nxt...>();
			Matrix<Decimal, Cur, 1> a[2]; //其实不需要[2]，只是为了写起来简便
			Layer<Nxt...> next;
		};
		template <int Pre,int Cur>
		struct Layer<Pre, Cur> //特例化，最后一层
		{
			Matrix<Decimal, Cur, Pre> w[3];
			Matrix<Decimal, Cur, 1> b[3];
			Matrix<Decimal, Cur, 1> a[2];
			Activation f;
			Layer()
			{
				w[0] = Matrix<Decimal, Cur, Pre>(std::sqrt(6 / (InputLayer + OutputLayer)));
				b[0] = Matrix<Decimal, Cur, 1>(std::sqrt(6 / (InputLayer + OutputLayer)));
			}
		}; 
		Layer <0, Shape... > first;
		Activation output = Activation::none;
		Loss loss;
		const static int InputLayer = fromArgs<0, int, Shape...>();
		const static int OutputLayer = fromArgs<sizeof...(Shape)-1, int, Shape...>();
		template <int ...Args>
		Matrix<Decimal, OutputLayer, 1> forwardProp(Layer<Args ...> &layer)
		{
			auto &now = layer.next;
			now.a[0] = now.f(now.b[0] + now.w[0] * layer.a[0]);
			return forwardProp(now);
		}
		template <int Pre,int Cur>
		Matrix<Decimal, OutputLayer, 1> forwardProp(Layer<Pre, Cur> &layer) { return layer.a[0]; }

		template <int ...Args>
		void backProp(Layer<Args...> &layer,const Matrix<Decimal, OutputLayer, 1> &train)
		{
			backProp(layer.next, train);
			auto &next = layer.next;
			next.b[1] = dot(next.f.dAtY(next.a[0]), next.a[1]); // dC/dbi = dC/dai * dai/dbi = a[1][i] * dAtY(a[0][i])
			next.w[1] = next.b[1] * layer.a[0].T(); //注意到w的维度是本层x前一层,dC/dwij = dC/dai(本层) * dai/dwij = a[1][i] * dAtY(a[0][i]) * prev.a[0][j]
			layer.a[1] = next.w[0].T() * next.b[1]; //(前一层的)dC/dai = sum(dC/daj(后一层的) * daj/dai) = sum(w[j][i] * a[1][j] * dAtY(a[0][j]))

			next.b[2] += next.b[1];
			next.w[2] += next.w[1];
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

		template <int Where,int ...Args>
		void setActivation(Layer<Args...> &layer,int type,std::false_type)
		{
			setActivation<Where - 1>(layer.next, type, typename IF<Where == 1, std::true_type, std::false_type>::type());
			//一个更简单的写法是std::integral_constant<bool,Where==0>(),但是不那么显然
		}
		template <int Where,int ...Args>
		void setActivation(Layer<Args...> &layer, int type, std::true_type) { layer.f = type; }
	public:
		Matrix<Decimal, OutputLayer, 1> forwardProp(const Matrix<Decimal, InputLayer, 1> &input)
		{
			first.a[0] = input;
			return forwardProp(first);
		}
		void backProp(const Matrix<Decimal, OutputLayer, 1> &train) { backProp(first, train); }
		void update(Decimal rate) { update(first.next, rate); }
		void setOutputActivation(int type) { output = type; }
		void setLoss(int type) { loss = type; }
		template <int Where>
		void setActivation(int type)
		{
			static_assert(Where >= 1 && Where <= sizeof...(Shape), "only layer within [1,sizeof...(Shape)] has activation");
			if (Where == sizeof...(Shape))
				setOutputActivation(type);
			else
				setActivation<Where>(first, type, std::false_type());
		}
	};
}
