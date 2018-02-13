#include "stdafx.h"
#include "Matrix.h"
#include "Network.h"
using namespace std;
using namespace nn;


int main()
{
	Network<float, 2, 2 ,2> net;
	for (int k = 0; k < 160000; ++k)
	{
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j)
			{
				Matrix<float, 2, 1> input = { { i * 1.0f },{ j* 1.0f } };
				Matrix<float, 2, 1> output;
				output[i ^ j][0] = 1.0f;
				if (k % 5000 == 0)
					cout << net.forwardProp(input)[1][0]<<' ';
				else
					net.forwardProp(input);
				net.backProp(output);
			}
		if (k % 500 == 0)
			net.update(0.01f);
		if (k % 5000 == 0)
			cout << endl;
	}
		
	
	getchar();
}

