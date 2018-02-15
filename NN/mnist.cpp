#include "stdafx.h"
#include "Matrix.h"
#include "Network.h"
using namespace std;
using namespace nn;

int main()
{
	freopen("mnist.txt", "r", stdin);
	const int batch = 100;
	const int N = 20000;
	Network<float, 784, 20, 10> net;
	for (int j = 0;j < N/batch;++j)
	{
		int correct = 0;
		for (int k = 0; k < batch; ++k)
		{
			Matrix<float, 784, 1> data;
			Matrix<float, 10, 1> ans;
			for (int i = 0; i < 784; ++i)
				cin >> data[i][0];
			for (int i = 0; i < 10; ++i)
				cin >> ans[i][0];
			correct += argmax(net.forwardProp(data)) == argmax(ans);
			net.backProp(ans);
		}
		net.update(0.01f);
		cout << 1.0 * correct / batch << endl;
	}
	freopen("con", "r", stdin);
	getchar();
	return 0;
}