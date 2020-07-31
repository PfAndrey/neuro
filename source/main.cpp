#include <iostream>
#include "NeuralNetwork.h"

#define MNIST_TEST

int main()
{
#ifdef MNIST_TEST //mnist training data

	//create 3-th layer FCNN
	NeuralLib::NeuralNetwork network({28*28,128,10}); 

	//load train and test data
	auto learn_data_raw = NeuralLib::loadMnistTrainingData(
		"../resources/mnist training data/train-labels.idx1-ubyte",
		"../resources/mnist training data/train-images.idx3-ubyte");

	auto test_data_raw = NeuralLib::loadMnistTrainingData(
		"../resources/mnist training data/t10k-labels.idx1-ubyte",
		"../resources/mnist training data/t10k-images.idx3-ubyte");

	auto [learn_data, learn_data2] = NeuralLib::splitTrainingData(learn_data_raw,10000); // get the first 10000 from 60000 only
	auto [test_data, test_data2] = NeuralLib::splitTrainingData(test_data_raw, 1000); // get the first 1000 from 5000 only

#else			//or simple digit training data
	//create 3-th layer FCNN
	NeuralLib::NeuralNetwork network({ 32 * 32,64,64,64,64,10 });

	//load train and test data
	auto training_data = NeuralLib::loadTrainingData("../resources/simple digit training data/training.data");
	auto [learn_data, test_data] = NeuralLib::splitTrainingData(training_data, training_data.first.size() - 100);

#endif

	printf("learn data loaded. data.size=%d\n", learn_data.first.size());
	printf("test data loaded. data.size=%d\n", test_data.first.size());

	network.rpropLearn(learn_data, test_data, 0.08);
	//network.backPropLearn(learn_data, test_data, 0.25);

	system("pause");
    return 0;
}

