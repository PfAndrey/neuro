#ifndef NEURALNETWORK
#define NEURALNETWORK

#include "Matrix.h"
#include <vector>
#include <chrono>
#include "ActivationFunctions.h"

namespace NeuralLib
{
	using FType = float;
	static auto hidden_activation_function = sigmoid<FType>;
	static auto hidden_activation_function_der = sigmoid_d<FType>;

	static auto output_activation_function = softMax<FType>;
	static auto output_activation_function_der = sigmoid_d<FType>;

	using MatrixList = std::vector<Matrix<FType>>;
	using TrainingData = std::pair<MatrixList, MatrixList>;

	std::tuple<TrainingData, TrainingData> splitTrainingData(const TrainingData& data, int from);

	class NeuralNetwork
	{
	public:
		NeuralNetwork(const std::vector<int>& list);
		int layers() const;
		Matrix<FType> computeOutput(const Matrix<FType>& inputs) const;
		void backPropLearn(const TrainingData& data, const TrainingData& test_data, double learning_rate);
		void rpropLearn(const TrainingData& data, const TrainingData& test_data, double learning_rate);
		float getCurrecy(const TrainingData& test_data);
	private:
		static void rpropInitMatrixs(
			MatrixList& weights, MatrixList& biases,
			MatrixList& prev_dEdW_acc, MatrixList& prev_dEdB_acc);
		static std::tuple<MatrixList, MatrixList> rpropAcummulateGradients(
			const MatrixList& weights, const MatrixList& biases,
			const TrainingData& data, int begin, int end);
		static void rpropUpdateWeightsAndBiases(
			MatrixList& weights, MatrixList& biases,
			MatrixList& weights_grad, MatrixList& biases_grad,
			MatrixList& prev_weights_grad,  MatrixList& prev_biases_grad, 
			double learning_rate,int begin,int end);
		MatrixList m_weights;
		MatrixList m_biases;
		void epochStartStat(int i);
		void epochEndStat(int i, const TrainingData& test_data);
		std::chrono::time_point<std::chrono::high_resolution_clock> m_epoch_start;
		const int MAX_EPOCH = 1000;
		float m_prev_currency;
		float m_max_currency;
		float m_eta;
		int m_increase_steps = 0;
	};

	TrainingData loadTrainingData(const std::string& file_path);
	TrainingData loadMnistTrainingData(const std::string& labels_file_path, const std::string& images_file_path);
}
#endif // ! NEURALNETWORK
