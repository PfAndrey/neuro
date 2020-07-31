#include "NeuralNetwork.h"
#include <fstream>
#include <thread>
#include <future>
#include "assert.h"
#include <string>
#include <iomanip>

namespace NeuralLib
{
	template <typename T> int sgn(T val) 
	{
		static const T ZERO_TOLERANCE = 1e-15;

		if (abs(val) < ZERO_TOLERANCE)
			return T(0);

		return (T(0) < val) - (val < T(0));
	}

	NeuralNetwork::NeuralNetwork(const std::vector<int>& list):
		m_prev_currency(0),
		m_max_currency(0),
		m_eta(0.1)
	{
		for (int i=0; i < list.size() - 1 ; ++i)
			m_weights.push_back(Matrix<FType>(list[i+1], list[i]));

		for (int i = 1; i < list.size(); ++i)
			m_biases.push_back(Matrix<FType>(list[i], 1));
	}

	int NeuralNetwork::layers() const
	{
		return m_weights.size();
	}

	Matrix<FType> NeuralNetwork::computeOutput(const Matrix<FType>& inputs) const
	{
		assert(inputs.cols() == m_weights[0].rows());

		Matrix<FType> outputs = inputs;
		for (int i = 0; i < m_weights.size(); ++i)
		{
			auto activation_function = (i != m_weights.size() - 1) ? hidden_activation_function : output_activation_function;
			outputs = activation_function(outputs*m_weights[i] + m_biases[i]);
		}

		return outputs;
	}

	void NeuralNetwork::backPropLearn(const TrainingData& learm_data, const TrainingData& test_data, double learning_rate)
	{
		assert(learm_data.first.size() == learm_data.second.size());
		m_eta = learning_rate;

		const auto& learning_inputs_set = learm_data.first;
		const auto& learning_outputs_set = learm_data.second;
	
		//intialization
		const int LAYERS = layers();

		srand(0);
		for (int i = 0; i < LAYERS; ++i)
		{
			m_weights[i].fillRandom(-0.5, 0.5);
			m_biases[i].fillRandom(-0.5, 0.5);
		}
		
		MatrixList real_outputs(LAYERS), sum(LAYERS);
		MatrixList dEdB(LAYERS),dEdW(LAYERS);

		for (int epoch = 0; epoch < MAX_EPOCH; ++epoch)
		{
			epochStartStat(epoch);

			for (int i = 0; i < learm_data.first.size(); ++i)
			{
				//Forward propagation 
				for (int j = 0; j < LAYERS; ++j)
				{
					const Matrix<FType>& input = (j==0) ? learning_inputs_set[i] : real_outputs[j-1];
					auto activation_function = (j != LAYERS - 1) ? hidden_activation_function : output_activation_function;
					sum[j] = input * m_weights[j] + m_biases[j];
					real_outputs[j] = activation_function(sum[j]);
				}

				//Back propagation
				for (int l = LAYERS-1; l >= 0; --l)
				{
					const Matrix<FType>& out = (l>0)? real_outputs[l-1] : learning_inputs_set[i];
					
					//Error function
					auto error_function = (l== LAYERS -1)? real_outputs[l] - learning_outputs_set[i] : dEdB[l+1] * m_weights[l + 1].transp();
										
					//Gradient descent
					auto activation_function_der = (l == LAYERS - 1) ? output_activation_function_der : hidden_activation_function_der;
					dEdB[l] = error_function.wiseproduct(output_activation_function_der(sum[l]));
					dEdW[l] = out.transp() * dEdB[l];
					
					//Update weights and biases
					m_weights[l] -= dEdW[l] * m_eta;
					m_biases[l] -= dEdB[l] * m_eta;
				}	
			}

			epochEndStat(epoch, test_data);
		}
	}

	void NeuralNetwork::rpropInitMatrixs(
		MatrixList& weights, MatrixList& biases,
		MatrixList& prev_dEdW_acc, MatrixList& prev_dEdB_acc)
	{
		const int LAYERS = weights.size();
		for (int j = 0; j < LAYERS; ++j)
		{
			prev_dEdW_acc[j].resize(weights[j].size());
			prev_dEdB_acc[j].resize(biases[j].size());
			weights[j].fillRandom(-0.5, 0.5);
			biases[j].fillRandom(-0.5, 0.5);
		}
	}

	std::tuple<MatrixList, MatrixList> NeuralNetwork::rpropAcummulateGradients(const MatrixList& weights, const MatrixList& biases, const TrainingData& training_data, int begin, int end)
	{
		const auto& learning_inputs_set = training_data.first;
		const auto& learning_outputs_set = training_data.second;
		const int LAYERS = weights.size();
		MatrixList real_outputs(LAYERS);
		MatrixList sum(LAYERS);

		MatrixList dEdB_acc(LAYERS), dEdW_acc(LAYERS);

		for (int j = 0; j < LAYERS; ++j)
		{
			dEdW_acc[j].resize(weights[j].size());
			dEdB_acc[j].resize(biases[j].size());
		}
	 
		// Calculate accumulate weights and biases gradients
		for (int i = begin; i < end; ++i)
		{
			//Forward propagation
			for (int j = 0; j < LAYERS; ++j)
			{
				const Matrix<FType>& input = (j==0) ? learning_inputs_set[i] : real_outputs[j - 1];
				auto activation_function = (j!=LAYERS-1) ? hidden_activation_function : output_activation_function;
				sum[j] = input * weights[j] + biases[j];
				real_outputs[j] = activation_function(sum[j]);
			}

			//Backward propagation
			Matrix<FType> dEdB, dEdW;
			for (int l = LAYERS - 1; l >= 0; --l)
			{
				const Matrix<FType>& out = (l==0)?learning_inputs_set[i]:real_outputs[l-1];

				//Error function
				auto error_function = (l==LAYERS-1)?(real_outputs[l]-learning_outputs_set[i]):dEdB*weights[l+1].transp();

				//Gradient descent
				auto& deriviate = (l==LAYERS-1)?output_activation_function_der:hidden_activation_function_der;

				dEdB = error_function.wiseproduct(deriviate(sum[l]));
				dEdW  = out.transp() * dEdB;

				dEdW_acc[l] += dEdW;
				dEdB_acc[l] += dEdB;
			}
		}

		return {dEdW_acc, dEdB_acc};
	}

	void NeuralNetwork::rpropUpdateWeightsAndBiases(MatrixList& weights, MatrixList& biases, MatrixList& dEdW_acc, MatrixList& dEdB_acc, MatrixList& prev_dEdW_acc, MatrixList& prev_dEdB_acc, double eta, int begin,int end)
	{
		static const double POSITIVE_ETA = 1.2;
		static const double NEGATIVE_ETA = 0.5;
		static const double DELTA_MIN = 0.1;
		static const double DELTA_MAX = 50.;

		const int LAYERS = weights.size();
		MatrixList delta_w(LAYERS), delta_b(LAYERS);

		for (int j = begin; j < end; ++j)
		{
			delta_w[j].resize(weights[j].size(), DELTA_MIN);
			delta_b[j].resize(biases[j].size(), DELTA_MIN);
		}

		// Update weights and biases
		for (int l = begin; l < end; ++l)
		{
			auto WIDTH = biases[l].cols();
			auto HEIGHT = weights[l].rows();

			auto sign_w = dEdW_acc[l].wiseproduct(prev_dEdW_acc[l]);
			auto sign_b = dEdB_acc[l].wiseproduct(prev_dEdB_acc[l]);

			for (int x = 0; x < WIDTH; ++x)
			{
				//Update weights
				for (int y = 0; y < HEIGHT; ++y)
				{
					const auto sign_g = sgn(dEdW_acc[l](x, y));
					const auto t = sign_w(x, y);
					auto& delta = delta_w[l](x, y);

					if (t > 0)
					{
						delta = std::min(delta*POSITIVE_ETA, DELTA_MAX);
						weights[l](x, y) -= sign_g * delta*eta;
					}
					else if (t < 0)
					{
						//weights[l](x, y) -= delta;
						delta = std::max(delta*NEGATIVE_ETA, DELTA_MIN);
						dEdW_acc[l](x, y) = 0;
					}
					else
					{
						weights[l](x, y) -= sign_g * delta*eta;
					}
				}
	
				//Update biases
				const auto sign_g = sgn(dEdB_acc[l](x, 0));
				const auto t = sign_b(x, 0);
				auto& delta = delta_b[l](x, 0);

				if (t > 0)
				{
					delta = std::min(delta*POSITIVE_ETA, DELTA_MAX);
					biases[l](x, 0) -= sign_g * delta*eta;
				}
				else if (t < 0)
				{
					//biases[l](x, y) -= delta;
					delta = std::max(delta*NEGATIVE_ETA, DELTA_MIN);
					dEdB_acc[l](x, 0) = 0;
				}
				else
				{
					biases[l](x, 0) -= sign_g * delta*eta;
				}
			}
			prev_dEdB_acc[l] = dEdB_acc[l];
			prev_dEdW_acc[l] = dEdW_acc[l];
		}
	}

	void NeuralNetwork::rpropLearn(const TrainingData& training_data, const TrainingData& test_data, double eta)
	{
		assert(training_data.first.size() == training_data.second.size());
		m_eta = eta;

		const auto& learning_inputs_set = training_data.first;
		const auto& learning_outputs_set = training_data.second;

		MatrixList prev_dEdW_acc(layers()), prev_dEdB_acc(layers());
		rpropInitMatrixs(m_weights, m_biases, prev_dEdW_acc, prev_dEdB_acc);

		static const int CORES = std::thread::hardware_concurrency();

		std::cout << "Rprop started - " << CORES << " threads, eta = " << eta << "." << std::endl;

		const auto start_range = 0;
		const auto end_range = learning_inputs_set.size();

		for (int epoch = 0; epoch < MAX_EPOCH; ++epoch)
		{
			epochStartStat(epoch);

			if (CORES == 1)
			{
				auto[dEdW_acc, dEdB_acc] = rpropAcummulateGradients(m_weights, m_biases, training_data, start_range, end_range);
				rpropUpdateWeightsAndBiases(m_weights, m_biases, dEdW_acc, dEdB_acc, prev_dEdW_acc, prev_dEdB_acc, m_eta, 0, 2);
			}
			else
			{
				std::vector<std::future<std::tuple<MatrixList, MatrixList>>> futures(CORES);
				const int step = (end_range - start_range) / CORES;

				for (int i = 0; i < CORES; ++i)
				{
					int start = start_range + i * step;
					int end = start + step;

					futures[i] = std::async(rpropAcummulateGradients, std::ref(m_weights), std::ref(m_biases), std::ref(training_data), start, end);
				}

				MatrixList dEdW_acc(layers()), dEdB_acc(layers());
				for (auto& future : futures)
				{
					auto[dEdW_acc1, dEdB_acc1] = future.get();
					dEdW_acc += dEdW_acc1;
					dEdB_acc += dEdB_acc1;
				}

				std::vector<std::future<void>> ufutures(CORES);

				const int up_step = std::max(layers() / CORES, 1);
				for (int i = 0; i < CORES; ++i)
				{
					auto start = up_step * i;
					auto end = start + up_step;

					if (i == CORES - 1)
						end = layers();

					ufutures[i] = std::async(rpropUpdateWeightsAndBiases, std::ref(m_weights), std::ref(m_biases), std::ref(dEdW_acc), std::ref(dEdB_acc), std::ref(prev_dEdW_acc), std::ref(prev_dEdB_acc), m_eta, start, end);

					if (end >= layers())
						break;
				}

				for (auto& future: ufutures)
					if (future.valid())
						future.get();
			}
			epochEndStat(epoch, test_data);
		}
	}

	void NeuralNetwork::epochStartStat(int epoch)
	{
		m_epoch_start = std::chrono::high_resolution_clock::now();
	}

	void NeuralNetwork::epochEndStat(int epoch, const TrainingData& test_data)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_epoch_start).count();
		float k = getCurrecy(test_data);
		m_max_currency = std::max(m_max_currency,k);
		std::string add_info;

		
		std::cout << std::fixed << std::setprecision(1);

		std::cout << "iteration: " << epoch 
			<< " currency: " << k << "% "
			<< "(" << m_max_currency << "% max) "
			<< ms << " ms.";

		
		auto new_eta = m_eta;
		if (m_prev_currency > k)
		{
			new_eta *= 0.5;
			m_increase_steps = 0;
		}
		else
		{
			if (++m_increase_steps > 3)
			{
				new_eta *= 1.2;
				m_increase_steps = 0;
			}
		}


		if (new_eta != m_eta)
		{
			const bool inr = new_eta > m_eta;
			std::cout << std::fixed << std::setw(4) << std::setprecision(4)
				<< " - learning rate: " << (inr?m_eta:new_eta) << (inr?" ==> ":" <== ") << (inr?new_eta:m_eta)
				<< std::defaultfloat;
			m_eta = new_eta;
		}

		std::cout << std::endl;
		m_prev_currency = k;
	}

	float NeuralNetwork::getCurrecy(const TrainingData& test_data)
	{
		int k = 0;
		const auto& test_inputs_set = test_data.first;
		const auto& test_outputs_set = test_data.second;
		const int size = test_outputs_set.size();

		for (int i = 0; i < size; ++i)
		{
			auto output = computeOutput(test_inputs_set[i]);
			int pred_result = output.maxVectorIndex();
			int real_result = test_outputs_set[i].maxVectorIndex();
			k += pred_result == real_result;
		}
		return 100.f * k / size;
	}

	//-----------------------------------------------------------------------------------------------------------

	TrainingData loadTrainingData(const std::string& file_path)
	{
		std::fstream file(file_path.c_str());

		TrainingData training_data;

		assert(file.is_open());

		while (!file.eof())
		{
			std::vector<FType> input;

			for (int i = 0; i < 32; ++i)
			{
				char buf[255] = { 0 };
				file.getline(buf, 255);

				int line[32] = { 0 };
				for (int i = 0; buf[i]; ++i)
					if (buf[i] == '1')
						line[i] = 1;
				input.insert(input.end(), &line[0], &line[32]);
			}

			training_data.first.push_back(Matrix<FType>::fromVector(input));

			char nums[2] = { 0 };
			file.getline(nums, 2);
			int num = atoi(nums);

			std::vector<FType> output(10, 0);
			output[num] = 1;
			training_data.second.push_back(Matrix<FType>::fromVector(output));
		}

		file.close();
		return training_data;
	}

	//-------------------------------------------------------------------------------------------------
	
	std::tuple<TrainingData, TrainingData> splitTrainingData(const TrainingData& data, int from)
	{
		TrainingData one, two;

		one.first = MatrixList(&data.first[0], &data.first[from]);
		one.second = MatrixList(&data.second[0], &data.second[from]);
		two.first = MatrixList(&data.first[from], &data.first[data.first.size()]);
		two.second = MatrixList(&data.second[from], &data.second[data.second.size()]);

		return { one,two };
	}
	
	void SwitchEndian(char* data)
	{
		std::swap(data[0], data[3]);
		std::swap(data[1], data[2]);
	}

	TrainingData loadMnistTrainingData(const std::string& labels_file_path, const std::string& image_file_path)
	{
		TrainingData training_data;

		//Load labels
		std::ifstream lab_file(labels_file_path.c_str(), std::ios_base::binary);
		assert(lab_file.is_open());
		uint32_t  magic = 0, items = 0;
		lab_file.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
		lab_file.read(reinterpret_cast<char*>(&items), sizeof(int32_t));
		SwitchEndian(reinterpret_cast<char*>(&magic));
		SwitchEndian(reinterpret_cast<char*>(&items));

		uint8_t lab;
		for (int i = 0; i < items; ++i)
		{
			lab_file.read(reinterpret_cast<char*>(&lab), sizeof(uint8_t));
			std::vector<FType> output(10, 0);
			output[lab] = 1;
			training_data.second.push_back(Matrix<FType>::fromVector(output));
		}
		lab_file.close();
		

		//Load images
		std::ifstream img_file(image_file_path.c_str(), std::ios_base::binary);
		assert(img_file.is_open());
		uint32_t images = 0, cols = 0, rows = 0;
		img_file.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
		img_file.read(reinterpret_cast<char*>(&images), sizeof(int32_t));
		img_file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
		img_file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
		SwitchEndian((char*)&magic);
		SwitchEndian((char*)&images);
		SwitchEndian((char*)&cols);
		SwitchEndian((char*)&rows);

		const int size = cols*rows;
		char* buff = new char[size];
		for (int k = 0; k < images; ++k)
		{
			img_file.read(buff, size);

			FType* fbuff = new FType[size];
			for (int i = 0; i < size; ++i)
				fbuff[i] =   buff[i] > 10;
			training_data.first.push_back(Matrix<FType>(fbuff, size,1));
		}
		delete buff;
		img_file.close();

		return training_data;
	}
}


