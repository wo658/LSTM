#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Utility function to initialize weights with small random values
double initWeight() {
	std::uniform_real_distribution<> dis(-0.1, 0.1);
	return dis(gen);
}

// Hyperbolic Tangent (tanh) activation function
double tanhActivation(double x) {
	return tanh(x);
}

// Derivative of tanh activation function
double tanh_derivative(double x) {
	double tanh_value = tanh(x);
	return 1.0 - pow(tanh_value, 2); 
}

class RNN {
private:
	int inputSize;
	int hiddenSize;
	int outputSize;
	std::vector<double> hiddenState;
	std::vector<std::vector<double>> hiddenStates;
	std::vector<std::vector<double>> Wih;
	std::vector<std::vector<double>> Whh;
	std::vector<std::vector<double>> Who;
	double learningRate;

public:
	RNN(int i, int h, int o, double lr) : inputSize(i), hiddenSize(h), outputSize(o), learningRate(lr) {
	
		hiddenState.resize(hiddenSize, 0.0);

		Wih.resize(inputSize, std::vector<double>(hiddenSize));
		Whh.resize(hiddenSize, std::vector<double>(hiddenSize));
		Who.resize(hiddenSize, std::vector<double>(outputSize));

		for (int i = 0; i < inputSize; i++)
			for (int h = 0; h < hiddenSize; h++)
				Wih[i][h] = initWeight();

		for (int h = 0; h < hiddenSize; h++)
			for (int hh = 0; hh < hiddenSize; hh++)
				Whh[h][hh] = initWeight();

		for (int h = 0; h < hiddenSize; h++)
			for (int o = 0; o < outputSize; ++o)
				Who[h][o] = initWeight();
	}

	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& inputs ) {
		std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(outputSize, 0.0));
		hiddenStates.clear();

		for (size_t t = 0; t < inputs.size(); ++t) {
			std::vector<double>& input = inputs[t];
			std::vector<double> newHidden(hiddenSize, 0.0);

			for (int h = 0; h < hiddenSize; ++h) {
				for (int i = 0; i < inputSize; ++i)
					newHidden[h] += input[i] * Wih[i][h];
				for (int hh = 0; hh < hiddenSize; ++hh)
					newHidden[h] += hiddenState[hh] * Whh[hh][h];
				newHidden[h] = tanhActivation(newHidden[h]);
			}

			hiddenStates.push_back(newHidden); 
			hiddenState = newHidden;

			for (int o = 0; o < outputSize; ++o) {
				for (int h = 0; h < hiddenSize; ++h)
					outputs[t][o] += hiddenState[h] * Who[h][o];
			}
		}

		return outputs;
	}

	void backward(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, std::vector<std::vector<double>>& outputs, double clipValue) {
		std::vector<std::vector<double>> dWhoSum(hiddenSize, std::vector<double>(outputSize, 0.0));
		std::vector<std::vector<double>> dWihSum(inputSize, std::vector<double>(hiddenSize, 0.0));
		std::vector<std::vector<double>> dWhhSum(hiddenSize, std::vector<double>(hiddenSize, 0.0));
		std::vector<double> nextHiddenDelta(hiddenSize, 0.0);

		for (int t = inputs.size() - 1; t >= 0; --t) {
			std::vector<double> outputDelta(outputSize, 0.0);
			std::vector<double> hiddenDelta(hiddenSize, 0.0);

			for (int o = 0; o < outputSize; ++o) {
				double error = (-targets[t][o] + outputs[t][o]) ;
				outputDelta[o] = error * tanh_derivative(hiddenStates[t][o]);
			}

			for (int h = 0; h < hiddenSize; ++h) {
				double error = 0.0;
				for (int o = 0; o < outputSize; ++o)
					error += outputDelta[o] * Who[h][o];
				for (int hh = 0; hh < hiddenSize; ++hh)
					error += nextHiddenDelta[hh] * Whh[h][hh];
				hiddenDelta[h] = error * tanh_derivative(hiddenStates[t][h]);
			}

			nextHiddenDelta = hiddenDelta;
			for (int h = 0; h < hiddenSize; ++h) {
				for (int o = 0; o < outputSize; ++o) {
					dWhoSum[h][o] += hiddenStates[t][h] * outputDelta[o];
				}
			}

			std::vector<double>& input = inputs[t];
			for (int i = 0; i < inputSize; ++i) {
				for (int h = 0; h < hiddenSize; ++h) {
					dWihSum[i][h] += input[i] * hiddenDelta[h];
				}
			}

			for (int h = 0; h < hiddenSize; ++h) {
				for (int hh = 0; hh < hiddenSize; ++hh) {
					dWhhSum[h][hh] += hiddenStates[t][hh] * hiddenDelta[h];
				}
			}
		}
		
		// Gradient Clipping
		for (int h = 0; h < hiddenSize; ++h) {
			for (int o = 0; o < outputSize; ++o) {
				dWhoSum[h][o] = std::max(std::min(dWhoSum[h][o], clipValue), -clipValue);
			}
		}
		for (int i = 0; i < inputSize; ++i) {
			for (int h = 0; h < hiddenSize; ++h) {
				dWihSum[i][h] = std::max(std::min(dWihSum[i][h], clipValue), -clipValue);
			}
		}
		for (int h = 0; h < hiddenSize; ++h) {
			for (int hh = 0; hh < hiddenSize; ++hh) {
				dWhhSum[h][hh] = std::max(std::min(dWhhSum[h][hh], clipValue), -clipValue);
			}
		}

		// Update 
		for (int h = 0; h < hiddenSize; ++h) {
			for (int o = 0; o < outputSize; ++o) {
				Who[h][o] -= learningRate * dWhoSum[h][o];
			}
		}
		for (int i = 0; i < inputSize; ++i) {
			for (int h = 0; h < hiddenSize; ++h) {
				Wih[i][h] -= learningRate * dWihSum[i][h];
			}
		}
		for (int h = 0; h < hiddenSize; ++h) {
			for (int hh = 0; hh < hiddenSize; ++hh) {
				Whh[h][hh] -= learningRate * dWhhSum[h][hh];
			}
		}
	}


};

