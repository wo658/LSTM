#pragma once

#include <iostream>
#include <vector>
#include <cmath> // for exp function


std::random_device rd;
std::mt19937 gen(rd());

double randomWeight() {
	std::uniform_real_distribution<> dis(-0.1, 0.1);
	return dis(gen);
}

// Sigmoid function definition
double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));	
}
double sigmoid_derivative(double x) {
	double sigmoid_value = sigmoid(x);
	return sigmoid_value * (1.0 - sigmoid_value);
}

// Hyperbolic Tangent
double tanhActivation(double x) {
	return tanh(x);
}

double tanh_derivative(double x) {
	double tanh_value = tanh(x);
	return 1.0 - pow(tanh_value, 2);
}








class LSTM {
	
private:
	int input_size;	 // 조업편차의 경우 ~ 53
	int output_size; // 조업편차의 경우 1
	int gate_size; // cell size 는 hidden size 와 동일

	std::vector<std::vector<double>> whf;   //  forget gate
	std::vector<std::vector<double>> wxf;	  
	std::vector<std::vector<double>> whi;	//  input gate
	std::vector<std::vector<double>> wxi;
	std::vector<std::vector<double>> whc;	//  cell candidate gate
	std::vector<std::vector<double>> wxc;
	std::vector<std::vector<double>> who;	//  output gate
	std::vector<std::vector<double>> wxo;
	std::vector<std::vector<double>> who2;
	//gate 의 출력벡터의 차원은 ? LSTM 에서 모든 Gate 의 출력 차원은 Layer 의 유닛수로 동일하게 설정된다.

	double bf = 1;
	double bi = 1;
	double bc = 1;
	double bo = 1;
	double learningRate;

	// TimeLine 단위로 저장되는 변수들
	std::vector<std::vector<double>> outputs;
	std::vector<double> hiddenstate;		
	std::vector<std::vector<double>> hiddenstates; 
	std::vector<double> cellstate;
	std::vector<std::vector<double>> cellstates;
	std::vector<double> forgetgate;
	std::vector<std::vector<double>> forgetgates;
	std::vector<double> inputgate;
	std::vector<std::vector<double>> inputgates;
	std::vector<double> cellgate;
	std::vector<std::vector<double>> cellgates;
	std::vector<double> outgate;
	std::vector<std::vector<double>> outgates;

public:
	LSTM(int i, int h, int o, double lr) : input_size(i), gate_size(h), output_size(o), learningRate(lr)
	{

		whf.resize(h, std::vector<double>(gate_size));
		wxf.resize(i, std::vector<double>(gate_size));
		whi.resize(h, std::vector<double>(gate_size));
		wxi.resize(i, std::vector<double>(gate_size));
		whc.resize(h, std::vector<double>(gate_size));
		wxc.resize(i, std::vector<double>(gate_size));
		who.resize(h, std::vector<double>(gate_size));
		wxo.resize(i, std::vector<double>(gate_size));
		who2.resize(h, std::vector<double>(gate_size));
		
		for (int i = 0; i < input_size; i++)
			for (int h = 0; h < gate_size; h++)
			{
				wxf[i][h] = randomWeight();
				wxi[i][h] = randomWeight();
				wxc[i][h] = randomWeight();
				wxo[i][h] = randomWeight();
			}
		for (int h=0;h<gate_size;h++)
			for (int hh = 0; hh < gate_size; hh++) {
				whf[h][hh] = randomWeight();
				whi[h][hh] = randomWeight();
				whc[h][hh] = randomWeight();
				who[h][hh] = randomWeight();
			}
		for (int h = 0; h < gate_size; h++)
			for (int o = 0; o < gate_size; o++)
				who2[h][o] = randomWeight();


	}
	// weight 초기화 + 값 초기화
	std::vector<std::vector<double>> feed(std::vector<std::vector<double>>& inputs) {
		std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(output_size, 0.0));

		// 행 만큼이 TimeLine

		for (int t = 0; t < inputs.size(); t++) {


			// 이전 상태정보가 필요한 변수들
			if (t != 0) {
				hiddenstate = hiddenstates[t - 1];
				cellstate = cellstates[t - 1];
			}
			else {
				hiddenstate = std::vector<double>(gate_size, 0);
				cellstate = std::vector<double>(gate_size, 0);

			}
			
			// forget gate
			forgetgate = gatefeed(inputs[t], wxf, whf);
			for(int i=0;i<gate_size;i++)
				forgetgate[i] = sigmoid(forgetgate[i]);
			forgetgates.push_back(forgetgate);


			// input gate
			inputgate = gatefeed(inputs[t], wxi, whi);
			for (int i = 0; i < gate_size; i++)
				inputgate[i] = sigmoid(inputgate[i]);
			inputgates.push_back(inputgate);


			// cell gate
			cellgate = gatefeed(inputs[t], wxc, whc);
			for (int i = 0; i < gate_size; i++)
				cellgate[i] = tanh(cellgate[i]);
			cellgates.push_back(cellgate);

			// out gate
			outgate = gatefeed(inputs[t], wxo, who);
			for (int i = 0; i < gate_size; i++)
				outgate[i] = sigmoid(outgate[i]);
			outgates.push_back(outgate);

			// 1 . forget gate X cellstates[t-1]    -> 원소별 곱셈 .
			for (int i = 0; i < gate_size; i++)
				cellstate[i] = forgetgate[i] * cellstate[i];

			// 초기 cellstate 는 이전 노드의 값  
			// 2 .  result(1) + ( input gate ) * ( cellgate)        = Cellstate[t]
			for (int i = 0; i < gate_size; i++) {
				cellstate[i] = cellstate[i] + (inputgate[i]) * cellgate[i];
			}
			// 여기까지 했으면 cellstate 는 t시점의 cellstate가 된다. 
			cellstates.push_back(cellstate);
			// 이제 hiddenstate 와 output을 구할 차례

			// 3 .  outgate * tanh(cellstate[t]) = hiddenstate[t]

			for (int i = 0; i < gate_size; i++) {
				hiddenstate[i] = outgate[i] * tanh(cellstate[i]);
			}

			hiddenstates.push_back(hiddenstate);


			// 4 . hiddenstate * node = output

			for (int i=0;i<gate_size;i++)
				for (int j = 0; j < output_size; j++) {
					outputs[t][j] += who2[i][j] * hiddenstate[i];
				}



		}


		return outputs;

	}

	void back(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, std::vector<std::vector<double>>& outputs) {
		// time 에 따른 weight 가중치 합
		std::vector<std::vector<double>> dWhfSum(gate_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWxfSum(input_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWhiSum(gate_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWxiSum(input_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWhcSum(gate_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWxcSum(input_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWhoSum(gate_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWxoSum(input_size, std::vector<double>(gate_size, 0.0));
		std::vector<std::vector<double>> dWho2Sum(gate_size, std::vector<double>(gate_size, 0.0));


		for (int t = inputs.size() - 1; t >= 0; --t) {
			std::vector<double> outputDelta(gate_size, 0.0);
			static std::vector<double> prevCellDelta(gate_size, 0.0);
			std::vector<double> cellDelta(gate_size, 0.0);
			std::vector<double> inputDelta(gate_size, 0.0); 
			std::vector<double> forgetDelta(gate_size, 0.0);
			// 각 가중합들의 그라디언트 계산

			// 0 . Error 계산
			double error = 0;
			for (int o = 0; o < output_size; ++o) 
				error += (-targets[t][o] + outputs[t][o]);
			// 1 . 출력에서의 그라디언트 계산
			// 출력값과 히든스테이트는 직접 연결임으로 error = Loss 에 대한 ht 미분

			// 2 . 각 게이트에 대한 그라디언트 계산

			// outputgate 의 그라디언트
			for (int i = 0; i < gate_size; i++) {
				outputDelta[i] = error * tanh(cellstates[t][i]);
			}

			//std::cout << "test1";

			// cellgate 의 그라디언트 
			for (int i = 0; i < gate_size; i++) {
				cellDelta[i] = error * outgates[t][i] * (1 - tanh(cellstates[t][i]) *tanh(cellstates[t][i])); 
				if (t != inputs.size() - 1)
					cellDelta[i] += prevCellDelta[i] * forgetgates[t + 1][i];

			//	std::cout << "test2";
			}
			prevCellDelta = cellDelta;

			//std::cout << "test2";

			// inputgate 의 그라디언트
			for (int i = 0; i < gate_size; i++) {
				inputDelta[i] = cellDelta[i] * inputgates[t][i] * (1 - inputgates[t][i]) *cellgates[t][i];
			}

			//std::cout << "test3";

			// forgetgate 의 그라디언트

			for (int i = 0; i < gate_size; i++) {
				forgetDelta[i] = cellDelta[i]  * forgetgates[t][i] * (1 - forgetgates[t][i]);
				if( t != 0)
					forgetDelta[i] *=cellstates[t - 1][i];
			}

			//std::cout << "test4";

			// 3 . 셀 상태 업데이트

			// 4 . 가중치와 바이어스에 대한 그라디언트 계산

			for (int h = 0; h < gate_size; ++h) {
				for (int o = 0; o < output_size; ++o) {
					dWho2Sum[h][o] += error;
				}
			}
			for (int i = 0; i < input_size; ++i) {
				for (int h = 0; h < gate_size; ++h) {
					dWxfSum[i][h] +=  forgetDelta[h] * sigmoid_derivative(forgetgates[t][h]) * inputs[t][i] ;
					dWxcSum[i][h] += cellDelta[h] * tanh_derivative(cellgates[t][h]) * inputs[t][i];
					dWxiSum[i][h] += inputDelta[h] * sigmoid_derivative(inputgates[t][h]) * inputs[t][i];
					dWxoSum[i][h] += outputDelta[h] * sigmoid_derivative(outgates[t][h]) * inputs[t][i];
				}
			}
			//std::cout << "test5";
			for (int h = 0; h < gate_size; ++h) {
				for (int hh = 0; hh < gate_size; ++hh) {
					if (t != 0) {
						dWhfSum[h][hh] += forgetDelta[hh] * sigmoid_derivative(forgetgates[t][hh]) * hiddenstates[t - 1][h];
						dWhcSum[h][hh] += cellDelta[hh] * tanh_derivative(cellgates[t][hh]) * hiddenstates[t - 1][h];
						dWhiSum[h][hh] += inputDelta[hh] * sigmoid_derivative(inputgates[t][hh]) * hiddenstates[t - 1][h];
						dWhoSum[h][hh] += outputDelta[hh] * sigmoid_derivative(outgates[t][hh]) * hiddenstates[t - 1][h];
					}
				}
			}











		}
		// 업데이트
		for (int h = 0; h < gate_size; ++h) {
			for (int o = 0; o < output_size; ++o) {
				who2[h][o] -= learningRate * dWho2Sum[h][o];
			}
		}
		for (int i = 0; i < input_size; ++i) {
			for (int h = 0; h < gate_size; ++h) {
				wxf[i][h] -= learningRate * dWxfSum[i][h];
				wxc[i][h] -= learningRate * dWxcSum[i][h];
				wxi[i][h] -= learningRate * dWxiSum[i][h];
				wxo[i][h] -= learningRate * dWxoSum[i][h];
			}
		}
		for (int h = 0; h < gate_size; ++h) {
			for (int hh = 0; hh < gate_size; ++hh) {
				whf[h][hh] -= learningRate * dWhfSum[h][hh];
				whc[h][hh] -= learningRate * dWhcSum[h][hh];
				whi[h][hh] -= learningRate * dWhiSum[h][hh];
				who[h][hh] -= learningRate * dWhoSum[h][hh];
			}
		}




	}


	// 하나의 타임라인에 대해서만 계산 feed 함수
	std::vector<double> gatefeed(std::vector<double>& input , std::vector<std::vector<double>>& startX , std::vector<std::vector<double>>& startH) {


		std::vector<double> gate;

		for (int i = 0; i < gate_size; i++)
			gate.push_back(0);
		// forget gate
		// input 
		for (int i = 0; i < input_size; ++i)
			for (int j = 0; j < gate_size; ++j)
				gate[j] += startX[i][j] * input[i];
		// hidden
		for (int i = 0; i < gate_size; ++i)
			for (int j = 0; j < gate_size; ++j)
				gate[j] += startH[i][j] * hiddenstate[i];
		// bias
		for (int j = 0; j < gate_size; ++j) {
			gate[j] += bf;
		}

		return gate;


	}





};






