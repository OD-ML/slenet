#ifndef LAYER_H
#define LAYER_H

class Layer{
	public:
		int M, N, O; // O: output, N: #feature, M: #params_per_feature
		float *pre_output, *output;
		float *weight, *bias;
		Layer(int M, int N, int O);
		~Layer();
};

Layer::Layer(int M, int N, int O) {
	this->M = M;
	this->N = N;
	this->O = O;
	cudaMalloc(&pre_output, this->O * sizeof(float));
	cudaMalloc(&output, this->O * sizeof(float));
	cudaMalloc(&weight, this->N * this->M * sizeof(float));
	cudaMalloc(&bias, this->N * sizeof(float));
}

Layer::~Layer() {
	cudaFree(pre_output);
	cudaFree(output);
	cudaFree(weight);
	cudaFree(bias);
}

#endif