#include "Layer.h"

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