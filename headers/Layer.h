#ifndef LAYER_H
#define LAYER_H

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
	float layerPreoutput[O] = {0};
	float layerOutput[O] = {0};
	gpuErrchk(cudaMalloc(&pre_output, O * sizeof(float)));
	gpuErrchk(cudaMalloc(&output, O * sizeof(float)));
	gpuErrchk(cudaMalloc(&weight, N * M * sizeof(float)));
	gpuErrchk(cudaMalloc(&bias, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(pre_output, layerPreoutput, O * sizeof(float), cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(output, layerOutput, O * sizeof(float), cudaMemcpyDefault));
}

Layer::~Layer() {
	gpuErrchk(cudaFree(pre_output));
	gpuErrchk(cudaFree(output));
	gpuErrchk(cudaFree(weight));
	gpuErrchk(cudaFree(bias));
}

#endif