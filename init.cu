#include <stdio.h>
#include <stdlib.h>
#include "headers/slenet_params.h"
#include "headers/load_mnist.h"
#include "headers/Layer.h"
#include "headers/Slenet_shv2.h"

// Layer declarations
Layer *convNet;
Layer *ss1Net;
Layer *fcNet;

float convWeights[CONV_FTRS][CONV_WSIZE][CONV_WSIZE];
float convBias[CONV_FTRS];
float ssWeights[SS_FTRS][SS_WSIZE][SS_WSIZE];
float ssBias[SS_FTRS];
float fcWeights[FC_FTRS][FC_WSIZE];
float fcBias[FC_FTRS];

int count = 0;

float forward_pass(double data[INSIZE][INSIZE]) {
	float *gInput;
	float arr[INSIZE][INSIZE];
	for (int i=0; i<INSIZE; i++)
		for (int j=0; j<INSIZE; j++)
			arr[i][j] = data[i][j];

	// Copying input to Cuda memory
	gpuErrchk(cudaMalloc(&gInput, INSIZE*INSIZE*sizeof(float)));
	gpuErrchk(cudaMemcpy(gInput, arr, INSIZE*INSIZE*sizeof(float), cudaMemcpyDefault));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Convolution
	kernel_conv_filter<<<cf_numBlocks, cf_threadPerBlock>>>(
      (float(*)[INSIZE])gInput, 
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      (float(*)[CONV_WSIZE][CONV_WSIZE])convNet->weight);
	kernel_conv_bias<<<cb_numBlocks, cb_threadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      convNet->bias);
	kernel_conv_sigmoid<<<cs_numBlocks, cs_threadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->output);

	// Subsampling
	kernel_ss1_filter<<<ssf_numBlocks, ssf_threadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->output, 
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      (float(*)[SS_WSIZE][SS_WSIZE])ss1Net->weight);
	kernel_ss1_bias<<<ssb_numBlocks, ssb_threadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      ss1Net->bias);
	kernel_ss1_sigmoid<<<sss_numBlocks, sss_threadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->output);

	// Fully Connected
	kernel_fc1_filter<<<fcfNumBlocks, fcfNthreadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->output, 
      fcNet->pre_output, 
      (float(*)[FC_WSIZE])fcNet->weight);
	kernel_fc1_bias<<<fcbsNumBlocks, fcbsNthreadPerBlock>>>(fcNet->pre_output, fcNet->bias);
	kernel_fc1_sigmoid<<<fcbsNumBlocks, fcbsNthreadPerBlock>>>(fcNet->pre_output, fcNet->output);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float eltime;
	cudaEventElapsedTime(&eltime, start, stop);
	return eltime;
}

int main() {
	int ret; int i;
	mnist_data *dataset = new mnist_data[10000];
	static unsigned int test_cnt;
	
	// load data
	if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &dataset, &test_cnt) != 0)
		printf("An error occurred: %d \n", ret);
	else
		printf("test_cnt = %d \n", test_cnt); // test_cnt must have the number of test images (i.e., 10K)

	for (int i=0; i < CONV_FTRS; i++)
		for (int j = 0; j < CONV_WSIZE; j++)
			for (int k = 0; k < CONV_WSIZE; k++)
				convWeights[i][j][k] = c1_weight[i][j*CONV_WSIZE+k];
  
	for (int i=0; i < CONV_FTRS; i++)
		convBias[i] = c1_bias[i];

	for (int i=0; i < SS_FTRS; i++)
		for (int j = 0; j < SS_WSIZE; j++)
			for (int k = 0; k < SS_WSIZE; k++)
				ssWeights[i][j][k] = s2_weight[i][j*SS_WSIZE+k];
  
	for (int i=0; i < SS_FTRS; i++)
		ssBias[i] = s2_bias[i];
  
	for (int i=0; i < FC_FTRS; i++)
		for (int j=0; j < FC_WSIZE; j++)
			fcWeights[i][j] = f3_weight[i][j];
  
	for (int i=0; i < FC_FTRS; i++)
		fcBias[i] = f3_bias[i];

	convNet = new Layer(CONV_WSIZE*CONV_WSIZE, CONV_FTRS, CONV_FTRS*CONV_OUTSIZE*CONV_OUTSIZE);
	ss1Net = new Layer(SS_WSIZE*SS_WSIZE, SS_FTRS, CONV_FTRS*SS_OUTSIZE*SS_OUTSIZE);
	fcNet = new Layer(FC_WSIZE, FC_FTRS, FC_OUTSIZE);
  	gpuErrchk(cudaMemcpy(convNet->weight, 
                      convWeights, 
                      CONV_WSIZE * CONV_WSIZE * CONV_FTRS * sizeof(float), 
                      cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(convNet->bias, 
                      convBias, 
                      CONV_FTRS * sizeof(float), 
                      cudaMemcpyDefault));
  	gpuErrchk(cudaMemcpy(ss1Net->weight, 
                      ssWeights, 
                      SS_FTRS * SS_WSIZE * SS_WSIZE * sizeof(float), 
                      cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(ss1Net->bias, 
                      ssBias, 
                      SS_FTRS * sizeof(float), 
                      cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->weight, 
                      fcWeights, FC_FTRS * FC_WSIZE * sizeof(float), 
                      cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->bias, 
                      fcBias, FC_FTRS * sizeof(float), 
                      cudaMemcpyDefault));
  
	float time_taken = 0;
	unsigned int error = 0;
	unsigned int max = 0;
	float res[10];
  
	for (i=0; i<10000; i++){
    time_taken += forward_pass(dataset[i].data);
    cudaMemcpy(res, fcNet->output, sizeof(float)*10, cudaMemcpyDefault);
    for(int j=0; j<10; j++){
      if (res[max] < res[j])
        max = j;
      }
    if (max != dataset[i].label) ++error; // error must have the number of incorrect predictions.
	}
	printf("Error Rate = %f%% (%d out of 10,000)\n", double(error)/double(test_cnt)*100.0, error);
	printf("Accuracy = %.3f%% (%d out of 10,000)\n",
		 100.0 - double(error)/double(test_cnt)*100.0, test_cnt - error);
	printf("Ex time = %f (ms) \n", time_taken);
  
	delete[] dataset;
	delete convNet;
	delete ss1Net;
	delete fcNet;
	return 0;
}
