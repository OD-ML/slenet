%%cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "/content/slenet_params.h"

#define INSIZE 28
#define CONV_OUTSIZE 24
#define CONV_FTRS 6
#define CONV_WSIZE 5
#define SS_OUTSIZE 6
#define SS_FTRS 1
#define SS_WSIZE 4
#define FC_OUTSIZE 10
#define FC_FTRS 10
#define FC_WSIZE 216

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct mnist_data{
	double data[INSIZE][INSIZE];
	unsigned int label;
} mnist_data;

static unsigned int mnist_bin_to_int(char *tmp) {
	return (tmp[0] << 24) + (tmp[1] << 16) + (tmp[2] << 8) + tmp[3];
}

static int mnist_load(const char *image_filename, 
                      const char *label_filename, 
                      mnist_data **data_set, 
                      unsigned int *count) 
{
	FILE *imgFile, *lblFile;
	imgFile = fopen(image_filename, "r");
	lblFile = fopen(label_filename, "r");
	char imgBuff[4][4];
	char lblBuff[2][4];
	fread(imgBuff, 4*sizeof(char), 4, imgFile);
	fread(lblBuff, 4*sizeof(char), 2, lblFile);
	// MAGIC NUMBER
	printf("Image magic number = %d\n", mnist_bin_to_int(imgBuff[0]));
	printf("Label magic number = %d\n", mnist_bin_to_int(lblBuff[0]));
	// TOTAL NUMBER
	int totalImg = mnist_bin_to_int(imgBuff[1]);
	int totalLbl = mnist_bin_to_int(lblBuff[1]);
	int totalTest = (totalImg == totalLbl) ? mnist_bin_to_int(imgBuff[1]) : -1;
	printf("Image total number = %d\n", totalImg);
	printf("Label total number = %d\n", totalLbl);
	// ROWS+COLS
	int rows = mnist_bin_to_int(imgBuff[2]), cols = mnist_bin_to_int(imgBuff[3]);
	printf("Rows = %d, Cols = %d\n", rows, cols);
	// LOAD DATA
	if (totalImg == totalLbl)
    *count = totalImg;
	int cnt = 0;
	for (int k=0; k<*count; k++) {
		fread(&(*data_set)[k].label, sizeof(unsigned char), 1, lblFile);
		for (int i=0; i<rows; i++) {
			for (int j=0; j<cols; j++) {
        unsigned char temp;
				fread(&temp, sizeof(unsigned char), 1, imgFile);
        (*data_set)[k].data[i][j] = temp/255.0;
      }
		}
    cnt++;
	}
	fclose(imgFile);
	fclose(lblFile);
	
	return *count-cnt;
}

void printData(mnist_data sample) {
	for (int i=0; i<28; i++) {
		for(int j=0; j<28; j++) {
			int temp = (sample.data[i][j]!=0) ? 1 : 0;
			printf("%d ", temp);
		}
		printf("\n");
	}
	printf("==> %u\n\n", sample.label); 
}

void printDatasetData(float data[INSIZE][INSIZE]) {
	for (int i=0; i<28; i++) {
		for(int j=0; j<28; j++) {
			int temp = (data[i][j]>0) ? 1 : 0;
			printf("%d ", temp);
		}
		printf("\n");
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

// FUNCTIONS FOR CONV LAYER
__global__ void kernel_conv_filter(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = blockIdx.x;
	for (int i=0; i < 5; i++)
		for (int j=0; j < 5; j++)
			preoutput[ftr][row][col] += 
      input[row+i][col+j] *
      weight[ftr][i][j];
}

__global__ void kernel_conv_bias(
    float preoutput[][CONV_OUTSIZE][CONV_OUTSIZE], 
    float bias[]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = blockIdx.x;
	preoutput[ftr][row][col] += bias[ftr];
}

__global__ void kernel_conv_sigmoid(
    float preoutput[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE], 
    float output[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = blockIdx.x;
	output[ftr][row][col] = 1/(1+exp(-preoutput[ftr][row][col]));
}

// FUNCTIONS FOR SS1 LAYER
__global__ void kernel_ss1_filter(
    float input[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE], 
    float preoutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE], 
    float weight[SS_FTRS][SS_WSIZE][SS_WSIZE]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = threadIdx.z;
	int multRow = row*gridDim.x + blockIdx.x;
	int multCol = col*gridDim.y + blockIdx.y;
	float mult = input[ftr][multRow][multCol] * weight[0][blockIdx.x][blockIdx.y];
	atomicAdd(&preoutput[ftr][row][col], mult);
}

__global__ void kernel_ss1_bias(
    float preoutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE], 
    float bias[SS_FTRS]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = threadIdx.z;
	preoutput[ftr][row][col] += bias[0];
}

__global__ void kernel_ss1_sigmoid(
    float preoutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE], 
    float output[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE]) 
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = threadIdx.z;
	output[ftr][row][col] = 1/(1+exp(-preoutput[ftr][row][col]));
}

//FUNCTIONS FOR FC LAYER
__global__ void kernel_fc1_filter(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float pre_output[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int wRow = threadIdx.x;
	int wCol = threadIdx.y;
	int wFtr = threadIdx.z;
	float mult = input[wFtr][wRow][wCol] * weight[blockIdx.x][wFtr*SS_OUTSIZE*SS_OUTSIZE+wRow*SS_OUTSIZE+wCol];
	atomicAdd(&pre_output[blockIdx.x], mult);
}

__global__ void kernel_fc1_bias(
    float pre_output[FC_OUTSIZE],
    float bias[FC_FTRS]
)
{
  pre_output[threadIdx.x] += bias[threadIdx.x];
}

__global__ void kernel_fc1_sigmoid(
    float pre_output[FC_OUTSIZE],
    float output[FC_OUTSIZE]
)
{
	output[threadIdx.x] = 1/(1+exp(-pre_output[threadIdx.x]));
}

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

float verifyArray(float arr[FC_OUTSIZE], 
                      float num) 
{
	float maxError = 0;
	for (int i=0; i < FC_OUTSIZE; i++)
		maxError = max(maxError, abs(arr[i]-num));
	return maxError;
}

int count = 0;

float forward_pass(double data[INSIZE][INSIZE]) {
	float *gInput;
  float arr[INSIZE][INSIZE];
  for (int i=0; i<INSIZE; i++)
    for (int j=0; j<INSIZE; j++)
      arr[i][j] = data[i][j];
	float convPreoutput[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE] = {0};
	float convOutput[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE] = {0};
	float ssPreoutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE] = {0};
	float ssOutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE] = {0};
	float fcPreoutput[FC_OUTSIZE] = {0};
	float fcOutput[FC_OUTSIZE] = {0};

	// Copying variables to Cuda memory
	gpuErrchk(cudaMalloc(&gInput, INSIZE*INSIZE*sizeof(float)));
	gpuErrchk(cudaMemcpy(gInput, arr, INSIZE*INSIZE*sizeof(float), cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(convNet->pre_output, 
            convPreoutput, 
            CONV_FTRS*CONV_OUTSIZE*CONV_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(convNet->output, 
            convOutput, 
            CONV_FTRS*CONV_OUTSIZE*CONV_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(convNet->weight, 
            convWeights, 
            CONV_FTRS*CONV_WSIZE*CONV_WSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(convNet->bias, 
            convBias, 
            CONV_FTRS*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(ss1Net->pre_output, 
            ssPreoutput, 
            CONV_FTRS*SS_OUTSIZE*SS_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(ss1Net->output, 
            ssOutput, 
            CONV_FTRS*SS_OUTSIZE*SS_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(ss1Net->weight, 
            ssWeights, 
            SS_FTRS*SS_WSIZE*SS_WSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(ss1Net->bias, 
            ssBias, 
            SS_FTRS*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->pre_output, 
            fcPreoutput, 
            FC_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->output, 
            fcOutput, 
            FC_OUTSIZE*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->weight, 
            fcWeights, 
            FC_WSIZE*FC_FTRS*sizeof(float), 
            cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(fcNet->bias, 
            fcBias, 
            FC_FTRS*sizeof(float), 
            cudaMemcpyDefault));
	dim3 numBlocks(CONV_FTRS);
	dim3 threadPerBlock(CONV_OUTSIZE, CONV_OUTSIZE);
	dim3 ss1NumBlocks(CONV_OUTSIZE/SS_OUTSIZE, CONV_OUTSIZE/SS_OUTSIZE);
	dim3 ss1NthreadPerBlock(SS_OUTSIZE, SS_OUTSIZE, CONV_FTRS);
	dim3 fcNumBlocks(FC_OUTSIZE);
	dim3 fcNthreadPerBlock(SS_OUTSIZE, SS_OUTSIZE, CONV_FTRS);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Convolution
	kernel_conv_filter<<<numBlocks, threadPerBlock>>>(
      (float(*)[INSIZE])gInput, 
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      (float(*)[CONV_WSIZE][CONV_WSIZE])convNet->weight);
	kernel_conv_bias<<<numBlocks, threadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      convNet->bias);
	kernel_conv_sigmoid<<<numBlocks, threadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->pre_output, 
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->output);

	// Subsampling
	kernel_ss1_filter<<<ss1NumBlocks, ss1NthreadPerBlock>>>(
      (float(*)[CONV_OUTSIZE][CONV_OUTSIZE])convNet->output, 
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      (float(*)[SS_WSIZE][SS_WSIZE])ss1Net->weight);
	kernel_ss1_bias<<<1, ss1NthreadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      ss1Net->bias);
	kernel_ss1_sigmoid<<<1, ss1NthreadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->pre_output, 
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->output);
  cudaMemcpy(ssOutput, ss1Net->output, 6*6*6*sizeof(float), cudaMemcpyDefault);

	// Fully Connected
	kernel_fc1_filter<<<fcNumBlocks, fcNthreadPerBlock>>>(
      (float(*)[SS_OUTSIZE][SS_OUTSIZE])ss1Net->output, 
      fcNet->pre_output, 
      (float(*)[FC_WSIZE])fcNet->weight);
	kernel_fc1_bias<<<1, FC_OUTSIZE>>>(fcNet->pre_output, fcNet->bias);
	kernel_fc1_sigmoid<<<1, FC_OUTSIZE>>>(fcNet->pre_output, fcNet->output);

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
