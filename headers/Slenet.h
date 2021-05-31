#include <math.h>

#define CONV_OUTSIZE 24
#define CONV_FTRS 6
#define CONV_WSIZE 5
#define SS_OUTSIZE 6
#define SS_FTRS 1
#define SS_WSIZE 4
#define FC_OUTSIZE 10
#define FC_FTRS 10
#define FC_WSIZE 216

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