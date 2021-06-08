#define CONV_OUTSIZE 24
#define CONV_FTRS 6
#define CONV_WSIZE 5
#define SS_OUTSIZE 6
#define SS_FTRS 1
#define SS_WSIZE 4
#define FC_OUTSIZE 10
#define FC_FTRS 10
#define FC_WSIZE 216

const dim3 cf_numBlocks(6);
const dim3 cf_threadPerBlock(24, 24);
const dim3 cb_numBlocks(6, 6);
const dim3 cb_threadPerBlock(CONV_OUTSIZE/cb_numBlocks.x, CONV_OUTSIZE/cb_numBlocks.y, 6);
const dim3 cs_numBlocks(6, 6, 1);
const dim3 cs_threadPerBlock(CONV_OUTSIZE/cs_numBlocks.x, CONV_OUTSIZE/cs_numBlocks.y, 6/cs_numBlocks.z);
const dim3 ssf_numBlocks(CONV_OUTSIZE/SS_OUTSIZE, CONV_OUTSIZE/SS_OUTSIZE, 1);
const dim3 ssf_threadPerBlock(SS_OUTSIZE, SS_OUTSIZE, CONV_FTRS);
const dim3 ssb_numBlocks(3, 2, 2);
const dim3 ssb_threadPerBlock(SS_OUTSIZE/ssb_numBlocks.x, SS_OUTSIZE/ssb_numBlocks.y, CONV_FTRS/ssb_numBlocks.z);
const dim3 sss_numBlocks(3, 2, 2);
const dim3 sss_threadPerBlock(SS_OUTSIZE/sss_numBlocks.x, SS_OUTSIZE/sss_numBlocks.y, CONV_FTRS/sss_numBlocks.z);
const dim3 fcNumBlocks(FC_OUTSIZE);
const dim3 fcNthreadPerBlock(SS_OUTSIZE, SS_OUTSIZE, CONV_FTRS);


// FUNCTIONS FOR CONV LAYER
__global__ void kernel_conv_filter(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int ftr = blockIdx.x;
    float sum = 0;
	for (int i=0; i < 5; i++)
		for (int j=0; j < 5; j++)
			sum += 
      input[row+i][col+j] *
      weight[ftr][i][j];
    preoutput[ftr][row][col] = sum;
}

__global__ void kernel_conv_bias(
    float preoutput[][CONV_OUTSIZE][CONV_OUTSIZE], 
    float bias[]) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ftr = threadIdx.z;
	__shared__ float sh_bias[CONV_FTRS];
	if (threadIdx.x == 0 && threadIdx.y == 0)
		sh_bias[ftr] = bias[ftr];
	__syncthreads();
	preoutput[ftr][row][col] += sh_bias[ftr];
}

__global__ void kernel_conv_sigmoid(
    float preoutput[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE], 
    float output[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE]) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ftr = blockIdx.z * blockDim.z + threadIdx.z;
	output[ftr][row][col] = 1/(1+__expf(-preoutput[ftr][row][col]));
	preoutput[ftr][row][col] = 0;
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
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ftr = blockIdx.z * blockDim.z + threadIdx.z;
	__shared__ float sh_bias;
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		sh_bias = bias[0];
	__syncthreads();
    preoutput[ftr][row][col] += sh_bias;
}

__global__ void kernel_ss1_sigmoid(
    float preoutput[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE], 
    float output[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE]) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int ftr = blockIdx.z * blockDim.z + threadIdx.z;
	output[ftr][row][col] = 1/(1+__expf(-preoutput[ftr][row][col]));
    preoutput[ftr][row][col] = 0;
}

//FUNCTIONS FOR FC LAYER
__global__ void kernel_fc1_filter(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int wRow = threadIdx.x;
	int wCol = threadIdx.y;
	int wFtr = threadIdx.z;
	float mult = input[wFtr][wRow][wCol] * weight[blockIdx.x][wFtr*SS_OUTSIZE*SS_OUTSIZE+wRow*SS_OUTSIZE+wCol];
	atomicAdd(&preoutput[blockIdx.x], mult);
}

__global__ void kernel_fc1_bias(
    float preoutput[FC_OUTSIZE],
    float bias[FC_FTRS]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < FC_OUTSIZE)
		preoutput[idx] += bias[idx];
}

__global__ void kernel_fc1_sigmoid(
    float preoutput[FC_OUTSIZE],
    float output[FC_OUTSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	output[idx] = 1/(1+__expf(-preoutput[idx]));
    preoutput[idx] = 0;
}