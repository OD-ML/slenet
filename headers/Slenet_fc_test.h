#define CONV_OUTSIZE 24
#define CONV_FTRS 6
#define CONV_WSIZE 5
#define SS_OUTSIZE 6
#define SS_FTRS 1
#define SS_WSIZE 4
#define FC_OUTSIZE 10
#define FC_FTRS 10
#define FC_WSIZE 216
#define FULL_MASK 0xffffffff

const dim3 cf_numBlocks(6, 6, 1);
const dim3 cf_threadPerBlock(214);
const dim3 cb_numBlocks(18, 6);
const dim3 cb_threadPerBlock(32);
const dim3 cs_numBlocks(18, 6);
const dim3 cs_threadPerBlock(32);
const dim3 ssf_numBlocks(CONV_OUTSIZE/SS_OUTSIZE, CONV_OUTSIZE/SS_OUTSIZE, 1);
const dim3 ssf_threadPerBlock(SS_OUTSIZE, SS_OUTSIZE, CONV_FTRS);
const dim3 ssb_numBlocks(3, 2, 2);
const dim3 ssb_threadPerBlock(SS_OUTSIZE/ssb_numBlocks.x, SS_OUTSIZE/ssb_numBlocks.y, CONV_FTRS/ssb_numBlocks.z);
const dim3 sss_numBlocks(3, 2, 2);
const dim3 sss_threadPerBlock(SS_OUTSIZE/sss_numBlocks.x, SS_OUTSIZE/sss_numBlocks.y, CONV_FTRS/sss_numBlocks.z);
const dim3 fcfNumBlocks7_10(7, 10);
const dim3 fcfNthreadPerBlock7_10(32, 1);
const dim3 fcfNumBlocks7_5(7, 5);
const dim3 fcfNthreadPerBlock7_5(32, 2);
const dim3 fcfNumBlocks7_3(7, 3);
const dim3 fcfNthreadPerBlock7_3(32, 4);
const dim3 fcfNumBlocks7_1(7, 1);
const dim3 fcfNthreadPerBlock7_1(32, 10);
const dim3 fcfNumBlocks4_10(4, 10);
const dim3 fcfNthreadPerBlock4_10(64, 1);
const dim3 fcfNumBlocks4_5(4, 5);
const dim3 fcfNthreadPerBlock4_5(64, 2);
const dim3 fcfNumBlocks4_3(4, 3);
const dim3 fcfNthreadPerBlock4_3(64, 4);
const dim3 fcfNumBlocks4_1(4, 1);
const dim3 fcfNthreadPerBlock4_1(64, 10);
const dim3 fcfNumBlocks2_10(2, 10);
const dim3 fcfNthreadPerBlock2_10(128, 1);
const dim3 fcfNumBlocks2_5(2, 5);
const dim3 fcfNthreadPerBlock2_5(128, 2);
const dim3 fcfNumBlocks2_3(2, 4);
const dim3 fcfNthreadPerBlock2_3(128, 4);
const dim3 fcfNumBlocks1_10(1, 10);
const dim3 fcfNthreadPerBlock1_10(256, 1);
const dim3 fcfNumBlocks1_5(1, 5);
const dim3 fcfNthreadPerBlock1_5(256, 2);
const dim3 fcfNumBlocks1_3(1, 3);
const dim3 fcfNthreadPerBlock1_3(256, 4);
const dim3 fcbsNumBlocks(10);
const dim3 fcbsNthreadPerBlock(FC_OUTSIZE/10);

// FUNCTIONS FOR CONV LAYER
__global__ void kernel_conv_filter(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 8;
	int img_col = idx % 8;
	int inp_row = blockIdx.x * 4 + img_row;
	int inp_col = blockIdx.y * 4 + img_col;
	int ftr = (idx - 64) / 25;
	int w_row = ((idx - 64) % 25) / 5;
	int w_col = ((idx - 64) % 25) % 5;
	__shared__ float sh_img[8][8];
	__shared__ float sh_weight[6][5][5];
	if (idx < 64)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 214)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 4 && w_col < 4 && ftr < 6) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 4 + w_row][blockIdx.y * 4 + w_col] = sum;
	}
}

__global__ void kernel_conv_bias(
    float preoutput[][CONV_OUTSIZE][CONV_OUTSIZE], 
    float bias[]) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row = idx / 24;
	int col = idx % 24;
	int ftr = blockIdx.y;
	preoutput[ftr][row][col] += bias[ftr];
}

__global__ void kernel_conv_sigmoid(
    float preoutput[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE], 
    float output[CONV_FTRS][CONV_OUTSIZE][CONV_OUTSIZE]) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row = idx / 24;
	int col = idx % 24;
	int ftr = blockIdx.y;
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
__global__ void kernel_fc1_filter7_10(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_5(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_3(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_1(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_10(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_5(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_3(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_1(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_10(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_5(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_3(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_10(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_5(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_3(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
        float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_10sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[32];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_5sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[32];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_3sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[32];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter7_1sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[32];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_10sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[64];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_5sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[64];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_3sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[64];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter4_1sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[64];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_10sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[128];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_5sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[128];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter2_3sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[128];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_10sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[256];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_5sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[256];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
}

__global__ void kernel_fc1_filter1_3sh(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 216) {
		__shared__ float sh_inp[256];
        int oftr = blockIdx.y * blockDim.y + threadIdx.y;
        int iftr = idx / 36;
        int row = (idx %= 36) / 6;
        int col = idx % 6;
		if (threadIdx.y == 0)
			sh_inp[threadIdx.x] = input[iftr][row][col];
		__syncthreads();
        float mult = sh_inp[threadIdx.x] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
        for (int offset = 16; offset > 0; offset /= 2)
            mult += __shfl_down_sync(FULL_MASK, mult, offset);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&preoutput[oftr], mult);
    }
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