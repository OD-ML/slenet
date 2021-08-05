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

const dim3 cf_numBlocks2_1(12, 12, 6);
const dim3 cf_threadPerBlock2_1(61);
const dim3 cf_numBlocks2_2(12, 12, 3);
const dim3 cf_threadPerBlock2_2(86);
const dim3 cf_numBlocks2_3(12, 12, 2);
const dim3 cf_threadPerBlock2_3(111);
const dim3 cf_numBlocks2_6(12, 12, 1);
const dim3 cf_threadPerBlock2_6(186);
const dim3 cf_numBlocks4_1(6, 6, 6);
const dim3 cf_threadPerBlock4_1(89);
const dim3 cf_numBlocks4_2(6, 6, 3);
const dim3 cf_threadPerBlock4_2(139);
const dim3 cf_numBlocks4_3(6, 6, 2);
const dim3 cf_threadPerBlock4_3(139);
const dim3 cf_numBlocks4_6(6, 6, 1);
const dim3 cf_threadPerBlock4_6(214);
const dim3 cf_numBlocks6_1(4, 4, 6);
const dim3 cf_threadPerBlock6_1(125);
const dim3 cf_numBlocks6_2(4, 4, 3);
const dim3 cf_threadPerBlock6_2(150);
const dim3 cf_numBlocks6_3(4, 4, 2);
const dim3 cf_threadPerBlock6_3(175);
const dim3 cf_numBlocks6_6(4, 4, 1);
const dim3 cf_threadPerBlock6_6(250);
const dim3 cf_numBlocks8_1(3, 3, 6);
const dim3 cf_threadPerBlock8_1(169);
const dim3 cf_numBlocks8_2(3, 3, 3);
const dim3 cf_threadPerBlock8_2(194);
const dim3 cf_numBlocks8_3(3, 3, 2);
const dim3 cf_threadPerBlock8_3(219);
const dim3 cf_numBlocks8_6(3, 3, 1);
const dim3 cf_threadPerBlock8_6(384);

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
const dim3 fcfNumBlocks(10, 7);
const dim3 fcfNthreadPerBlock(32);
const dim3 fcbsNumBlocks(10);
const dim3 fcbsNthreadPerBlock(FC_OUTSIZE/10);


// FUNCTIONS FOR CONV LAYER
__global__ void kernel_conv_filter2_1(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 6;
	int img_col = idx % 6;
	int inp_row = blockIdx.x * 2 + img_row;
	int inp_col = blockIdx.y * 2 + img_col;
	int ftr = (idx - 36) / 25;
	int w_row = ((idx - 36) % 25) / 5;
	int w_col = ((idx - 36) % 25) % 5;
	__shared__ float sh_img[6][6];
	__shared__ float sh_weight[1][5][5];
	if (idx < 36)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 61)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 2 && w_col < 2 && ftr < 1) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 2 + w_row][blockIdx.y * 2 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter2_2(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 6;
	int img_col = idx % 6;
	int inp_row = blockIdx.x * 2 + img_row;
	int inp_col = blockIdx.y * 2 + img_col;
	int ftr = (idx - 36) / 25;
	int w_row = ((idx - 36) % 25) / 5;
	int w_col = ((idx - 36) % 25) % 5;
	__shared__ float sh_img[6][6];
	__shared__ float sh_weight[2][5][5];
	if (idx < 36)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 86)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 2 && w_col < 2 && ftr < 2) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 2 + w_row][blockIdx.y * 2 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter2_3(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 6;
	int img_col = idx % 6;
	int inp_row = blockIdx.x * 2 + img_row;
	int inp_col = blockIdx.y * 2 + img_col;
	int ftr = (idx - 36) / 25;
	int w_row = ((idx - 36) % 25) / 5;
	int w_col = ((idx - 36) % 25) % 5;
	__shared__ float sh_img[6][6];
	__shared__ float sh_weight[3][5][5];
	if (idx < 36)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 111)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 2 && w_col < 2 && ftr < 3) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 2 + w_row][blockIdx.y * 2 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter2_6(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 6;
	int img_col = idx % 6;
	int inp_row = blockIdx.x * 2 + img_row;
	int inp_col = blockIdx.y * 2 + img_col;
	int ftr = (idx - 36) / 25;
	int w_row = ((idx - 36) % 25) / 5;
	int w_col = ((idx - 36) % 25) % 5;
	__shared__ float sh_img[6][6];
	__shared__ float sh_weight[6][5][5];
	if (idx < 36)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 186)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 2 && w_col < 2 && ftr < 6) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 2 + w_row][blockIdx.y * 2 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter4_1(
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
	__shared__ float sh_weight[1][5][5];
	if (idx < 64)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 89)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 4 && w_col < 4 && ftr < 1) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 4 + w_row][blockIdx.y * 4 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter4_2(
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
	__shared__ float sh_weight[2][5][5];
	if (idx < 64)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 114)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 4 && w_col < 4 && ftr < 2) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 4 + w_row][blockIdx.y * 4 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter4_3(
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
	__shared__ float sh_weight[3][5][5];
	if (idx < 64)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 139)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 25) / 5;
    w_col = (idx % 25) % 5;
    ftr = idx / 25;
	if (w_row < 4 && w_col < 4 && ftr < 3) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 4 + w_row][blockIdx.y * 4 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter4_6(
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

__global__ void kernel_conv_filter6_1(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 10;
	int img_col = idx % 10;
	int inp_row = blockIdx.x * 6 + img_row;
	int inp_col = blockIdx.y * 6 + img_col;
	int ftr = (idx - 100) / 25;
	int w_row = ((idx - 100) % 25) / 5;
	int w_col = ((idx - 100) % 25) % 5;
	__shared__ float sh_img[10][10];
	__shared__ float sh_weight[1][5][5];
	if (idx < 100)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 125)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 36) / 6;
    w_col = (idx % 36) % 6;
    ftr = idx / 36;
	if (ftr < 1) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 6 + w_row][blockIdx.y * 6 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter6_2(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 10;
	int img_col = idx % 10;
	int inp_row = blockIdx.x * 6 + img_row;
	int inp_col = blockIdx.y * 6 + img_col;
	int ftr = (idx - 100) / 25;
	int w_row = ((idx - 100) % 25) / 5;
	int w_col = ((idx - 100) % 25) % 5;
	__shared__ float sh_img[10][10];
	__shared__ float sh_weight[2][5][5];
	if (idx < 100)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 150)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 36) / 6;
    w_col = (idx % 36) % 6;
    ftr = idx / 36;
	if (ftr < 2) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 6 + w_row][blockIdx.y * 6 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter6_3(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 10;
	int img_col = idx % 10;
	int inp_row = blockIdx.x * 6 + img_row;
	int inp_col = blockIdx.y * 6 + img_col;
	int ftr = (idx - 100) / 25;
	int w_row = ((idx - 100) % 25) / 5;
	int w_col = ((idx - 100) % 25) % 5;
	__shared__ float sh_img[10][10];
	__shared__ float sh_weight[3][5][5];
	if (idx < 100)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 175)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 36) / 6;
    w_col = (idx % 36) % 6;
    ftr = idx / 36;
	if (ftr < 3) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 6 + w_row][blockIdx.y * 6 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter6_6(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 10;
	int img_col = idx % 10;
	int inp_row = blockIdx.x * 6 + img_row;
	int inp_col = blockIdx.y * 6 + img_col;
	int ftr = (idx - 100) / 25;
	int w_row = ((idx - 100) % 25) / 5;
	int w_col = ((idx - 100) % 25) % 5;
	__shared__ float sh_img[10][10];
	__shared__ float sh_weight[6][5][5];
	if (idx < 100)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 250)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 36) / 6;
    w_col = (idx % 36) % 6;
    ftr = idx / 36;
	if (ftr < 6) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 6 + w_row][blockIdx.y * 6 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter8_1(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 12;
	int img_col = idx % 12;
	int inp_row = blockIdx.x * 8 + img_row;
	int inp_col = blockIdx.y * 8 + img_col;
	int ftr = (idx - 144) / 25;
	int w_row = ((idx - 144) % 25) / 5;
	int w_col = ((idx - 144) % 25) % 5;
	__shared__ float sh_img[12][12];
	__shared__ float sh_weight[1][5][5];
	if (idx < 144)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 169)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 64) / 8;
    w_col = (idx % 64) % 8;
    ftr = idx / 64;
	if (w_row < 8 && w_col < 8 && ftr < 1) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 8 + w_row][blockIdx.y * 8 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter8_2(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 12;
	int img_col = idx % 12;
	int inp_row = blockIdx.x * 8 + img_row;
	int inp_col = blockIdx.y * 8 + img_col;
	int ftr = (idx - 144) / 25;
	int w_row = ((idx - 144) % 25) / 5;
	int w_col = ((idx - 144) % 25) % 5;
	__shared__ float sh_img[12][12];
	__shared__ float sh_weight[2][5][5];
	if (idx < 144)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 194)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 64) / 8;
    w_col = (idx % 64) % 8;
    ftr = idx / 64;
	if (w_row < 8 && w_col < 8 && ftr < 2) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 8 + w_row][blockIdx.y * 8 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter8_3(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 12;
	int img_col = idx % 12;
	int inp_row = blockIdx.x * 8 + img_row;
	int inp_col = blockIdx.y * 8 + img_col;
	int ftr = (idx - 144) / 25;
	int w_row = ((idx - 144) % 25) / 5;
	int w_col = ((idx - 144) % 25) % 5;
	__shared__ float sh_img[12][12];
	__shared__ float sh_weight[3][5][5];
	if (idx < 144)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 219)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 64) / 8;
    w_col = (idx % 64) % 8;
    ftr = idx / 64;
	if (w_row < 8 && w_col < 8 && ftr < 3) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 8 + w_row][blockIdx.y * 8 + w_col] = sum;
	}
}

__global__ void kernel_conv_filter8_6(
    float input[][28], 
    float preoutput[][24][24], 
    float weight[][5][5])
{
	int idx = threadIdx.x;
	int img_row = idx / 12;
	int img_col = idx % 12;
	int inp_row = blockIdx.x * 8 + img_row;
	int inp_col = blockIdx.y * 8 + img_col;
	int ftr = (idx - 144) / 25;
	int w_row = ((idx - 144) % 25) / 5;
	int w_col = ((idx - 144) % 25) % 5;
	__shared__ float sh_img[12][12];
	__shared__ float sh_weight[6][5][5];
	if (idx < 144)
		sh_img[img_row][img_col] = input[inp_row][inp_col];
	else if (idx < 294)
		sh_weight[ftr][w_row][w_col] = weight[blockIdx.z * CONV_FTRS / gridDim.z + ftr][w_row][w_col];
	__syncthreads();
	float sum = 0;
    w_row = (idx % 64) / 8;
    w_col = (idx % 64) % 8;
    ftr = idx / 64;
	if (w_row < 8 && w_col < 8 && ftr < 6) {
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ftr][i][j];
		preoutput[blockIdx.z * CONV_FTRS / gridDim.z + ftr][blockIdx.x * 8 + w_row][blockIdx.y * 8 + w_col] = sum;
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
__global__ void kernel_fc1_filter(
    float input[CONV_FTRS][SS_OUTSIZE][SS_OUTSIZE],
    float preoutput[FC_OUTSIZE],
    float weight[FC_FTRS][FC_WSIZE]
)
{
	int idx = blockIdx.y * blockDim.x + threadIdx.x;
	int oftr = blockIdx.x;
	int iftr = idx / 36;
	int row = (idx %= 36) / 6;
	int col = idx % 6;
	float mult = input[iftr][row][col] * weight[oftr][iftr*SS_OUTSIZE*SS_OUTSIZE+row*SS_OUTSIZE+col];
	for (int offset = 16; offset > 0; offset /= 2)
    	mult += __shfl_down_sync(FULL_MASK, mult, offset);
	if (threadIdx.x == 0)
		atomicAdd(&preoutput[oftr], mult);
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