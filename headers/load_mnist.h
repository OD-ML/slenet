#include <stdio.h>
#include <stdlib.h>

#define INSIZE 28

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