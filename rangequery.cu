#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "wrappers.h"
#include "h_rangequery.h"
#include "d_rangequery.h"

#define CHANNELS 3

//prototypes for functions in this file 
void parseCommandArgs(int, char **, char **);
void printUsage();
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size);
void setBit(unsigned long &source, int position, int value);
void printBins(int rows, int numBins, unsigned long ** cols);
void checkDecomp(unsigned long* gpuResult, unsigned long** realData, int dataInCol);

int main(int argc, char * argv[])
{
    // read inputs...
    char * fileName;
    parseCommandArgs(argc, argv, &fileName);

    FILE* f = fopen(fileName, "r");
    if(f == NULL){
        printf("Error: Couldn't Open file");
        exit(EXIT_FAILURE);
    } 

    int i, j, rows, range;
	int rowID, rowVal, binIndex;
	int argCnt = 0;

	// gets num rows and range of col values from top of file
	argCnt += fscanf(f, "%d", &rows);
	argCnt += fscanf(f, "%d", &range);

	if(argCnt != 2){
		printf("Can't Read Start of .gen File :(");
		return EXIT_FAILURE;
	}

	argCnt = 0;

	// We will assume that we have a multiple of 63 to make life easier
	// figure out how many words we need to represent the data vertically.
	// Ie, if this is 10, then we need 10 uls for 1 column, ie, we have 630 elements
	int numOfWords = rows/63; 
	// figure out how many bins we have
	int numBins = range/BINSIZE;
	// make an array to story all the data...
	unsigned long cols[numBins][numOfWords];
	unsigned long ** realCols = (unsigned long **) Malloc(numBins * sizeof(sizeof(unsigned long) * numOfWords));
	for(i = 0; i < numBins; i++){
		for(j = 0; j < numOfWords; j++){
			cols[i][j] = 0;
		}
	}
	// every row has its label, which we will need for later...
	// but, for cols[i][j], its label should be 
	// 0 1 0
	// 1 0 0
	// 0 0 1
	char ** labels = (char **) Malloc(sizeof(11 * sizeof(char)) * rows);
	unsigned long* basic_values = (unsigned long *) Malloc(sizeof(unsigned long) * rows);
	
	//printf("rows = %d, range = %d\n", rows, range);
	//printf("cols[%d][%d]\n", numBins, numOfWords);

	// bin the column data 
	for(i = 0; i < rows; i++){
		// parse row data from file
		// argCnt += fscanf(f, "%d", &rowID);
		// argCnt += fscanf(f, "%s", labels[i]);
		// argCnt += fscanf(f, "%d", &rowVal);
		
		char* label = (char *) Malloc(sizeof(char) * 11);
		argCnt = fscanf(f, "%d %s %d", &rowID, label, &rowVal);
		labels[i] = label;
		basic_values[rowID] = rowVal;

		if(argCnt != 3){
			printf("Can't Read Start of .gen File, in loop :( %d", argCnt);
			return EXIT_FAILURE;
		}

		argCnt = 0;

		//printf("row %d, %s, has val %d\n", rowID, rowName, rowVal);

		// decide which bin to fill
		binIndex = rowVal/BINSIZE;
		// update bins
		for(j = 0; j < numBins; j++){
			if(j == binIndex) {
				// set bit to 1
				setBit(cols[j][i/63], (i % 63), 1);
			} else {
				// set bit to 0 
				setBit(cols[j][i/63], (i % 63), 0);
			}
		}
	}

	printf("Finished Reading File :) \n");

	// Answers a query directly O(nk)
	// k is length of bins of interest
	// n is the number of rows
	// int binsOfInterest[4] = {0,3,5,7};
	// simpleQuery(binsOfInterest, basic_values, labels, rows, 4);

	// int k;
	// for(k = 0; k < numBins; k++){
	// 	realCols[k] = cols[k];
	// }


	// for(k = 0; k < numOfWords; k++){
	// 	printf("%lx \n", realCols[0][k]);
	// }
	

	realCols[0] = compress(cols[0], numOfWords);
	realCols[1] = compress(cols[3], numOfWords);
	realCols[2] = compress(cols[5], numOfWords);
	realCols[3] = compress(cols[7], numOfWords);

	unsigned long** temps = (unsigned long **) Malloc(4 * sizeof(sizeof(unsigned long) * numOfWords));;
	temps[0] = cols[0];
	temps[1] = cols[3];
	temps[2] = cols[5];
	temps[3] = cols[7];

	unsigned long* result = (unsigned long *) Malloc(sizeof(unsigned long) * numOfWords * 4);
	printf("result has finished compressing\n");
	// int y = 1;
	// for(i = 0; i < numOfWords; i++){
	// 	printf("%lx ", cols[y][i]);
	// }
	// printf("\n");
	// for(i = 0; i < 1+realCols[y][0]; i++){
	// 	printf("%lx ", realCols[y][i]);
	// }
	// printf("\n");
	// for(i = 0; i < 1+realCols[1][0]; i++){
	// 	printf("%lx ", realCols[1][i]);
	// }
	// printf("\n");
	// 	for(i = 0; i < 1+realCols[2][0]; i++){
	// 	printf("%lx ", realCols[2][i]);
	// }
	// printf("\n");
	// 	for(i = 0; i < 1+realCols[3][0]; i++){
	// 	printf("%lx ", realCols[3][i]);
	// }
	// printf("\n");
	int k;

	// for(k = 0; k < numOfWords; k++){
	// 	printf("%lx %lx %lx %lx \n", cols[0][k], cols[3][k], cols[5][k], cols[7][k]);
	// }
	float time_taken_gpu = d_rangequery(realCols, result, 4, numOfWords);

	//checkDecomp(result, temps, numOfWords);

	printf("Time of range query on a GPU: %f msec\n", time_taken_gpu);

	realCols[0] = cols[0];
	realCols[1] = cols[3];
	realCols[2] = cols[5];
	realCols[3] = cols[7];
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

	CHECK(cudaEventCreate(&start_cpu));  
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));   


	long unsigned* temp = COA(realCols, 4, numOfWords);
	
	CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu)); 
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));

	printf("Time of range query on a CPU: %f msec\n", cpuMsecTime);
	printf("Speedup: %f\n", cpuMsecTime/time_taken_gpu);

		// for(k = 0; k < numOfWords; k++){
		// 	printf("%lx \n", temp[k]);
		// }

	// int k;
	for(k = 0; k < numOfWords; k++){
		if(temp[k] != result[k]){
			printf("Error, GPU isn't getting the same as CPU! \n");
			printf("CPU: %lx GPU: %lx row: %d \n", temp[k], result[k], k);
			return EXIT_FAILURE;
		}else{
			//printf("CPU: %lx GPU: %lx row: %d \n", temp[k], result[k], k); 
		}
	}

	printf("Yay, COA Works...\n");
	
	// for label[i] it goes to cols[X][i/63] 

	// testCompressDecompress();
	// printf("%lx\n", cols[0][0]);
	// testCompressDecompress(&cols[0][0], 1);
	// testCompressDecompress(cols[0], numOfWords);

	// // testing GPU access to compressed words
	
	// // number of unsigned longs used to compress a col. 
	// unsigned long numCWords;
	// // compressed cols
	// unsigned long * cData;

	// cData = compress(cols[0], numOfWords);
	// numCWords = cData[0];
	
	// printf("numCWords: %lu\n", numCWords);

	// for(i = 1; i < numCWords + 1; i++) 
	// {
	// 	printf("%lx\n", cData[i]);
	// }

    // //use the CPU to perform the greyscale
    // unsigned char * h_Pout; 
    // h_Pout = (unsigned char *) Malloc(sizeof(unsigned char) * width * height);
    // float cpuTime = h_colorToGreyscale(h_Pout, Pin, width, height);

    // //use the GPU to perform the greyscale 
    // unsigned char * d_Pout; 
    // d_Pout = (unsigned char *) Malloc((sizeof(unsigned char) * width * height));
    // float gpuTime = d_colorToGreyscale(d_Pout, Pin, width, height, blkWidth, blkHeight);

    // //compare the CPU and GPU results
    // compare(d_Pout, h_Pout, width * height);

    // printf("CPU time: %f msec\n", cpuTime);
    // printf("GPU time: %f msec\n", gpuTime);
    // printf("Speedup: %f\n", cpuTime/gpuTime);
    return EXIT_SUCCESS;
}

void checkDecomp(unsigned long* gpuResult, unsigned long** realData, int numInCol){
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < numInCol; j++){
			if(gpuResult[i*numInCol + j] != realData[i][j]){
				printf("Something wrong in Decomp :( %lx %lx %d %d  %lx\n", gpuResult[i*numInCol + j], realData[i][j], i, i*numInCol + j, gpuResult[i*numInCol + j+1]);
				exit(0);
			}
		}
	}
}

void printBins(int rows, int numBins, unsigned long ** cols)
{
	int i, j;
	for(j = 0; j < (rows-1)/63 + 1; j++){
		for(i = 0; i < numBins; i++){
			printf("%lx \n", cols[i][j]);
		}
		printf("\n---63 rows---\n");
	}
}



void setBit(unsigned long &source, int position, int value)
{
	// unsigned long mask = 0xffffffffffffffff;
	// mask = mask << position;
	// mask = mask >> 63;
	// mask = mask << 63 - position;
	// if (value == 1 ) { 
	// 	source = source | mask;
	// } else {
	// 	mask = ~mask;
	// 	source = source & mask;
	// }
	// need to set position to value in source...

	if(value){
		unsigned long mask = 0x0000000000000001;
		// we have 1 at the start
		source |= (mask << position);
	}else{
		unsigned long mask = 0x0000000000000001;
		source &= ~(mask << position);
	}
}

void parseCommandArgs(int argc, char * argv[], char ** fileNm)
{
    struct stat buffer;
    if (argc != 2) printUsage();

    int len = strlen(argv[1]);
    if (strncmp(".gen", &argv[1][len - 4], 4) != 0) printUsage();

    //stat function returns 1 if file does not exist
    if (stat(argv[1], &buffer)) printUsage();
    *fileNm = argv[1];
}


void printUsage()
{
    printf("This application takes as input the name of a .gen file\n");
    printf("Thie file should be generated from generatedata.py\n");
    printf("To use that file, provide as input from system.in a file\n");
    printf("That contains two numbers: # of Elements, # of buckets (each of size 100) \n");
    printf("    Example:\n");
    printf("    python3 generatedata.py < 10 5 > out.gen\n");
    printf("    ./rangequery out.gen\n");
    exit(EXIT_FAILURE);
}
