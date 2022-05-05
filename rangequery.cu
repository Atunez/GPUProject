#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <jerror.h>
#include "wrappers.h"
#include "h_rangequery.h"

#define CHANNELS 3
#define BINSIZE 10

//prototypes for functions in this file 
void parseCommandArgs(int, char **, char **);
void printUsage();
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size);
void setBit(unsigned long &source, int position, int value);

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

    	int i, j, rows, range, numBins;
	int rowID, rowVal, binIndex;
	char rowName[200];
	// gets num rows and range of col values from top of file
    	fscanf(f, "%d", &rows);
    	fscanf(f, "%d", &range);
	numBins = range/BINSIZE;
	unsigned long cols[numBins][(rows/63) + 1];
	
	printf("rows = %d, range = %d\n", rows, range);
	printf("cols[%d][%d]\n", numBins, ((rows-1)/63) + 1);

    	// bin the column data 
    	for(i = 0; i < rows; i++){
		// parse row data from file
		fscanf(f, "%d", &rowID);
		fscanf(f, "%s", rowName);
		fscanf(f, "%d", &rowVal);
		//printf("row %d, %s, has val %d\n", rowID, rowName, rowVal);

		// decide which bin to fill
		binIndex = rowVal/BINSIZE;
		// update bins
		for(j = 0; j < numBins; j++){
			if(j == binIndex) {
				// set bit to 1
				setBit(cols[j][i/63], (i % 63) + 1, 1);
			} else {
				// set bit to 0 
				setBit(cols[j][i/63], (i % 63) + 1, 0);
			}
		}
    	}
	
	for(j = 0; j < (rows-1)/63 + 1; j++){
		for(i = 0; i < numBins; i++){
			printf("%lx ", cols[i][j]);
		}
		printf("\n");
	}

    	testCompressDecompress();
	testCompressDecompress2(&cols[i][j], 1);


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

void setBit(unsigned long &source, int position, int value)
{
	unsigned long mask = 0xffffffffffffffff;
	mask = mask << position;
	mask = mask >> 63;
	mask = mask << 63 - position;
	if (value == 1 ) { 
		source = source | mask;
	} else {
		mask = ~mask;
		source = source & mask;
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
