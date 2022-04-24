#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <jerror.h>
#include "wrappers.h"
#include "h_rangequery.h"

#define CHANNELS 3

//prototypes for functions in this file 
void parseCommandArgs(int, char **, char **);
void printUsage();
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size);

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

    int i, rows, range;
    // read basic information...
    fscanf(f, "%d", &rows);
    fscanf(f, "%d", &range);
    // for(i = 0; i < rows; i++){

    // }

    testCompressDecompress();


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
