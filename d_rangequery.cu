/*
  Brady Adcock
  Abdel Issa 

  Implementing cool paper on range query. 

  Changes were made to Compression for the sake of cleaner 
  device code. Compression now does the following...

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "DEBUG.h"
#include "d_rangequery.h"
#include "config.h"  //defines THREADSPERBLOCK
#include "wrappers.h"

// Functions called from CPU
static __global__ void d_decompressionKernel(unsigned long* cData, 
		   	 		     unsigned long* cSizes, 
	           	 		     unsigned long cChunkSize, 
		   	 		     unsigned long* cEndPoints,
		   			     unsigned long numRows,
		   	  		     unsigned long numCols,
		   	  		     unsigned long* dData);
static __global__ void d_COAKernel(unsigned long* decompData, 
				   int numOfRows, 
				   int numOfCols);
static __device__ void d_ExclusiveScan(unsigned long* decompData, 
				       unsigned long* output, 
				       int decompDataSize); 
// Debugging Function
static __device__ void printUlongArray(unsigned long* array, 
				       int startIdx, 
				       int length);

/* d_rangequery
 
   CPU code called by the CPU. Launches decomp and COA for the time being. 
   Also handles timing of range query function.
   
   cData is 3D: (num dData columns) x (num dWords/1024) x (1024)  
   cSizes is 2D: (num dData columns) x (num dWords/1024)
   cEndPoints: inclusive scan of cSizes 

   @params:
	- cData, flattened compressed data 
	- cSizes, flattened cSizes for each column of a compressed cData column
	- cChunkSize, number of elements in cSize that makes up one compressed cData column
	- cEndPoints, endpoints for each column of a compressed cData column across entire cData
	- numRows, number of rows in dData
	- numCols, number of cols in dData 
	- dSize, number of longs in dData for a ONE SINGLE column
	- result, pointer in main mem resultant array for range query results

   @returns:
	- time taken by the GPU to complete the COA range query
*/
float d_rangequery(unsigned long* cData, 
		   unsigned long* cSizes, 
	           unsigned long cChunkSize, 
		   unsigned long* cEndPoints,
		   unsigned long numRows,
		   unsigned long numCols,
		   unsigned long* result)
{
	// timing operations 
	float gpuMsecTime;
	cudaEvent_t start_gpu, stop_gpu;
    	CHECK(cudaEventCreate(&start_gpu));
    	CHECK(cudaEventCreate(&stop_gpu));
    	CHECK(cudaEventRecord(start_gpu));

	// Device Variables
	unsigned long* d_cData;
	unsigned long* d_dData;
	unsigned long* d_cSizes;
	unsigned long* d_cEndPoints; 

	// size of arrays to be copied to and from device
	int d_cData_size = sizeof(unsigned long) * cEndPoints[numCols * cChunkSize - 1];
	int d_dData_size = sizeof(unsigned long) * numRows * numCols;
	int d_cSizes_size = sizeof(unsigned long) * numCols * cChunkSize;

	// cSize is determined by last element in cEndPoints 
	CHECK(cudaMalloc((void **) &d_cData, d_cData_size));
	// dSize is determined by numRows and numCols
	CHECK(cudaMalloc((void **) &d_dData, d_dData_size));
	// cSizes & cEndPoints are same size : determined by numCols and cChunkSize
	CHECK(cudaMalloc((void **) &d_cSizes, d_cSizes_size)); 
	CHECK(cudaMalloc((void **) &d_cEndPoints, d_cSizes_size));
	
	// copy over cData, cSizes, and cEndPoints
	CHECK(cudaMemcpy(d_cData, cData, d_cData_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cSizes, cSizes, d_cSizes_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cEndPoints, cEndPoints, d_cSizes_size, cudaMemcpyHostToDevice));

	// determine grid and block sizes for decomp launch
	// eliminating the need for a dedicated decompression kernel
	// as stated in the paper.
	int numBlocks = (numRows * numCols * 1.0) / 1024 + 1;
	dim3 grid(numBlocks, 1, 1);
	dim3 block(THREADSPERBLOCK, 1, 1);
	d_decompressionKernel<<<grid, block>>>(d_cData, 
		   			       d_cSizes, 
				               cChunkSize, 
			  		       d_cEndPoints,
		   			       numRows,
		   			       numCols,
		   			       d_dData);
	
	// wait for decomp to complete before begining the range qeury
	cudaDeviceSynchronize();

	// launch COA range query 
	d_COAKernel<<<grid, block>>>(d_dData, numRows, numCols);
	
	// wait for COA to compelete before copying result back to host 
	cudaDeviceSynchronize();
	CHECK(cudaMemcpy(result, d_dData, d_dData_size, cudaMemcpyDeviceToHost));

	// finish timing 
	CHECK(cudaEventRecord(stop_gpu));
    	CHECK(cudaEventSynchronize(stop_gpu));
    	CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
	return gpuMsecTime;
}

// Helper meant to deal with a singular column...
__device__
void d_COAHelper(unsigned long* decompData, int numOfRows, int numOfCols, int start){
	int s = numOfCols/2;
	while(s >= 1){
		if(threadIdx.x <= s){
			int firstC = threadIdx.x * numOfRows + start;
			int secondC = (threadIdx.x + s) * numOfRows + start;
			for(int i = 0; i < numOfRows; i++){
				decompData[firstC + i] |= decompData[secondC + i];
			}
		}
		s /= 2;
	}
}

// decompData is a flattened version of the data.
// 0-numOfRows-1 | numOfRows-....
// If you wanted to OR two columns, then we have an issue with accessing memory.
__global__
void d_COAKernel(unsigned long* decompData, int numOfRows, int numOfCols){
	d_COAHelper(decompData, numOfRows, numOfCols, blockIdx.x*numOfRows);
	//__syncthreads();
}

/* d_decompressionKernel
 
   GPU code called by the CPU. Decompresses each column passed and stores
   dData in result. 
   
   cData is 3D: (num dData columns) x (num dWords/1024) x (1024)  
   cSizes is 2D: (num dData columns) x (num dWords/1024)
   cEndPoints: inclusive scan of cSizes 

   @params:
	- cData, flattened compressed data 
	- cSizes, flattened cSizes for each column of a compressed cData column
	- cChunkSize, number of elements in cSize that makes up one compressed cData column
	- cEndPoints, endpoints for each column of a compressed cData column across entire cData
	- numRows, number of rows in dData
	- numCols, number of cols in dData 
	- dSize, number of longs in dData for a ONE SINGLE column
	- dData, pointer to decompressed data in global 
*/
__global__
void d_decompressionKernel(unsigned long* cData, 
		   	   unsigned long* cSizes, 
	           	   unsigned long cChunkSize, 
		   	   unsigned long* cEndPoints,
		   	   unsigned long numRows,
		   	   unsigned long numCols,
		   	   unsigned long* dData)
{
	// Shared memory needed to speed up scans
	__shared__ unsigned long decompSizes[THREADSPERBLOCK];
	__shared__ unsigned long startingPoints[THREADSPERBLOCK];
	__shared__ unsigned long endPoints[THREADSPERBLOCK];
	__shared__ unsigned long wordIndex[THREADSPERBLOCK];
	
	// thread ids
	int tid = threadIdx.x + blockIdx.x * THREADSPERBLOCK;
	// get this blocks cSize
	unsigned long cSize = cSizes[blockIdx.x];
	// get this blocks dSize : all but last block will have dSize = THREADSPERBLOCK
	unsigned long dSize = THREADSPERBLOCK;
	// get this blocks index into cData from endPoints
	int cData_index = cEndPoints[blockIdx.x] - cSize; 	
	int dData_index = blockIdx.x * THREADSPERBLOCK;

	// populate decompSizes
	unsigned long dataToPut = 0;
	if(threadIdx.x < cSize){
		dataToPut = cData[cData_index];
		if(dataToPut >> 63){ 				// if cWord is a fill atom 
			dataToPut = (dataToPut << 2) >> 2; 
		}else{						// else cWord is literal atom
			dataToPut = 1;
		}
	}
	decompSizes[threadIdx.x] = dataToPut;

	// scan decompSizes to create startingPoints
	startingPoints[threadIdx.x] = 0;
	__syncthreads();
	if(threadIdx.x == 0) // change to more optimal exclusive scan as shown in slides
		d_ExclusiveScan(decompSizes, startingPoints, cSize);
	__syncthreads();

	// create endPoints from startingPoints	
	if(threadIdx.x < cSize && threadIdx.x > 0){
		endPoints[startingPoints[threadIdx.x] - 1] = 1;
	}else{
		endPoints[threadIdx.x] = 0;
	}	

	// scan endPoints to create wordIndex
	__syncthreads();
	if(threadIdx.x == 0)
		d_ExclusiveScan(endPoints, wordIndex, dSize);
	__syncthreads();

	// update dData 
	if(threadIdx.x < dSize){
		unsigned long tempWord = cData[wordIndex[threadIdx.x] + cData_index];
		if(tempWord >> 63){
			unsigned int whatFill = (tempWord << 1) >> 63;
			if(whatFill){
				tempWord = 0x7fffffffffffffff;
			}else{
				tempWord = 0;
			}
		}
		dData[dData_index + threadIdx.x] = tempWord;
	}
}


__device__
void d_ExclusiveScan(unsigned long* data, unsigned long* output, int size){
	for(int i = 0; i < size; i++){
		if(i){
			output[i] = output[i-1] + data[i-1];
		}else{
			output[i] = 0;
		}
	}
}

__device__
void printUlongArray(unsigned long* array, int startIdx, int length){
	for(int i = startIdx; i < startIdx+length; i++){
		printf("%lx ", array[i]);
	}
	printf("\n");
}



