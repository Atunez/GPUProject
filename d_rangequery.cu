/*
  Brady Adcock
  Abdel Issa 

  Implementing cool paper on range query. 
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
#include "NEXTPOW2.h"


// prototype for kernels
// __global__ void decompress (ulong *, ulong *, ulong *, ulong, int, ulong *, ulong *, ulong *);
// static __global__ void makeDecompSizes(ulong*, ulong, ulong*);
// static __global__ void sumKernel(ulong *, ulong *, ulong);
// static __global__ void sweepKernel(ulong *, ulong *) ;
// __device__ void gpuPrintVec(const char * label, ulong * vector, ulong length);

// /*
//  Given an array of relevant cols (representing bins which
//  classify the variable of interest) return resultant 
//  bit vector R. 

//  params,
//  	cols:   array of bitvectors representing bins of variable 
//  		being queried. 2d array reprsented as 1d array because
// 		each element of the array is variable length.  
// 	cSizes: array desciribing the size of each compressed col
// 		in terms of number of 64 bit words.  
// 	dSizes: number of 64 bit words in the decompressed cols
//  return,
// 	R:	bitvector representing rows who match range query.
// */
// void d_decompress (ulong * cols, ulong * cSizes, ulong * dData, ulong dSize, int numCols)
// {
// 	ulong * R; 
// 	ulong * d_cols; 	// 1d array representing the 2d array of compressed bins
// 	ulong * d_cSizes;  
// 	ulong * d_dData; 	// 1d array representing the 2d array of decompressed bins

// 	// following mem is only used in decomp
// 	ulong * d_decompSizes;  // becomes startingPoints after scan
// 	ulong * d_startingPoints;
// 	ulong * d_endPoints;    // becomes wordIndex after scan
// 	ulong * d_wordIndexes;
// 	ulong * d_sums;		// used for scan

// 	ulong cTotalSize = 0; // size used for cols, decompSize, startingPoints 2d arrays
// 	ulong dTotalSize = 0; // size used for endPoints and WordIndex 2d arrays
// 	int i;
// 	// determine total sizes based off of the num of cols, cData. 
// 	for(i = 0; i < numCols; i++) 
// 	{
// 		// have to ensure these sizes are powers of two for the sake of exclusive scan efficeny 
// 		cTotalSize += NEXTPOW2(cSizes[i]);
// 		dTotalSize += NEXTPOW2(dSize);
// 	}
// 	cTotalSize *= sizeof(ulong);
// 	dTotalSize *= sizeof(ulong);

// 	// malloc cData && dData
// 	CHECK(cudaMalloc((void **) &d_cols, cTotalSize));
// 	CHECK(cudaMalloc((void **) &d_dData, dSize * sizeof(ulong)));
// 	// malloc cSizes
// 	CHECK(cudaMalloc((void **) &d_cSizes, numCols * sizeof(ulong)));
// 	// malloc decompSizes arry for decomp
// 	CHECK(cudaMalloc((void **) &d_decompSizes, cTotalSize));
// 	CHECK(cudaMalloc((void **) &d_startingPoints, cTotalSize));
// 	// malloc endPoints array for decomp
// 	CHECK(cudaMalloc((void **) &d_endPoints, dSize * sizeof(char)));
// 	CHECK(cudaMalloc((void **) &d_wordIndexes, dTotalSize));
// 	// malloc sum array, used for the exclusive scans in decomp.
// 	// worst case it is of size dSize/THREADSPERBLOCK for the exlusive scan on endPoints
// 	CHECK(cudaMalloc((void **) &d_sums, NEXTPOW2(dSize) * numCols * sizeof(ulong)/THREADSPERBLOCK));
// 	// malloc R (same as dSize)
// 	CHECK(cudaMalloc((void **) &R, dTotalSize));	

// 	// copy cData over to device
// 	CHECK(cudaMemcpy(d_cols, cols, totalSize, cudaMemcpyHostToDevice));
// 	// copy cSizes over to device
// 	CHECK(cudaMemcpy(d_cSizes, cSizes, numCols * sizeof(ulong), cudaMemcpyHostToDevice));

// 	// make grid && block 
// 	dim3 grid(1, 1, 1); 		// only need a few threads because this 
// 	dim3 block(numCols, 1, 1);	// kernel will launch many more

// 	// launch kernel decompress
// 	decompress<<<grid, block>>>(d_cols, d_cSizes, d_dData, dSize, numCols, 
// 				    d_decompSizes, d_startingPoints, d_endPoints, d_wordIndexes, d_sums);

// 	// copy decompressed data back from device
// 	CHECK(cudaMemcpy(dData, d_dData, dSize * sizeof(ulong), cudaMemcpyDeviceToHost));

// 	// cudaFree everything on device
// 	CHECK(cudaFree(d_cols));
// 	CHECK(cudaFree(d_cSizes));
// 	CHECK(cudaFree(d_dData));
// 	CHECK(cudaFree(d_decompSizes));
// 	CHECK(cudaFree(d_endPoints));
// 	CHECK(cudaFree(d_sums));
// }

// /*
//  Given pointer to array of WAH bitvectors, decompress them
//  by launching decomp kernels.

//  params,	
// 	cols:	array of bitvectors with WAH 64 bit encoding
// */
// __global__ void decompress (ulong * cols, ulong * cSizes, ulong * dData, ulong dSize, int numCols,
// 			    ulong * decompSizes, ulong * startingPoints, ulong * endPoints, 
// 			    ulong* wordIndexes, ulong * sums) 
// {
// 	int col = threadIdx.x;
// 	ulong col_offset = 0;	
// 	ulong sum_offset = 0;	
// 	ulong * bitVec, decompSize, sum;
// 	int i;

// 	// compute pointer arithmetic
// 	for(i = 0; i < col; i++)
// 	{
// 		col_offset += NEXTPOW2(cSizes[i]); 
// 	}
// 	sum_offset = col * NEXTPOW2(dSize);
// 	// bitVec = pointer to cData to be processed by decomp kernel
// 	bitVec = cols + col_offset;
// 	// decompSize = pointer to the decompSize to be processed by this thread
// 	decompSize = decompSizes + col_offset;
// 	// sum = pointer to the sum array to be used in the decomp of this column
// 	sum = sums + sum_offset;

// 	// create grid and block according to cSize
// 	dim3 grid0(ceil(1.0 * NEXTPOW2(cSizes[col])/THREADSPERBLOCK), 1, 1);
// 	dim3 block0(THREADSPERBLOCK, 1, 1);

// 	// set-up decompSizes array
// 	makeDecompSizes<<<grid0, block0>>>(bitVec, cSize[col], decompSize);
// 	__syncthreads();
// 	// create startingPoints array by exclusive scan of decompSize array
// 	// monkey man way...
// 	ulong accumulator = 0;
// 	for(i = 0; i < cSize[col]; i++)
// 	{
// 		decompSize
// 	}

// // --- THIS IS THE CORRECT WAY ASSUMING WE CAN FIGURE OUT HOW TO USE A FOR LOOP RATHER THAN RECURRSION ---
// //	sweepKernel<<<grid0, block0>>>(decompSize, sum);
// //	__syncthreads();
	
// //	dim3 grid1(ceil(1.0 * 
// //	dim3 block1(THREADSPERBLOCK/numEles, 1, 1);
// //	sumKernel<<<
// // --- 




// // --- gonna have to break up decomp into a few different kernels --- 
// 	// 2) make StartingPoints (scan DecompSizes)
// 	// 3) create EndPoints and fill it with useful data
// 	// 4) make WordIndex (scan EndPoints)
// 	// 5) fill dData (using WordIndex).
// }


// /*
//  * makeDecompSizes
//  * 
//  * Sets decompSize for a given bitVector. Has to init the entire decompSize to 0 
//  * so that the exlcusive scan works correctly. 
//  */
// __global__ void makeDecompSizes(ulong * cData, ulong cSize, ulong * decompSize)
// { 
// 	ulong tid = threadIdx.x + blockDim.x * THREADSPERBLOCK;	
// 	// this should never happen. Prevents writing over another columns decompSize array
// 	if (tid > NEXTPOW2(cSize)) { return; } 
// 	// init entire decompSize to 0
// 	decompSize[tid] = 0;
// 	// bounds checking
// 	if (tid >= cSize) { return; }
// 	// incrimenting decompSize array by one when cData[i] is literal atom
// 	if (cData[tid] >> 63 == 0) { decompSize[tid] = 1; }
// 	// incrimenting decompSize by len defined in cData[i] when fill atom 
// 	else { decompSize[tid] = (cData[tid] << 2) >> 2; }
// }

// /*
//  * sweepKernel
//  * Performs an exclusive scan on the data on the d_output
//  * array. In addition, one thread in each block will set an 
//  * element in the sum array to the value that needs to be
//  * added to the elements in the next section of d_output
//  * to complete the scan.
//  *
//  * @param - d_output points to an array in the global memory
//  *          that holds the input and will be modified to hold
//  *          the output
//  * @param - sum points to an array to hold the value to be 
//  *          added to the section handled by blockIdx.x + 1
//  *          in order to complete the scan
// */
// __global__ void sweepKernel(int * d_output, int * sum)
// {
//    __syncthreads();
//    int tid = threadIdx.x;
//    int blkD = blockDim.x;
//    int blkI = blockIdx.x;

//    //d_input points to the section of the input to be
//    //handled by this block
//    int * d_input = d_output + blkI * blkD;
//    __shared__ int shInput[THREADSPERBLOCK];

//    //initialize the value in the sum array
//    if (tid == (blkD >> 2) - 1)
//    {
//       sum[blkI] = d_input[blkD - 1];
//    }

//    //all threads participate in loading a
//    //value into the shared memory
//    shInput[tid] = d_input[tid];

//    __syncthreads();
//    int thid0Index = 0;
//    int index;
//    for (int i = 1; i < blkD; i<<=1)
//    {
//       thid0Index = thid0Index + i; 
//       index = thid0Index + tid * 2 * i;
//       if (index < blockDim.x) 
//       {
//          shInput[index] += shInput[index-i];
//       }
//       __syncthreads();
//    }
  
//    //set the last element in the section to 0 
//    //before the next sweep
//    if (tid == (blkD >> 2) - 1) shInput[blkD - 1] = 0;
//    __syncthreads();  
//    int i, j, topIndex, botIndex, tmp;
//    for (j=1, i = blkD >> 1; i >= 1; i >>= 1, j <<= 1)
//    {
//       //first iteration thread 0 is active
//       //second iteration threads 0, 1 are active
//       //third iteration threads 0, 1, 2, 4
//       if (tid < j)
//       {
//          topIndex = (tid + 1) * 2 * i - 1;
//          botIndex = topIndex - i;
//          tmp = shInput[botIndex];
//          shInput[botIndex] = shInput[topIndex];
//          shInput[topIndex] += tmp;
//       }
//       __syncthreads();
//    }
//    d_input[tid] = shInput[tid];
//    //update sum using last element in the block
//    if (tid == (blkD >> 2) - 1) sum[blkI] += shInput[blkD - 1];

//    __syncthreads();
// }

// /*
//  * sumKernel
//  * Adds elements in sum to the elements in the d_output array.
//  * The elements in the d_output array are sectioned into chunks
//  * of size THREADSPERBLOCK.  sum[0] is added to the first chunk.
//  * sum[1] added to the second block, etc.  The work is partitioned
//  * among the threads in the block using cyclic partitioning.  Each thread
//  * computes numElements results.
//  * @param - sum points to an array of values to use to update
//  *          d_output
//  * @param - d_output points to the array of partially scanned
//  *          values
// */
// __global__ void sumKernel(int * sum, int * d_output, int numElements)
// {
//    for(int j = 0; j < numElements; j++){
//       d_output[ j*blockDim.x + threadIdx.x + blockDim.x * blockIdx.x * numElements] += sum[blockIdx.x];
//    }
// }

// /* 
//  * gpuPrintVec
//  * Prints the contents a vector that is in the GPU memory, 10 elements
//  * per line.  This can be used for debugging.
// */
// __device__ void gpuPrintVec(const char * label, int * vector, int length)
// {
//     if (threadIdx.x == 0) 
//     {
//         int i;
//         printf("%s", label);
//         for (i = 0; i < length; i++)
//         {
//             if ((i % 20) == 0) printf("\n%4d: ", i);
//             printf("%3d ", vector[i]);
//         }
//         printf("\n");
//     }
// }



// REDO...

// Functions called from CPU
static __global__ void d_decompressionKernel(unsigned long* compData, unsigned long* decompData, int* dataNeeded, int* dataNeededPrefix);
static __global__ void d_COAKernel(unsigned long* decompData, int numOfRows, int numOfCols);

// Functions called from GPU
static __device__ void d_decompressionHelperKernel(unsigned long* cData, unsigned long cSize, int cDecompSize, int start, unsigned long* output);
static __device__ void d_ExclusiveScan(unsigned long* decompData, unsigned long* output, int decompDataSize);

// Debugging Function
static __device__ void printUlongArray(unsigned long* array, int startIdx, int length);


long unsigned* organizeData(unsigned long** compData, int numOfCols, int numOfRows, int &blocksNeeded){
	// CPU to change how the data is given and make our life easier...
	// Currently, compData[i] contains two parts. We wish to merge them and make everything easier...
	// We will still give the guarantee that each chunk is numOfRows+1 away from the next

	int ratio = numOfRows/THREADSPERBLOCK;
	if(!ratio) ratio++;

	long unsigned* output = (long unsigned*) Malloc(numOfCols*(numOfRows+1) * sizeof(unsigned long));

	int totalSoFar = 0;
	int lastWrittenTo = 1;
	int lastChunkUpdate = 0;
	int chunksCompleted = 0;
	for (int i = 0; i < numOfCols * (numOfRows+1); i++) {
		output[i] = 0;
	}
	for(int i = 0; i < numOfCols; i++){
		// if we can write data to this chunk still...
		if(totalSoFar + compData[i][0] <= THREADSPERBLOCK){
			for(int j = 1; j < 1+compData[i][0]; j++){
				output[lastWrittenTo] = compData[i][j];
				lastWrittenTo++;
			}
			output[lastChunkUpdate] += compData[i][0];
		}else if(compData[i][0] > THREADSPERBLOCK){
			// If some section is big, then we need to break it down...
			
			// Set up... 
			// The previous entries could've ended anywhere, so we will just try to ignore them...
			
			// The Next Index is at:
			if(output[lastChunkUpdate] != 0){
				chunksCompleted++;
				lastChunkUpdate = chunksCompleted * (THREADSPERBLOCK + 1);
				lastWrittenTo = lastChunkUpdate + 1;
			}
			int completeChunks = compData[i][0]/THREADSPERBLOCK;
			for(int j = 0; j < completeChunks; j++){
				output[lastChunkUpdate] = THREADSPERBLOCK;
				for(int k = 0; k < THREADSPERBLOCK; k++){
					output[lastWrittenTo] = compData[i][j * THREADSPERBLOCK + k + 1];
					lastWrittenTo++;
				}
				lastChunkUpdate = lastWrittenTo;
				lastWrittenTo++;
			}
			chunksCompleted += completeChunks;

			int leftOver = compData[i][0] % THREADSPERBLOCK;
			output[lastChunkUpdate] = leftOver;
			for(int k = 0; k < leftOver; k++){
				output[lastWrittenTo] = compData[i][completeChunks * THREADSPERBLOCK + k + 1];
				lastWrittenTo++;
			}
		}else{
			// The addition of this column will bring the number to something scuffed...
			// We will redo this as if we started at a new block...
			totalSoFar = 0;
			chunksCompleted++;
			lastChunkUpdate = chunksCompleted * (THREADSPERBLOCK + 1);
			lastWrittenTo = lastChunkUpdate++;
			i--;
		}
	}

	blocksNeeded = chunksCompleted+1;

	return output;
}

int** getDataSize(unsigned long* compData, int numOfChunks){
	int* ansPrefix = (int *) Malloc(sizeof(int) * numOfChunks);
	int* ans = (int *) Malloc(sizeof(int) * numOfChunks);
	for(int i = 0; i < numOfChunks; i++){
		int sizeOfChunk = compData[i * (THREADSPERBLOCK + 1)];
		ans[i] = 0;
		ansPrefix[i] = 0;
		for(int j = 0; j < sizeOfChunk; j++){
			unsigned long t = compData[i * (THREADSPERBLOCK + 1) + 1 + j];
			if(t >> 63){
				int sizeOfFill = (t << 2) >> 2;
				ans[i] += sizeOfFill;
			}else{
				ans[i]++;
			}
		}
		if(i){
			ansPrefix[i] = ansPrefix[i-1] + ans[i-1];
		}
	}

	int** ret = (int **) Malloc(sizeof(int *) * 2);
	ret[0] = ans;
	ret[1] = ansPrefix;
	return ret;
}

// method that is called by CPU
// for now do decomp...
// the numOfCols and Rows refer to the decompressed version of the data...
// numOfCols is the same for compressed and decompressed.
float d_rangequery(unsigned long** compData, unsigned long* result, int numOfCols, int numOfRows){
	// For now, this method will take the result of called compress on some data
	// the format of compData is:
	// compData[0] = length of data
	// compData[1] is actual results...

	// printf("got here tho!\n");
	float gpuMsecTime;
	cudaEvent_t start_gpu, stop_gpu;

	// printf("rows %d cols %d\n", numOfRows, numOfCols);

	int blocksNeeded = 0;

	// We will flatten the data, but to every column starts at i*numOfRows+1
	unsigned long* h_compData = organizeData(compData, numOfCols, numOfRows, blocksNeeded);
	int** h_dataNeeded = getDataSize(h_compData, blocksNeeded);
	// printf("done creating events\n");
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

	printf("organizing complete\n");
	// Device Variables
	// We will need to rework compData from 2D to be 1D
	unsigned long* d_compData;
	int* d_dataNeeded;
	int* d_dataNeededPrefix;
	unsigned long* d_decompData;

	// the size of the vector is now however much it takes for one row times all the columns
	CHECK(cudaMalloc((void **) &d_compData, sizeof(unsigned long) * blocksNeeded * (THREADSPERBLOCK + 1)));
	// Similar argument for the decompressed version of the data...
	CHECK(cudaMalloc((void **) &d_decompData, sizeof(unsigned long) * numOfRows * numOfCols));

	CHECK(cudaMalloc((void **) &d_dataNeeded, sizeof(int) * blocksNeeded));
	CHECK(cudaMalloc((void **) &d_dataNeededPrefix, sizeof(int) * blocksNeeded));


	CHECK(cudaMemcpy(d_dataNeededPrefix, h_dataNeeded[1], sizeof(int) * blocksNeeded, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_dataNeeded, h_dataNeeded[0], sizeof(int) * blocksNeeded, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_compData, h_compData, sizeof(unsigned long) * blocksNeeded * (THREADSPERBLOCK + 1), cudaMemcpyHostToDevice));
	// We know that the data is a multiple of 63.
	// We know how many threads are per block, so we need to figure out the number of blocks needed..
	dim3 grid(blocksNeeded, 1, 1);
	// Each block will be 63 threads since data is a multiple of 63
	dim3 block(THREADSPERBLOCK, 1, 1);


	for(int i = 0; i < blocksNeeded; i++){
		printf("CPU: Size of Chunk: %d is %lu\n", i, h_compData[i + THREADSPERBLOCK*i]);
		printf("CPU: Prefix and Not: %d is %d\n", h_dataNeeded[0][i], h_dataNeeded[1][i]);
	}
	// unsigned long compSizeCol1 = h_compData[0];
	// printf("CPU SAYS | compSizeCol1 : %lu\n\n", compSizeCol1);
	// unsigned long compSizeCol2 = h_compData[THREADSPERBLOCK+1];
	// printf("CPU SAYS | compSizeCol2 : %lu\n\n", compSizeCol2);
	// unsigned long compSizeCol3 = h_compData[THREADSPERBLOCK+THREADSPERBLOCK+2];
	// printf("CPU SAYS | compSizeCol3 : %lu\n\n", compSizeCol3);
	// unsigned long compSizeCol4 = h_compData[3];
	/*for (int i = 0; i < compSizeCol1 + 1; i++) {
		printf("%lu\n", h_compData[i+4]);
	}*/

	d_decompressionKernel<<<grid, block>>>(d_compData, d_decompData, d_dataNeeded, d_dataNeededPrefix);

	cudaDeviceSynchronize();

// --- THIS FAILING
	// // We can now do stuff on d_decompData...
	// d_COAKernel<<<grid, block>>>(d_decompData, numOfRows, numOfCols);
	// printf("thiccc got here 3\n");
	// cudaDeviceSynchronize();q

	CHECK(cudaMemcpy(result, d_decompData, sizeof(unsigned long)*numOfRows*numOfCols, cudaMemcpyDeviceToHost));
// --- END FAIL


	CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
	// We are not timing for now...
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

__global__
void d_decompressionKernel(unsigned long* compData, unsigned long* decompData, int* dataNeeded, int* dataNeededPrefix){
	//__syncthreads();

	// Each block is responsible for 1 chunk of THREADSPERBLOCK data...
	// 0 1 -- 1008 | 1009 1010 - ... | ...
	// So, cSize is in blockIdx.x * THREADSPERBLOCK
	d_decompressionHelperKernel(compData, compData[blockIdx.x * (THREADSPERBLOCK + 1)], dataNeeded[blockIdx.x], dataNeededPrefix[blockIdx.x] + blockIdx.x + 1, decompData);
	
	// Index of chunk i is at (numOfRows+1) * i
	// Size of chunk i is at compData[(numOfRows+1) * i]

}

__device__
void d_decompressionHelperKernel(unsigned long* cData, unsigned long cSize, int cDecompSize, int start, long unsigned* decompDataOut){
	// For reference, cData is the entire array of compressed data
	// cSize is how many elements we will process
	// cDecompSize is the output number of elements
	// start is the start point	

	// Shared memory needed for all operations...
	__shared__ unsigned long decompDataShared[THREADSPERBLOCK];
	__shared__ unsigned long startingPoints[THREADSPERBLOCK];
	__shared__ unsigned long endPoints[THREADSPERBLOCK];
	__shared__ unsigned long wordIndex[THREADSPERBLOCK];
	int id = threadIdx.x;

	if(threadIdx.x == 0) {
		printf("Block : %d\n", blockIdx.x);
		printf("cSize : %lu\n", cSize);
		printf("cDecompSize : %d\n", cDecompSize);
		printf("start : %d\n", start);
	}

	if(id < cSize){
		int put = start + id;
		unsigned long dataToPut = cData[put];
		if(dataToPut >> 63){
			decompDataShared[id] = (dataToPut << 2) >> 2; 
		}else{
			decompDataShared[id] = 1;
		}
	}else{
		decompDataShared[id] = 0;
	}

	__syncthreads();

	if(threadIdx.x == 0)
		d_ExclusiveScan(decompDataShared, startingPoints, cSize);

	__syncthreads();
	
	if(id < cSize && id > 0){
		endPoints[startingPoints[id] - 1] = 1;
	}else{
		endPoints[id] = 0;
	}	

	__syncthreads();
	if(threadIdx.x == 0)
		d_ExclusiveScan(endPoints, wordIndex, cDecompSize);
	__syncthreads();

	if(id < cDecompSize){
		unsigned long tempWord = cData[wordIndex[id] + start];
		int startForOut = start - blockIdx.x - 1;
		if(tempWord >> 63){
			uint whatFill = (tempWord << 1) >> 63;
			if(whatFill){
				decompDataOut[startForOut + threadIdx.x] = 0x7fffffffffffffff;
			}else{
				decompDataOut[startForOut + threadIdx.x] = 0;
			}
		}else{
			if(4096 == startForOut + threadIdx.x){
				printf("Sad %lx %d %lu\n",tempWord, startForOut + id, wordIndex[id] + start);
			}

			decompDataOut[startForOut + threadIdx.x] = tempWord;
		}
	}

	// // Need to redo this whole method to be in parallel...
	// if(threadIdx.x == 0){
	// 	// for(int i = 0; i < cSize; i++){
	// 	// 	// For each piece of data, how much will it need?
	// 	// 	if(cData[start+i] >> 63){
	// 	// 		// Clear the last two bits
	// 	// 		decompData[i] = (cData[start+i] << 2) >> 2;
	// 	// 	}else{
	// 	// 		decompData[i] = 1;
	// 	// 	}
	// 	// }
	// 	// int* startingPoints;
	// 	// int Q[10];
	// 	// startingPoints = Q;
	// 	// d_ExclusiveScan(decompData, startingPoints, cDecompSize);
	// 	// long unsigned* endPoints;
	// 	// long unsigned E[10];
	// 	// endPoints = E;
	// 	// for(int i = 1; i < cSize; i++){
	// 	// 	endPoints[startingPoints[i] - 1] = 1;
	// 	// }
	// 	// int* wordIndex;
	// 	// int K[10];
	// 	// wordIndex = K;
	// 	// d_ExclusiveScan(endPoints, wordIndex, cDecompSize);
	// 	for(int i = 0; i < cDecompSize; i++){
	// 		unsigned long tempWord = cData[wordIndex[i] + start];
	// 		if(tempWord >> 63){
	// 			uint whatFill = (tempWord << 1) >> 63;
	// 			if(whatFill){
	// 				decompData[cDecompSize*blockIdx.x + i] = 0x7fffffffffffffff;
	// 			}else{
	// 				decompData[cDecompSize*blockIdx.x + i] = 0;
	// 			}
	// 		}else{
	// 			decompData[cDecompSize*blockIdx.x + i] = tempWord;
	// 		}
	// 		//printf("Thread: %d, in block: %d, put %lx in %d", threadIdx.x, blockIdx.x, decompData[i], i);
	// 	}
	// }

	
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



