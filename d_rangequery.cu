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
__global__ void decompress (ulong *, ulong *, ulong *, ulong, int, ulong *, ulong *, ulong *);
static __global__ void makeDecompSizes(ulong*, ulong, ulong*);
static __global__ void sumKernel(ulong *, ulong *, ulong);
static __global__ void sweepKernel(ulong *, ulong *) ;
__device__ void gpuPrintVec(const char * label, ulong * vector, ulong length);

/*
 Given an array of relevant cols (representing bins which
 classify the variable of interest) return resultant 
 bit vector R. 

 params,
 	cols:   array of bitvectors representing bins of variable 
 		being queried. 2d array reprsented as 1d array because
		each element of the array is variable length.  
	cSizes: array desciribing the size of each compressed col
		in terms of number of 64 bit words.  
	dSizes: number of 64 bit words in the decompressed cols
 return,
	R:	bitvector representing rows who match range query.
*/
void d_decompress (ulong * cols, ulong * cSizes, ulong * dData, ulong dSize, int numCols)
{
	ulong * R; 
	ulong * d_cols; 	// 1d array representing the 2d array of compressed bins
	ulong * d_cSizes;  
	ulong * d_dData; 	// 1d array representing the 2d array of decompressed bins

	// following mem is only used in decomp
	ulong * d_decompSizes;  // becomes startingPoints after scan
	ulong * d_startingPoints;
	ulong * d_endPoints;    // becomes wordIndex after scan
	ulong * d_wordIndexes;
	ulong * d_sums;		// used for scan

	ulong cTotalSize = 0; // size used for cols, decompSize, startingPoints 2d arrays
	ulong dTotalSize = 0; // size used for endPoints and WordIndex 2d arrays
	int i;
	// determine total sizes based off of the num of cols, cData. 
	for(i = 0; i < numCols; i++) 
	{
		// have to ensure these sizes are powers of two for the sake of exclusive scan efficeny 
		cTotalSize += NEXTPOW2(cSizes[i]);
		dTotalSize += NEXTPOW2(dSize);
	}
	cTotalSize *= sizeof(ulong);
	dTotalSize *= sizeof(ulong);

	// malloc cData && dData
	CHECK(cudaMalloc((void **) &d_cols, cTotalSize));
	CHECK(cudaMalloc((void **) &d_dData, dSize * sizeof(ulong)));
	// malloc cSizes
	CHECK(cudaMalloc((void **) &d_cSizes, numCols * sizeof(ulong)));
	// malloc decompSizes arry for decomp
	CHECK(cudaMalloc((void **) &d_decompSizes, cTotalSize));
	CHECK(cudaMalloc((void **) &d_startingPoints, cTotalSize));
	// malloc endPoints array for decomp
	CHECK(cudaMalloc((void **) &d_endPoints, dSize * sizeof(char)));
	CHECK(cudaMalloc((void **) &d_wordIndexes, dTotalSize));
	// malloc sum array, used for the exclusive scans in decomp.
	// worst case it is of size dSize/THREADSPERBLOCK for the exlusive scan on endPoints
	CHECK(cudaMalloc((void **) &d_sums, NEXTPOW2(dSize) * numCols * sizeof(ulong)/THREADSPERBLOCK));
	// malloc R (same as dSize)
	CHECK(cudaMalloc((void **) &R, dTotalSize));	

	// copy cData over to device
	CHECK(cudaMemcpy(d_cols, cols, totalSize, cudaMemcpyHostToDevice));
	// copy cSizes over to device
	CHECK(cudaMemcpy(d_cSizes, cSizes, numCols * sizeof(ulong), cudaMemcpyHostToDevice));

	// make grid && block 
	dim3 grid(1, 1, 1); 		// only need a few threads because this 
	dim3 block(numCols, 1, 1);	// kernel will launch many more

	// launch kernel decompress
	decompress<<<grid, block>>>(d_cols, d_cSizes, d_dData, dSize, numCols, 
				    d_decompSizes, d_startingPoints, d_endPoints, d_wordIndexes, d_sums);

	// copy decompressed data back from device
	CHECK(cudaMemcpy(dData, d_dData, dSize * sizeof(ulong), cudaMemcpyDeviceToHost));

	// cudaFree everything on device
	CHECK(cudaFree(d_cols));
	CHECK(cudaFree(d_cSizes));
	CHECK(cudaFree(d_dData));
	CHECK(cudaFree(d_decompSizes));
	CHECK(cudaFree(d_endPoints));
	CHECK(cudaFree(d_sums));
}

/*
 Given pointer to array of WAH bitvectors, decompress them
 by launching decomp kernels.

 params,	
	cols:	array of bitvectors with WAH 64 bit encoding
*/
__global__ void decompress (ulong * cols, ulong * cSizes, ulong * dData, ulong dSize, int numCols,
			    ulong * decompSizes, ulong * startingPoints, ulong * endPoints, 
			    ulong* wordIndexes, ulong * sums) 
{
	int col = threadIdx.x;
	ulong col_offset = 0;	
	ulong sum_offset = 0;	
	ulong * bitVec, decompSize, sum;
	int i;

	// compute pointer arithmetic
	for(i = 0; i < col; i++)
	{
		col_offset += NEXTPOW2(cSizes[i]); 
	}
	sum_offset = col * NEXTPOW2(dSize);
	// bitVec = pointer to cData to be processed by decomp kernel
	bitVec = cols + col_offset;
	// decompSize = pointer to the decompSize to be processed by this thread
	decompSize = decompSizes + col_offset;
	// sum = pointer to the sum array to be used in the decomp of this column
	sum = sums + sum_offset;

	// create grid and block according to cSize
	dim3 grid0(ceil(1.0 * NEXTPOW2(cSizes[col])/THREADSPERBLOCK), 1, 1);
	dim3 block0(THREADSPERBLOCK, 1, 1);

	// set-up decompSizes array
	makeDecompSizes<<<grid0, block0>>>(bitVec, cSize[col], decompSize);
	__syncthreads();
	// create startingPoints array by exclusive scan of decompSize array
	// monkey man way...
	ulong accumulator = 0;
	for(i = 0; i < cSize[col]; i++)
	{
		decompSize
	}

// --- THIS IS THE CORRECT WAY ASSUMING WE CAN FIGURE OUT HOW TO USE A FOR LOOP RATHER THAN RECURRSION ---
//	sweepKernel<<<grid0, block0>>>(decompSize, sum);
//	__syncthreads();
	
//	dim3 grid1(ceil(1.0 * 
//	dim3 block1(THREADSPERBLOCK/numEles, 1, 1);
//	sumKernel<<<
// --- 




// --- gonna have to break up decomp into a few different kernels --- 
	// 2) make StartingPoints (scan DecompSizes)
	// 3) create EndPoints and fill it with useful data
	// 4) make WordIndex (scan EndPoints)
	// 5) fill dData (using WordIndex).
}


/*
 * makeDecompSizes
 * 
 * Sets decompSize for a given bitVector. Has to init the entire decompSize to 0 
 * so that the exlcusive scan works correctly. 
 */
__global__ void makeDecompSizes(ulong * cData, ulong cSize, ulong * decompSize)
{ 
	ulong tid = threadIdx.x + blockDim.x * THREADSPERBLOCK;	
	// this should never happen. Prevents writing over another columns decompSize array
	if (tid > NEXTPOW2(cSize)) { return; } 
	// init entire decompSize to 0
	decompSize[tid] = 0;
	// bounds checking
	if (tid >= cSize) { return; }
	// incrimenting decompSize array by one when cData[i] is literal atom
	if (cData[tid] >> 63 == 0) { decompSize[tid] = 1; }
	// incrimenting decompSize by len defined in cData[i] when fill atom 
	else { decompSize[tid] = (cData[tid] << 2) >> 2; }
}

/*
 * sweepKernel
 * Performs an exclusive scan on the data on the d_output
 * array. In addition, one thread in each block will set an 
 * element in the sum array to the value that needs to be
 * added to the elements in the next section of d_output
 * to complete the scan.
 *
 * @param - d_output points to an array in the global memory
 *          that holds the input and will be modified to hold
 *          the output
 * @param - sum points to an array to hold the value to be 
 *          added to the section handled by blockIdx.x + 1
 *          in order to complete the scan
*/
__global__ void sweepKernel(int * d_output, int * sum)
{
   __syncthreads();
   int tid = threadIdx.x;
   int blkD = blockDim.x;
   int blkI = blockIdx.x;

   //d_input points to the section of the input to be
   //handled by this block
   int * d_input = d_output + blkI * blkD;
   __shared__ int shInput[THREADSPERBLOCK];

   //initialize the value in the sum array
   if (tid == (blkD >> 2) - 1)
   {
      sum[blkI] = d_input[blkD - 1];
   }

   //all threads participate in loading a
   //value into the shared memory
   shInput[tid] = d_input[tid];

   __syncthreads();
   int thid0Index = 0;
   int index;
   for (int i = 1; i < blkD; i<<=1)
   {
      thid0Index = thid0Index + i; 
      index = thid0Index + tid * 2 * i;
      if (index < blockDim.x) 
      {
         shInput[index] += shInput[index-i];
      }
      __syncthreads();
   }
  
   //set the last element in the section to 0 
   //before the next sweep
   if (tid == (blkD >> 2) - 1) shInput[blkD - 1] = 0;
   __syncthreads();  
   int i, j, topIndex, botIndex, tmp;
   for (j=1, i = blkD >> 1; i >= 1; i >>= 1, j <<= 1)
   {
      //first iteration thread 0 is active
      //second iteration threads 0, 1 are active
      //third iteration threads 0, 1, 2, 4
      if (tid < j)
      {
         topIndex = (tid + 1) * 2 * i - 1;
         botIndex = topIndex - i;
         tmp = shInput[botIndex];
         shInput[botIndex] = shInput[topIndex];
         shInput[topIndex] += tmp;
      }
      __syncthreads();
   }
   d_input[tid] = shInput[tid];
   //update sum using last element in the block
   if (tid == (blkD >> 2) - 1) sum[blkI] += shInput[blkD - 1];

   __syncthreads();
}

/*
 * sumKernel
 * Adds elements in sum to the elements in the d_output array.
 * The elements in the d_output array are sectioned into chunks
 * of size THREADSPERBLOCK.  sum[0] is added to the first chunk.
 * sum[1] added to the second block, etc.  The work is partitioned
 * among the threads in the block using cyclic partitioning.  Each thread
 * computes numElements results.
 * @param - sum points to an array of values to use to update
 *          d_output
 * @param - d_output points to the array of partially scanned
 *          values
*/
__global__ void sumKernel(int * sum, int * d_output, int numElements)
{
   for(int j = 0; j < numElements; j++){
      d_output[ j*blockDim.x + threadIdx.x + blockDim.x * blockIdx.x * numElements] += sum[blockIdx.x];
   }
}

/* 
 * gpuPrintVec
 * Prints the contents a vector that is in the GPU memory, 10 elements
 * per line.  This can be used for debugging.
*/
__device__ void gpuPrintVec(const char * label, int * vector, int length)
{
    if (threadIdx.x == 0) 
    {
        int i;
        printf("%s", label);
        for (i = 0; i < length; i++)
        {
            if ((i % 20) == 0) printf("\n%4d: ", i);
            printf("%3d ", vector[i]);
        }
        printf("\n");
    }
}

