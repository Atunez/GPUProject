/*
  Brady Adcock
  Abdel Issa 

  Implementing cool paper on range query
*/

#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "d_rangequery.h"
#include "config.h"  //defines THREADSPERBLOCK
#include "wrappers.h"

// prototype for kernels
static __global__ void decomp(ulong*, ulong, ulong);
static __global__ void decompress(ulong*, ulong*, ulong*, ulong, int);

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
	ulong * d_cols; // 1d array representing the 2d array of compressed bins
	ulong * d_cSizes;  
	ulong * d_dData; // 1d array representing the 2d array of decompressed bins
	ulong totalSize =  0;
	int i;

	// determine total size of cols (cData)
	for(i = 0; i < numCols; i++) 
	{
		totalSize += cSizes[i];
	}
	totalSize *= sizeof(ulong); 

	// malloc cData && dData
	CHECK(cudaMalloc((void **) &d_cols, totalSize));
	CHECK(cudaMalloc((void **) &d_dData, dSize * sizeof(ulong)));
	// malloc cSizes
	CHECK(cudaMalloc((void **) &d_cSizes, numCols * sizeof(ulong)));
	// malloc R (same as dSize)
	CHECK(cudaMalloc((void **) &R, dSize * sizeof(ulong)));

	// copy cData over to device
	CHECK(cudaMemcpy(d_cols, cols, totalSize, cudaMemcpyHostToDevice));
	// copy cSizes over to device
	CHECK(cudaMemcpy(d_cSizes, cSizes, numCols * sizeof(ulong), cudaMemcpyHostToDevice));

	// make grid && block 
	dim3 grid(1, 1, 1); 		// only need a few threads because this 
	dim3 block(numCols, 1, 1);	// kernel will launch many more

	// launch kernel decompress
	decompress<<<grid, block>>>(d_cols, d_cSizes, d_dData, dSize, numCols);

	// copy decompressed data back from device
	CHECK(cudaMemcpy(dData, d_dData, dSize * sizeof(ulong), cudaMemcpyDeviceToHost));

	// cudaFree everything on device
	CHECK(cudaFree(d_cols));
	CHECK(cudaFree(d_cSizes));
	CHECK(cudaFree(d_dData));
}

/*
 Given pointer to array of WAH compressed bitvectors, decompress them
 by launching decomp kernels.

 params,
	cols:	array of bitvectors with WAH 64 bit encoding
*/
__global__ void decompress (ulong * cols, ulong * cSizes, ulong * dData, ulong dSize, int numCols) 
{
	int col = threadIdx.x;
	ulong col_offset = 0;		
	ulong * bitVec;
	int i;

	// compute pointer arithmetic
	for(i = 0; i < col; i++)
	{
		col_offset += cSizes[i]; 
	}
	// bitVec = pointer to cData to be processed by decomp kernel
	bitVec = cols + col_offset;

	// create grid and block according to cSize
	dim3 grid(ceil(1.0 * cSizes[col]/THREADSPERBLOCK), 1, 1);
	dim3 block(THREADSPERBLOCK, 1, 1);

	// launch decomp kernel
	decomp<<<grid, block>>>(bitVec, cSizes[col], dSize); 	
}

/*
 Given a bitvector representing a single bin, decompress its
 data and return it.

 params,
	cData:	compressed WAH vector
	cSize:	number of 64 bit words CData represents
	dSize:	number of 64 bit words in orignal (decompressed) data 
*/
__global__ void decomp(ulong * cData, ulong cSize, ulong dSize)
{
	// debugging...
	printf("cData[0]: %lu, cSize: %lu, dSize: %lu\n", cData[0], cSize, dSize);
	
	// create index in cData
	uint cWordIndex = blockIdx.x * THREADSPERBLOCK + threadIdx.x;

	// bounds checking
	if (cWordIndex >= cSize) { return; }

	
	
//-------------------------

	// TODO: for each 64-bit WAH word in cData check word type
		// update DecompSize[index] with 1:lit or len:fill 
			// PARALLIZATION OPPORTUNITY

	// TODO: create startingPoints array using exclusive scan of DecompSize[]

	// TODO: create endPoints[dSize], init with 0's

	// TODO: for each 64-bit WAH word in cData add entry into endPoints
		// endPoints[startingPoints[i]-1] = 1
			// PARALLIZATION OPPORTUNITY

	// TODO: create wordIndex to store index of atom in cData that contains data
	// for the wordIndex[i] decompressed word
		// exclusive scan on endPoints
			// PARALLIZATION OPPORTUNITY

	// TODO: for each word in the decompressed data : decompress
		// grab tempWord, the cData word encoding this dData word
		// if tempWord is a literal then update dData[i] with tempWord
		// else fill dData[i] with all 0's or one 0 and sixtythree 1's
}
	
