/*
  Brady Adcock
  Abdel Issa 

  Implementing cool paper on range query
*/

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "d_rangequery.h"
#include "config.h"  //defines THREADSPERBLOCK
#include "wrappers.h"

static __global__ void decomp(float*, float, float);
static __global__ void decompress(float*, float*, float*, float*, int);

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
void d_decompress (float * cols, float * cSizes, float * dData, float dSize, int numCols)
{
	float * R; 
	float * d_cols; // 1d array representing the 2d array of compressed bins
	float * d_cSizes;  
	float * d_dData; // 1d array representing the 2d array of decompressed bins
	float totalSize =  0;
	int i;

	// determine total size of cols (cData)
	for(i = 0; i < numCols; i++) 
	{
		totalSize += cSize[i];
	}
	totalSize *= sizeof(float); 

	// malloc cData && dData
	CHECK(cudaMalloc((void **) &d_cols, totalSize));
	CHECK(cudaMalloc((void **) &d_dData, dSize * sizeof(float)));
	// malloc cSizes
	CHECK(cudaMalloc((void **) &d_cSizes, numCols * sizeof(float)));
	// malloc R (same as dSize)
	CHECK(cudaMalloc((void **) &R, dSize * sizeof(float)));

	// copy cData over to device
	CHECK(cudaMemCopy(d_cols, cols, totalSize, cudaMemcpyHostToDevice));
	// copy cSizes over to device
	CHECK(cudaMemCopy(d_cSizes, cSizes, numCols * sizeof(float), cudaMemcpyHostToDevice));

	// make grid && block 
	dim3 grid(1, 1, 1); 		// only need a few threads because this 
	dim3 block(numCols, 1, 1);	// kernel will launch many more

	// launch kernel decompress
	decompress<<<grid, block>>>(d_cols, d_cSizes, d_dData, dSize, numCols);

	// copy decompressed data back from device
	CHECK(cudaMemCopy(dData, d_dData, dSize * sizeof(float)));

	// cudaFree everything on device
	CHECK(cudaFree(d_cols);
	CHECK(cudaFree(d_cSizes);
	CHECK(cudaFree(d_dData);
}

/*
 Given pointer to array of WAH compressed bitvectors, decompress them
 by launching decomp kernels.

 params,
	cols:	array of bitvectors with WAH 64 bit encoding
*/
__global__ void decompress (float * cols, float * cSizes, float * dData, float dSize, int numCols) 
{
	int col = threadIdx.x;
	float col_offset = 0;		
	float * bitVec;
	int i;

	// compute pointer arithmetic
	for(i = 0; i < col; i++)
	{
		col_offset += cSize[i]; 
	}
	// bitVec = pointer to cData to be processed by decomp kernel
	bitVec = cols + offset;

	// create grid and block according to cSize
	dim3 gird = (ceil(1.0 * cSize[col]/THREADSPERBLOCK, 1, 1));
	dim3 block = (THREADSPERBLOCK, 1, 1);

	// launch decomp kernel
	decomp<<<grid, block>>>(bitVec, cSize[col], dSize); 	
}

/*
 Given a bitvector representing a single bin, decompress its
 data and return it.

 params,
	cData:	compressed WAH vector
	cSize:	number of 64 bit words CData represents
	dSize:	number of 64 bit words in orignal (decompressed) data 
*/
__global__ void decomp(float * cData, float cSize; float dSize)
{
	// debugging...
	printf("cData[0]: %ul, cSize: %d, dSize: %ul\n", cData[0], cSize, dSize);

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
	
