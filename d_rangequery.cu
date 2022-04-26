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

static __global__ void decomp(float *, float, float);

/*
 Given an array of relevant cols (representing bins which
 classify the variable of interest) return resultant 
 bit vector R. 

 params,
 	cols:   array of bitvectors representing bins of variable 
 		being queried. 
 return,
	R:	bitvector representing rows who match range query.
*/
float * d_decompress (float ** cols) 
{
	// TODO: for each col launch kernel to decompress it
		// or the result to past results to create R

	// TODO: return R
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
	
