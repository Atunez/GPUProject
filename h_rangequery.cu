#include "h_rangequery.h"
#include "config.h"
#include "wrappers.h"
#include <stdlib.h>
#include <stdio.h>


long unsigned* compressNew(long unsigned* bitvector, int number, int &trackerOut)
{
    // worst case no copmression
    long unsigned* answer = (long unsigned*) malloc(sizeof(long unsigned) * number); 
    int tracker = 0;
    //answer[0] = 0;
    int i;
    for(i = 0; i < number; i++){
        // we should do the compression here...
        // all the numbers in bitvector are 63 bit numbers..
        int isfill = 1;
        int j = 0;
        long unsigned start = 0;
        for(; j < 63; j++){
            long unsigned toshift = bitvector[i];
            long unsigned consider = (toshift >> j) & 1;
            if(j == 0){
                start = consider;
                continue;
            }
            if(consider != start){
                isfill = 0;
                break;
            }
        }
        
        
        // tracker should keep track of where we are going to insert next 
        
        if(isfill){
            // fill atom
            // has the form X|Y|Z, where X=1, Y= 0 or 1 and Z is count
            long unsigned lastAnswer;
            // if we are at the start then we can't go anywhere...
            if(i == 0){
                lastAnswer = 0x0;
            }else{
                // but otherwise, check the previous slot.
                // if i = 2, then answer[0] = 2, so this would look at 1
                lastAnswer = answer[tracker-1];
            }
            if(lastAnswer >> 63){
                // if the last was a fill, then we need to do more work :(
                // if they have the same fill then just increment...
                if(!((lastAnswer >> 62) & 1 ^ (bitvector[i] >> 62) & 1)){
                    // we will assume that we don't actually overflow here...
                    answer[tracker-1]++;
                    // we don't add anything new
                    continue;
                }else{
                    // fresh insert....
                    long unsigned toput = 0x8000000000000001;
                    toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                    answer[tracker] = toput;
                }
                
            }else{
                long unsigned toput = 0x8000000000000001;
                toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                answer[tracker] = toput;
            }

        }else{
            // literal atom
            // and is for safety, shouldn't be needed....
            answer[tracker] = bitvector[i] & 0x7fffffffffffffff;
        }
        //printf("At i: %d, last thing is: %lx\n", i, answer[answer[0]]);
        tracker++;
    }
    trackerOut = tracker;
    return answer;
}


long unsigned** compressFullColumn(long unsigned* bitvector, int sizeOfVector){
    // Given a vector of size n divide it into compressed 1D version....
    int newCols = ceil( (1.0 * sizeOfVector) / THREADSPERBLOCK);
    // This is slow :^)
    // We move though... So no judging zone...
    // At worse, we do no compression...
    // We will linearize the data here... but it goes from 1D -> 2D -> 1D
    long unsigned * cData = (long unsigned *) Malloc(sizeof(long unsigned) * newCols * THREADSPERBLOCK);
    long unsigned * cSize = (long unsigned *) Malloc(sizeof(long unsigned) * newCols);
    long unsigned * cSizeScan = (long unsigned *) Malloc(sizeof(long unsigned) * newCols);
    long unsigned cChunkSize = newCols; // This will get optimized out but we move...

    // Data Holder...
    long unsigned * emptySize = (long unsigned *) Malloc(sizeof(long unsigned) * THREADSPERBLOCK);
    long unsigned ** cDataHolder = (long unsigned **) Malloc(sizeof(emptySize) * newCols);
    for(int i = 0; i < newCols; i++){
        int broke = 0;
        for(int j = 0; j < THREADSPERBLOCK; j++){
            if(i * THREADSPERBLOCK + j == sizeOfVector){
                broke = 1;
                break;
            }
            cDataHolder[i][j] = bitvector[i * THREADSPERBLOCK + j];
        }
        if(broke)
            break;
    }

    // Start on decompression....
    int completeCounter = 0;
    for(int i = 0; i < newCols; i++){
        int amountOfData = 0;
        int number = 0;
        if(i+1 == newCols){
            number = sizeOfVector;
        }else{
            sizeOfVector -= THREADSPERBLOCK;
            number = THREADSPERBLOCK;
        }
        // Compress a col segment...
        long unsigned * compressedCol = compressNew(cDataHolder[i], number, amountOfData);
        for(int j = 0; j < THREADSPERBLOCK; j++){
            // Put it in...
            cData[completeCounter] = compressedCol[j];
            completeCounter++;
        }
        cSize[i] = number;
        cSizeScan[i] = completeCounter;
    }

    long unsigned ** output = (long unsigned **) Malloc(4 * sizeof(long unsigned *));

    output[0] = cData;
    output[1] = cSize;
    output[2] = cSizeScan;
    output[3] = &cChunkSize;

    return output;
}

long unsigned ** compressMultipleRows(long unsigned** bitVectors, int sizeOfBitVectors, int numberOfBitVectors){
    // This will just call compressFullColumn numberOfBitVector times and flatten them and combine them....

    // Need to Malloc a lot of data....
    int ratio = ceil(sizeOfBitVectors/(1.0*THREADSPERBLOCK));

    // Each column will need ratio * THREADSPERBLOCK space... but then we need numOfBV of those...
    long unsigned * cData = (long unsigned *) Malloc(ratio * numberOfBitVectors * sizeof(long unsigned) * THREADSPERBLOCK);
    long unsigned * cSize = (long unsigned *) Malloc(ratio * sizeof(long unsigned) * numberOfBitVectors);
    long unsigned * cSizeScan = (long unsigned *) Malloc(ratio * sizeof(long unsigned) * numberOfBitVectors);
    long unsigned cChunkSize = 0;

    int counterForcData = 0;
    int counterForcSize = 0;
    for(int i = 0; i < numberOfBitVectors; i++){
        long unsigned ** compressOneVector = compressFullColumn(bitVectors[i], sizeOfBitVectors);
        for(int j = 0; j < compressOneVector[2][compressOneVector[3][0] - 1]; j++){
            // This loop goes through everything... 
            cData[counterForcData] = compressOneVector[0][j];
            if(j < compressOneVector[3][0]){
                cSize[counterForcSize] = compressOneVector[1][j];
                cSizeScan[counterForcSize] = compressOneVector[2][j];
                counterForcSize++;
            }
            counterForcData++;
        }
        if(cChunkSize == 0){
            cChunkSize = compressOneVector[3][0];
        }else{
            if(cChunkSize != compressOneVector[3][0]){
                printf("Inconsistent? \n");
                exit(0);
            }
        }
    }

    long unsigned ** output = (long unsigned **) Malloc(4 * sizeof(long unsigned *));

    output[0] = cData;
    output[1] = cSize;
    output[2] = cSizeScan;
    output[3] = &cChunkSize;

    return output; 
}

long unsigned* compress(long unsigned* bitvector, int number)
{
    // worst case no copmression
    long unsigned* answer = (long unsigned*) malloc(sizeof(long unsigned) * number); 
    answer[0] = 0;
    int i;
    for(i = 0; i < number; i++){
        // we should do the compression here...
        // all the numbers in bitvector are 63 bit numbers..
        int isfill = 1;
        int j = 0;
        long unsigned start = 0;
        for(; j < 63; j++){
            long unsigned toshift = bitvector[i];
            long unsigned consider = (toshift >> j) & 1;
            if(j == 0){
                start = consider;
                continue;
            }
            if(consider != start){
                isfill = 0;
                break;
            }
        }
        
        
        // answer[0] should keep track of where we are going to insert next 
        answer[0]++;
        if(isfill){
            // fill atom
            // has the form X|Y|Z, where X=1, Y= 0 or 1 and Z is count
            long unsigned lastAnswer;
            // if we are at the start then we can't go anywhere...
            if(i == 0){
                lastAnswer = 0x0;
            }else{
                // but otherwise, check the previous slot.
                // if i = 2, then answer[0] = 2, so this would look at 1
                lastAnswer = answer[answer[0]-1];
            }
            if(lastAnswer >> 63){
                // if the last was a fill, then we need to do more work :(
                // if they have the same fill then just increment...
                if(!((lastAnswer >> 62) & 1 ^ (bitvector[i] >> 62) & 1)){
                    // we will assume that we don't actually overflow here...
                    answer[answer[0]-1]++;
                    // we don't add anything new
                    answer[0]--;
                    continue;
                }else{
                    // fresh insert....
                    long unsigned toput = 0x8000000000000001;
                    toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                    answer[answer[0]] = toput;
                }
                
            }else{
                long unsigned toput = 0x8000000000000001;
                toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                answer[answer[0]] = toput;
            }

        }else{
            // literal atom
            // and is for safety, shouldn't be needed....
            answer[answer[0]] = bitvector[i] & 0x7fffffffffffffff;
        }
        //printf("At i: %d, last thing is: %lx\n", i, answer[answer[0]]);
    }

    return answer;
}


int inBin(long unsigned value, int* binsOfInterest, int lengthOfBins){
    int i;
    for(i = 0; i < lengthOfBins; i++){
        if(value/BINSIZE == binsOfInterest[i])
            return 1;
    }
    return 0;
}

// Given some startBin and endBin, find all the values that are in those bins...
void simpleQuery(int* binsOfInterest, long unsigned* vector, char ** labels, int numOfValues, int lengthOfBins){
    int i;
    for(i = 0; i < numOfValues; i++){
        if(inBin(vector[i], binsOfInterest, lengthOfBins)){
            printf("%s %d %d %d\n", labels[i], vector[i], i/63, (i % 63));
        }
    }
}

// // Do the math for two atoms
// // which decides the boolean operation...
// long unsigned atomMath(long unsigned op1, long unsigned op2, int which){
//     long unsigned out = 0;
//     if(op1 >> 63 && op2 >> 63){
//         out = 0x8000000000000000;
//         int value = 0;
//         if(which == 1){

//         }else{

//         }
//     }else if(!(op1 >> 63) && !(op2 >> 63)){

//     }else{
//         if(op2 >> 63){
//             long unsigned t = op2;
//             op2 = op1;
//             op1 = t;
//         }
//     }
//     return out;
// }

long unsigned* COA(long unsigned** Cols, int m, int n){
    int s = m/2;
    while(s >= 1){
        int i;
        for(i = 0; i < s; i++){
            long unsigned* c1 = Cols[i];
            long unsigned* c2 = Cols[i+s];
            int j;
            for(j = 0; j < n; j++){
                Cols[i][j] = c1[j] | c2[j];
            }
        }
        s /= 2;
    }
    return Cols[0];
}

long unsigned* decompress(long unsigned* bitvector, int number)
{
    // should be the reverse of compress...
    int todecomp = bitvector[0];
    long i;
    long unsigned * answer = (long unsigned*) malloc(sizeof(long unsigned) * number);
    int addat = 0;
    for(i = 1; i < todecomp+1; i++){
        if(bitvector[i] >> 63){
            // if this vector is a fill..
            long unsigned value = (bitvector[i] >> 62) & 0x1; // get the 63rd bit...
            if(value) // if value was 1, it is actually just f...
                value = 0x7fffffffffffffff;
            int amount =  bitvector[i] & 0x3fffffffffffffff; // clear off the last 2 bits..
            int j = 0;
            for(; j < amount; j++){
                answer[addat] = value;
                addat++;
            }
        }else{
            // this is the boring case...
            answer[addat] = bitvector[i] & 0x7fffffffffffffff;
            addat++;
        }
    }
    return answer;
}

void testCompressDecompress()
{
    long unsigned int randomnumbers[] = {0x0, 			0x0, 
					 0x7fffffffffffffff, 	0x7fffffffffffffff, 
					 0x0, 			0x0000027592351768, 
					 0x1ffffffffffffff1};
    int number = 7;
    testCompressDecompress(randomnumbers, number);
    int numOfElements = 0;
    long unsigned * t1 = compressNew(randomnumbers, number, numOfElements);
    long unsigned * t2 = compress(randomnumbers, number);
    for(int i = 0; i < numOfElements; i++){
        if(t2[i+1] != t1[i]){
            printf("Sad ... %d %lx %lx\n", i, t2[i+1], t1[i]);
            exit(0);
        }
    }
    printf("Done\n");
}

void testCompressDecompress(unsigned long * bitVector, int size){ 
    long unsigned int* temp = compress(bitVector, size);
    int i;
    // for(i = 0; i < (int) temp[0]+1; i++){
    //     printf("%lx \n", temp[i]);
    // }
    long unsigned int* output = decompress(temp, size);
    
    for(i = 0; i < size; i++){
        if(output[i] != bitVector[i]){
            printf("Test failed at index: %d, original: %lx, decompressed: %lx\n", i, bitVector[i], output[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test compression passed...\n");
}
