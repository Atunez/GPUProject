#include "h_rangequery.h"
#include "wrappers.h"
#include <stdlib.h>
#include <stdio.h>

long unsigned* compress(long unsigned* bitvector, int number)
{
    // worst case no copmression
    long unsigned* answer = (long unsigned*) malloc(sizeof(long unsigned) * number); 
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
    long unsigned int* totest = randomnumbers;
    // temp should be: 0x80.........02, 0xc0.....02, 0x80.....01, ....
    long unsigned int* temp = compress(randomnumbers, number);
    int i;
    // for(i = 0; i < (int) temp[0]+1; i++){
    //     printf("%lx \n", temp[i]);
    // }
    long unsigned int* output = decompress(temp, number);
    
    for(i = 0; i < number; i++){
        if(output[i] != totest[i]){
            printf("Test failed at index: %d, original: %lx, decompressed: %lx\n", i, totest[i], output[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test compression passed...\n");
}

void testCompressDecompress2(unsigned long * bitVector, int size){
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
