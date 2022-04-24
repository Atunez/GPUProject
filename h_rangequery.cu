#include "h_rangequery.h"
#include "wrappers.h"

long* compress(long* bitvector, int number){
    long* answer = (long*) Malloc(sizeof(long) * (number+1)); // at worst, we don't compress...
    // answer[0] will denote the number of used spots in answer...
    int i;
    for(i = 0; i < number; i++){
        // we should do the compression here...
        // all the numbers in bitvector are 63 bit numbers..
        bool isfill = true;
        int j = 0;
        int start = 0;
        for(; j < 63; j++){
            long toshift = bitvector[i];
            long consider = toshift & (0x1 << j);
            if(j == 0){
                start = consider;
                continue;
            }
            if(consider != start){
                isfill = false;
                break;
            }
        }

        if(isfill){
            // fill atom
            // has the form X|Y|Z, where X=1, Y= 0 or 1 and Z is count
            long lastAnswer = answer[answer[0]-1];
            if(lastAnswer >> 63){
                // if the last was a fill, then we need to do more work :(

                // if they have the same fill then just increment...
                if(lastAnswer & 0x1 == bitvector[i] & 0x1){
                    // we will assume that we don't actually overflow here...
                    answer[answer[0]-1]++;
                    // we don't add anything new
                    continue;
                }else{
                    // fresh insert....
                    long toput = 0x8000000000000001;
                    toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                    answer[answer[0]] = toput;
                }
                
            }else{
                long toput = 0x8000000000000000;
                toput |= (bitvector[i] & 0x1) << 62; // either 0 or 1...
                answer[answer[0]] = toput;
            }

        }else{
            // literal atom
            // and is for safety, shouldn't be needed....
            answer[answer[0]] = bitvector[i] & 0x7fffffffffffffff;
        }
        answer[0]++;
    }

    return answer;
}

long* decompress(long* bitvector, int number){
    // should be the reverse of compress...
    int todecomp = bitvector[0];
    int i;
    long * answer = (long*) Malloc(sizeof(long) * number);
    int addat = 0;
    for(i = 1; i < todecomp+1; i++){
        if(bitvector[i] >> 63){
            // if this vector is a fill..
            int value = (bitvector[i] >> 62) & 0x1; // get the 63rd bit...
            if(value) // if value was 1, it is actually just f...
                value = 0xffffffffffffffff;
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