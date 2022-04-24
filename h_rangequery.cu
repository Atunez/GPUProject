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
    
}