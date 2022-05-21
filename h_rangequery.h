#define BINSIZE 10

long unsigned int* compress(long unsigned int*, int);
long unsigned int* decompress(long unsigned int*, int);
long unsigned ** compressMultipleRows(long unsigned** bitVectors, int sizeOfBitVectors, int numberOfBitVectors);
void simpleQuery(int*, long unsigned*, char **, int, int);
void testCompressDecompress();
void testCompressDecompress(unsigned long *, int);
long unsigned* COA(long unsigned** Cols, int m, int n);
