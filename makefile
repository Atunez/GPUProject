NVCC = /usr/bin/nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = rangequery.o wrappers.o h_rangequery.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

rangequery: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -ljpeg -o rangequery

rangequery.o: rangequery.cu wrappers.h h_rangequery.h 

h_rangequery.o: h_rangequery.cu h_rangequery.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm rangequery *.o
