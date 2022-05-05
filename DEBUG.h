#define PRINTONCE(call) \
{ \
	if (blockIdx.x * blockdim.x + threadIdx.x == 0) \
	{ \
		printf(call); \
	} \
}

