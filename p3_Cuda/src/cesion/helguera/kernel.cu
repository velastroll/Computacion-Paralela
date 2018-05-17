__global__ void ReductionMax2(float *input, float *results, int n)    //take thread divergence into account
{
	extern __shared__ int sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tx = threadIdx.x;
	 //load input into __shared__ memory
	int x = INT_MIN;
	if(i < n)
		x = input[i];
	sdata[tx] = x;
	__syncthreads();

	// block-wide reduction
	for(unsigned int offset = blockDim.x>>1; offset > 0; offset >>= 1)
	{
		__syncthreads();
		if(tx < offset)
	    {
			if(sdata[tx + offset] > sdata[tx])
				sdata[tx] = sdata[tx + offset];
		}

	}

		// finally, thread 0 writes the result
	if(threadIdx.x == 0)
	{
		// the result is per-block
		results[blockIdx.x] = sdata[0];
	}
}

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}