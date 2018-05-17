#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include "kernel.cu"



int main()
{
	unsigned int N = 1024*1024*20;
	float *h_array;
	float *d_array;
	float *h_max;
	float *d_max;
	int *d_mutex;


	// allocate memory
	h_array = (float*)malloc(N*sizeof(float));
	h_max = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_array, N*sizeof(float));
	cudaMalloc((void**)&d_max, sizeof(float));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(float));


	// fill host array with data
	for(unsigned int i=0;i<N;i++){
		h_array[i] = i;
	}


	// set up timing variables
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


	// copy from host to device
	cudaEventRecord(gpu_start, 0);
	cudaMemcpy(d_array, h_array, N*sizeof(float), cudaMemcpyHostToDevice);


	// call kernel
	
	dim3 block(1,1204);
	dim3 grid(1,1);
	find_maximum_kernel<<< grid, block >>>(d_array, d_max, d_mutex, N);
	


	// copy from device to host
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);


	//report results
	std::cout<<"Maximo gpu: "<<*h_max<<std::endl;
	//std::cout<<"The gpu took: "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;


	/*//run cpu version
	clock_t cpu_start = clock();
	//for(unsigned int j=0;j<1000;j++){
		*h_max = -1.0;
		for(unsigned int i=0;i<N;i++){
			if(h_array[i] > *h_max){
				*h_max = h_array[i];
			}
		}
	//}
	clock_t cpu_stop = clock();
	clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;

	std::cout<<"Maximo: "<<*h_max<<std::endl;
	std::cout<<"CPU: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;*/



	// free memory
	free(h_array);
	free(h_max);
	cudaFree(d_array);
	cudaFree(d_max);
    cudaFree(d_mutex);
    
    return 0;
}