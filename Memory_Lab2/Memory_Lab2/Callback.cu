#include "Callback.h"


void MesurePerfomance(void(*f)(float *, float *, int), float * devInputArr, float * devOutputArr, int size, int block_size, char* type)
{
	float time_elapsed = 0;
	cudaEvent_t start;
	cudaEvent_t stop;

	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < 1000; i++)
	{
		f << <size / block_size, block_size >> > (devInputArr, devOutputArr, size);
	}

	CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
	CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
	CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_elapsed, start, stop));

	printf("%s done in %f milliseconds\n", type, time_elapsed);
}
