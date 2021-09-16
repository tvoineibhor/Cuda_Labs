#include "tangent.h"


__global__ void tangentKernel(double* res, double* arr, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
		res[idx] = tan(arr[idx]);
}