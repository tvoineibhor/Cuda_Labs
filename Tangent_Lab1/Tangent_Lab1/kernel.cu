
#include "Utility.h"
#include "tangent.h"

using namespace std;

#define N 1000000
#define Block_Size 200

cudaError_t tangentCuda(double* res, double* arr, int size)
{
	double* arr_device = 0;
	double* res_device = 0;

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSetDevice(0);

	cudaMalloc((void**)&res_device, size * sizeof(double));
	cudaMalloc((void**)&arr_device, size * sizeof(double));

	cudaMemcpy(arr_device, arr, size * sizeof(double), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	tangentKernel << <N / Block_Size, Block_Size >> > (res_device, arr_device, size);
	cudaEventRecord(stop);

	cudaGetLastError();

	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(res, res_device, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);

	float time_elapsed = 0;
	cudaEventElapsedTime(&time_elapsed, start, stop);

	printf("GPU done in %f milliseconds\n", time_elapsed);

	cudaFree(res_device);
	cudaFree(arr_device);

	return cudaStatus;
}

void tangentCpu(double* res, double* arr, int size)
{
	float start = ((float)clock() / (CLOCKS_PER_SEC)) * 1000.0;

	tangent(res, arr, N); // On CPU

	float end = ((float)clock() / (CLOCKS_PER_SEC)) * 1000.0;

	float time_elapsed = end - start;
	printf("CPU done in %f milliseconds\n", time_elapsed);
}

int main()
{
	double* arr = new double[N];
	double* res = new double[N];

	createArr(arr, N);

	cout << "Result: " << endl;

	cudaError_t cudaStatus = tangentCuda(res, arr, N); // On GPU
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "tanWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	tangentCpu(res, arr, N);

	return 0;
}