#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Callback.h"


#ifndef __CUDACC__ 
	#define __CUDACC__
#endif

#include <device_functions.h>

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <vector>

using namespace std;

#define N 100000
#define block_size 1000

texture<float, cudaTextureType1D, cudaReadModeElementType> texX;

__global__ void textureInverse(float * inputArr, float * outputArr, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int id_reverse = (size - 1 - idx);

	if (idx < size)
		outputArr[idx] = tex1Dfetch(texX, float(id_reverse));
}

__global__ void globalInverse(float * inputArr, float * outputArr, int size)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int id_reverse = (blockDim.x * gridDim.x) - 1 - idx;

	if (idx < size)
		outputArr[idx] = inputArr[id_reverse];
}

__global__ void sharedInverse(float * inputArr, float * outputArr, int size)
{

	__shared__ float temp[block_size];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int id_reverse = (blockDim.x * gridDim.x) - 1 - idx;


	temp[threadIdx.x] = inputArr[id_reverse];

	__syncthreads();

	if (idx < size)
		outputArr[idx] = temp[threadIdx.x];

}


int main()
{

	float* inputArr = new float[N];
	float* outputArr = new float[N];

	for (int i = 0; i < N; i++)
	{
		inputArr[i] = i + 1;
	}

	float* devInputArr;
	float* devOutputArr;
	
	CUDA_CHECK_ERROR(cudaMalloc((void**)&devInputArr, N * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&devOutputArr, N * sizeof(float)));

	CUDA_CHECK_ERROR(cudaMemcpy(devInputArr, inputArr, N * sizeof(float), cudaMemcpyHostToDevice));

	texX.normalized = false;
	CUDA_CHECK_ERROR(cudaBindTexture((size_t)0, &texX, devInputArr, &texX.channelDesc, N * sizeof(float)));

	//cudaMemcpyToSymbol(temp_c, inputArr, N * sizeof(float), 0);

	MesurePerfomance(globalInverse, devInputArr, devOutputArr, N, block_size, "Global");
	//MesurePerfomance(constantInverse, temp_c, devOutputArr, N, block_size, "Constant");
	MesurePerfomance(sharedInverse, devInputArr, devOutputArr, N, block_size, "Shared");
	MesurePerfomance(textureInverse, devInputArr, devOutputArr, N, block_size, "Texture");

	CUDA_CHECK_ERROR(cudaMemcpy(outputArr, devOutputArr, N * sizeof(float), cudaMemcpyDeviceToHost));

	delete inputArr;
	delete outputArr;

	CUDA_CHECK_ERROR(cudaFree(devInputArr));
	CUDA_CHECK_ERROR(cudaFree(devOutputArr));

	CUDA_CHECK_ERROR(cudaUnbindTexture(&texX));

	cout << endl;
}
