#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Helper.h"
#include <ctime>


#ifndef __CUDACC__ 
	#define __CUDACC__
#endif

#include <device_functions.h>
#include "EBMP/EasyBMP.h"

#define STEP 4

enum FilterMethod
{
	CPU,
	GPU
};

void saveImage(float* image, int height, int width, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}

	if (method)
		Output.WriteToFile("icon_linear.bmp");
	else
		Output.WriteToFile("icon_linear.bmp");

}


__host__ __device__
float cubicInterpolate(float x, float* p)
{
	return
		p[1] + (-0.5 * p[0] + 0.5 * p[2]) * x
		+ (p[0] - 2.5 * p[1] + 2.0 * p[2]
			- 0.5 * p[3]) * x * x
		+ (-0.5 * p[0] + 1.5 * p[1] - 1.5 * p[2]
			+ 0.5 * p[3]) * x * x * x;
}


__host__ __device__
float bicubicInterpolate(float x, float y, float p[4][4])
{
	float arr[4];
	arr[0] = cubicInterpolate(x, p[0]);
	arr[1] = cubicInterpolate(x, p[1]);
	arr[2] = cubicInterpolate(x, p[2]);
	arr[3] = cubicInterpolate(x, p[3]);

	return cubicInterpolate(y, arr);
}


void saveImage_(float* image, int height, int width, FilterMethod method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width +  j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}

	if(method == FilterMethod::CPU)
		Output.WriteToFile("image_CPU_filtered.bmp");
	else if(method == FilterMethod::GPU)
		Output.WriteToFile("image_GPU_filtered.bmp");

}


float BicubicFilterCPU(float* input, int h, int w, float* output)
{
	float point[4][4];
	for (int i = 0; i < (h - 1); i++)
	{
		for (int j = 0; j < (w - 1); j++)
		{
			point[0][0] = input[(i + 0) * (h + 2) + j];
			point[1][0] = input[(i + 1) * (h + 2) + j];
			point[2][0] = input[(i + 2) * (h + 2) + j];
			point[3][0] = input[(i + 3) * (h + 2) + j];

			point[0][1] = input[(i + 0) * (h + 2) + j + 1];
			point[1][1] = input[(i + 1) * (h + 2) + j + 1];
			point[2][1] = input[(i + 2) * (h + 2) + j + 1];
			point[3][1] = input[(i + 3) * (h + 2) + j + 1];

			point[0][2] = input[(i + 0) * (h + 2) + j + 2];
			point[1][2] = input[(i + 1) * (h + 2) + j + 2];
			point[2][2] = input[(i + 2) * (h + 2) + j + 2];
			point[3][2] = input[(i + 3) * (h + 2) + j + 2];

			point[0][3] = input[(i + 0) * (h + 2) + j + 3];
			point[1][3] = input[(i + 1) * (h + 2) + j + 3];
			point[2][3] = input[(i + 2) * (h + 2) + j + 3];
			point[3][3] = input[(i + 3) * (h + 2) + j + 3];

			for (float y = 0; y < STEP; y++)
			{
				for (float x = 0; x < STEP; x++)
				{
					int rx = (j)* STEP + x;
					int ry = (i)* STEP + y;

					float ax = x / STEP;
					float ay = y / STEP;

					float res = bicubicInterpolate(ax, ay, point);
					
					if (res < 0) res = 0.;
					if (res > 255) res = 255.;

					output[ry * (h - 1) * STEP + rx] = res;
				}
			}
		}

	}	

	return 0;
}

__global__ void BicubicFilterGPU(int h, int w, float* input, float* output)
{
	__shared__ float point[4][4];

	int col = blockIdx.x + threadIdx.x;
	int row = blockIdx.y + threadIdx.y;

	
	point[threadIdx.y][threadIdx.x] = input[row * (h + 2) + col];

	__syncthreads();

	int rx = blockIdx.x * STEP + threadIdx.x;
	int ry = blockIdx.y * STEP + threadIdx.y;

	float ax = float(threadIdx.x) / STEP;
	float ay = float(threadIdx.y) / STEP;

	float res = bicubicInterpolate(ax, ay, point);

	if (res < 0) res = 0.;
	if (res > 255) res = 255.;

	output[ry * (h - 1) * STEP + rx] = res;
}

int main(int argc, char **argv)
{
	int iterations = 100;
	BMP Image;
	Image.ReadFromFile("nyan-cat-150x150.bmp");

	int height = Image.TellHeight();
	int width = Image.TellWidth();

	int new_h = height + 2;
	int new_w = width + 2;

	int out_h = (height - 1) * STEP;
	int out_w = (width - 1) * STEP;

	float* outputCPU = (float*)calloc(out_h * out_w, sizeof(float));
	float* outputGPU = (float*)calloc(out_h * out_w, sizeof(float));
	float* imageArray = (float*)calloc(new_h * new_w, sizeof(float));


	for (int j = 0; j < Image.TellHeight(); j++)
		for (int i = 0; i < Image.TellWidth(); i++)
			imageArray[(j + 1) * new_h + (i+1)] = Image(i, j)->Red;



	////////////////////////////////////////////////////////////////////
	
	unsigned int start_time = clock();

	for (int i = 0; i < iterations; i++)
		BicubicFilterCPU(imageArray, height, width, outputCPU);

	float elapsed_time = clock() - start_time;
	float cpu_time = elapsed_time / iterations;

	printf("CPU time: %f msec\n", cpu_time);

	float* devImageArray;
	float* devOutputGPU;

	cudaMalloc((void**)&devImageArray, new_h * new_w * sizeof(float));
	cudaMalloc((void**)&devOutputGPU, out_h * out_w * sizeof(float));

	cudaMemcpy(devImageArray, imageArray, new_h * new_w * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(4, 4);
	dim3 blocks((height - 1), (width - 1));

	cudaEvent_t start;
	cudaEvent_t stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < iterations; i++)
		BicubicFilterGPU << <blocks, threadsPerBlock >> > (height, width, devImageArray, devOutputGPU);

	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(stop));

	elapsed_time = 0.;
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));

	float gpu_time = elapsed_time / iterations;

	printf("GPU time: %f msec\n", gpu_time);

	cudaDeviceSynchronize();

	cudaMemcpy(outputGPU, devOutputGPU, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();


	saveImage_(outputCPU, out_h, out_w, FilterMethod::CPU);
	saveImage_(outputGPU, out_h, out_w, FilterMethod::GPU);

	cudaFree(devImageArray);
	cudaFree(devOutputGPU);

	free(outputCPU);
	free(outputGPU);

	return 0;
}

