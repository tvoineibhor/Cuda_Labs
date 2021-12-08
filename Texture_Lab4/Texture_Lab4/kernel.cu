#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Helper.h"
#include <ctime>

#ifndef __CUDACC__ 
	#define __CUDACC__
#endif

#include <device_functions.h>
#include "EBMP/EasyBMP.h"

#define STEP 2

using namespace std;

enum FilterMethod
{
	CPU,
	GPU
};

__host__ __device__
RGBApixel cubicInterpolate(double x, RGBApixel* p)
{
	RGBApixel res;
	double r = p[1].Red + 0.5 * x*(p[2].Red - p[0].Red +
		x * (2.0*p[0].Red - 5.0*p[1].Red + 4.0*p[2].Red - p[3].Red +
			x * (3.0*(p[1].Red - p[2].Red) + p[3].Red - p[0].Red)));

	if (r < 0.) r = 0.0;
	if (r > 255.) r = 255.;

	double g = p[1].Green + 0.5 * x*(p[2].Green - p[0].Green +
		x * (2.0*p[0].Green - 5.0*p[1].Green + 4.0*p[2].Green - p[3].Green +
			x * (3.0*(p[1].Green - p[2].Green) + p[3].Green - p[0].Green)));

	if (g < 0.) g = 0.0;
	if (g > 255.) g = 255.;

	double b = p[1].Blue + 0.5 * x*(p[2].Blue - p[0].Blue +
		x * (2.0*p[0].Blue - 5.0*p[1].Blue + 4.0*p[2].Blue - p[3].Blue +
			x * (3.0*(p[1].Blue - p[2].Blue) + p[3].Blue - p[0].Blue)));

	if (b < 0.) b = 0.0;
	if (b > 255.) b = 255.;

	double a = p[1].Alpha + 0.5 * x*(p[2].Alpha - p[0].Alpha +
		x * (2.0*p[0].Alpha - 5.0*p[1].Alpha + 4.0*p[2].Alpha - p[3].Alpha +
			x * (3.0*(p[1].Alpha - p[2].Alpha) + p[3].Alpha - p[0].Alpha)));

	if (a < 0.) a = 0.0;
	if (a > 255.) a = 255.;

	res.Red = (ebmpBYTE)r;
	res.Green = (ebmpBYTE)g;
	res.Blue = (ebmpBYTE)b;
	res.Alpha = (ebmpBYTE)a;

	return res;
}


__host__ __device__
RGBApixel bicubicInterpolate(double x, double y, RGBApixel p[4][4])
{
	RGBApixel arr[4];
	arr[0] = cubicInterpolate(x, p[0]);
	arr[1] = cubicInterpolate(x, p[1]);
	arr[2] = cubicInterpolate(x, p[2]);
	arr[3] = cubicInterpolate(x, p[3]);

	return cubicInterpolate(y, arr);
}

void saveImage(RGBApixel* image, int height, int width, FilterMethod method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j].Red;
			pixel.Green = image[i * width + j].Green;
			pixel.Blue = image[i * width + j].Blue;
			Output.SetPixel(j, i, pixel);
		}
	}

	if(method == FilterMethod::CPU)
		Output.WriteToFile("image_CPU_filtered.bmp");
	else if(method == FilterMethod::GPU)
		Output.WriteToFile("image_GPU_filtered.bmp");

}


void BicubicFilterCPU(RGBApixel* input, int h, int w, RGBApixel* output)
{
	RGBApixel point[4][4];
	
	for (int i = 0; i < (h - 2); i++)
	{
		
		for (int j = 0; j < (w - 2); j++)
		{
			point[0][0] = input[(i + 0) * (w) + j];
			point[1][0] = input[(i + 1) * (w) + j];
			point[2][0] = input[(i + 2) * (w) + j];
			point[3][0] = input[(i + 3) * (w) + j];
										   
			point[0][1] = input[(i + 0) * (w) + j + 1];
			point[1][1] = input[(i + 1) * (w) + j + 1];
			point[2][1] = input[(i + 2) * (w) + j + 1];
			point[3][1] = input[(i + 3) * (w) + j + 1];
										   
			point[0][2] = input[(i + 0) * (w) + j + 2];
			point[1][2] = input[(i + 1) * (w) + j + 2];
			point[2][2] = input[(i + 2) * (w) + j + 2];
			point[3][2] = input[(i + 3) * (w) + j + 2];
										   
			point[0][3] = input[(i + 0) * (w) + j + 3];
			point[1][3] = input[(i + 1) * (w) + j + 3];
			point[2][3] = input[(i + 2) * (w) + j + 3];
			point[3][3] = input[(i + 3) * (w) + j + 3];

			for (int y = 0; y < STEP; y++)
			{
				for (int x = 0; x < STEP; x++)
				{
					int rx = (j) * STEP + x;
					int ry = (i) * STEP + y;

					double ax = (double)x / STEP;
					double ay = (double)y / STEP;

					RGBApixel res = bicubicInterpolate(ax, ay, point);

					output[ry * (w - 2) * STEP + rx] = res;
				}
			}
		}
	}
}

__global__ void BicubicFilterGPU(int height, int width, RGBApixel* input, RGBApixel* output)
{
	__shared__ RGBApixel point[4][4];

	if (threadIdx.x < 4 && threadIdx.y < 4)
	{
		int col = blockIdx.x + threadIdx.x;
		int row = blockIdx.y + threadIdx.y;

		point[threadIdx.y][threadIdx.x] = input[row * width + col];
	}

	__syncthreads();

	if (threadIdx.x < STEP && threadIdx.y < STEP)
	{
		int rx = blockIdx.x * STEP + threadIdx.x;
		int ry = blockIdx.y * STEP + threadIdx.y;

		double ax = double(threadIdx.x) / STEP;
		double ay = double(threadIdx.y) / STEP;

		RGBApixel res = bicubicInterpolate(ax, ay, point);

		output[ry * (width - 2) * STEP + rx] = res;
	}
}

int main(int argc, char **argv)
{
	int iterations = 100;

	BMP Image;
	
	Image.ReadFromFile("TRU256.BMP");

	int height = Image.TellHeight();
	int width = Image.TellWidth();

	int out_h = (height - 2) * STEP;
	int out_w = (width - 2) * STEP;

	RGBApixel* outputCPU = (RGBApixel*)calloc(out_h * out_w, sizeof(RGBApixel));
	RGBApixel* outputGPU = (RGBApixel*)calloc(out_h * out_w, sizeof(RGBApixel));
	RGBApixel* imageArray = (RGBApixel*)calloc(height * width, sizeof(RGBApixel));

	for (int j = 0; j < Image.TellHeight(); j++)
		for (int i = 0; i < Image.TellWidth(); i++)
			imageArray[j * Image.TellWidth() + i] = Image.GetPixel(i, j);


	////////////////////////////////////////////////////////////////////
	
	unsigned int start_time = clock();

	for (int i = 0; i < iterations; i++)
		BicubicFilterCPU(imageArray, height, width, outputCPU);

	float elapsed_time = clock() - start_time;
	float cpu_time = elapsed_time / iterations;

	printf("CPU time: %f msec\n", cpu_time);

	RGBApixel* devImageArray;
	RGBApixel* devOutputGPU;

	cudaMalloc((void**)&devImageArray, height * width * sizeof(RGBApixel));
	cudaMalloc((void**)&devOutputGPU, out_h * out_w * sizeof(RGBApixel));

	cudaMemcpy(devImageArray, imageArray, height * width * sizeof(RGBApixel), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock((STEP <= 4) ? 4 : STEP, (STEP <= 4) ? 4 : STEP);
	dim3 blocks((width - 2), (height - 2));

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

	cudaMemcpy(outputGPU, devOutputGPU, out_h * out_w * sizeof(RGBApixel), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	saveImage(outputCPU, out_h, out_w, FilterMethod::CPU);
	saveImage(outputGPU, out_h, out_w, FilterMethod::GPU);

	cudaFree(devImageArray);
	cudaFree(devOutputGPU);

	free(outputCPU);
	free(outputGPU);

	return 0;
}

