#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void tangent(double * res, double * arr, int size);

__global__ void tangentKernel(double * res, double * arr, int size);
