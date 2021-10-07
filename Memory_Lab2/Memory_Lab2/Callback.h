#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "CheckError.h"
															  \
void MesurePerfomance(void(*f)(float *, float *, int), float * devInputArr, float * devOutputArr, int size, int block_size, char* type);

