#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Mem.h"

class Global : IMemType
{
	// Унаследовано через IMemType
	
public:
	virtual void Inverse() override;
	float * inputArr;
	float * outputArr;
	int size;
	Global(float * devInputArr, float * devOutputArr, int size, int block_size);
};

