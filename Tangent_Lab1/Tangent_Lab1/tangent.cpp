#include "tangent.h"

void tangent(double* res, double* arr, int size)
{
	for (int i = 0; i < size; i++)
		res[i] = tan(arr[i]);
}