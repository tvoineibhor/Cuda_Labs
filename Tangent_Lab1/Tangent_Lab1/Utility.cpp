#include "Utility.h"


void createArr(double * arr, int size)
{
	srand(time(0));
	for (int i = 0; i < size; i++)
	{
		arr[i] = rand();
	}
}

void printArr(double * arr, double * res, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << "tg(" << arr[i] << ")" << " = " << res[i] << std::endl;
	}
	std::cout << "\n\n";
}