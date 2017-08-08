#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define LEN(x)  (sizeof(x) / sizeof((x)[0]))

int main(int argc, char const *argv[])
{
	int i;
	int sum = 0;
	int num_threads = strtol(argv[1], NULL, 10);
	int array[10] = {1,2,3,4,5,6,7,8,9,10};

	#pragma omp parallel for num_threads(num_threads) shared(sum) private(i)
	for( i = 0; i < LEN(array); i++)
	{
		printf("Sumando indice: %d, valor: %d\n", i+1, array[i]);
		#pragma omp critical
		sum += array[i];
	}

	printf("sum: %d\n", sum);

	return 0;
}