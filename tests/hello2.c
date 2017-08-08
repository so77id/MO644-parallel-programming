#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void hello()
{
	int my_rank = 	omp_get_thread_num();
	int thread_count = omp_get_num_threads();

	printf("Hello from thread %d of %d\n", my_rank, thread_count);
}


int main(int argc, char const *argv[])
{
	int n_threads = strtol(argv[1], NULL, 10);

	#pragma omp parallel num_threads(n_threads)
	hello();

	return 0;
}