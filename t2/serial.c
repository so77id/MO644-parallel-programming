#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

 
void producer_consumer(int *buffer, int size, int *vec, int n) {
	int i, j;
	long long unsigned int sum = 0;

	for(i=0;i<n;i++) {
		if(i % 2 == 0) {	// PRODUTOR
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
			for(j=0;j<size;j++) {
				sum += buffer[j];
			}
		}
	}
	printf("%llu\n",sum);
}




int main(int argc, char const *argv[])
{
	int n_threads, size, n, *vec, *buffer, i;
	double start, end;

	//Scan NThreads
	scanf("%d", &n_threads);

	//Scan number of iterations
	scanf("%d", &n);

	//Scan size of buffer
	scanf("%d", &size);

	//Create vec
	vec = (int *)malloc(n*sizeof(int));
	buffer = (int *)malloc(size*sizeof(int));

	//Scan vec
	for (i = 0; i < n; ++i)
	{
		scanf("%d", &vec[i]);
	}

	//Get start time
	start = omp_get_wtime();
	// Call function producer consumer
	producer_consumer(buffer, size, vec, n);
	// Get end time
	end = omp_get_wtime();

	// print duration of function
	printf("%lf\n", end - start);

	//Free memory used by vec
	free(vec);

	return 0;
}