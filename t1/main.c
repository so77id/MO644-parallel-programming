#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

double count_sort_serial(double a[], int n) {
	int i, j, count;
	double *temp;
	double start, end, duration;

	temp = (double *)malloc(n*sizeof(double));

	start = omp_get_wtime();
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}

	memcpy(a, temp, n*sizeof(double));
	free(temp);

	end = omp_get_wtime();

	duration = end - start;
	return duration;
}


double count_sort_parallel(double a[], int n, int n_threads) {
	int i, j, count;
	double *temp;
	double start, end, duration;

	temp = (double *)malloc(n*sizeof(double));

	start = omp_get_wtime();
	#pragma omp parallel for num_threads(n_threads) shared(a, temp, n) private(i, j, count)
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}

	memcpy(a, temp, n*sizeof(double));
	free(temp);

	end = omp_get_wtime();

	duration = end - start;
	return duration;
}

double count_sort_parallel2(double a[], int n, int n_threads) {
	int i, j, count;
	double *temp;
	double start, end, duration;

	temp = (double *)malloc(n*sizeof(double));

	start = omp_get_wtime();
	#pragma omp parallel for num_threads(n_threads) shared(a, temp, n) private(i, j, count)
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}

	#pragma omp parallel for num_threads(n_threads) shared(a, temp, n) private(i)
	for (i = 0; i < n; ++i)
	{
		a[i] = temp[i];
	}

	free(temp);

	end = omp_get_wtime();

	duration = end - start;
	return duration;
}

int main(int argc, char const *argv[])
{
	int n_threads, size, i;
	double *array, duration;

	//Scan NThreads
	scanf("%d", &n_threads);

	//Scan size of array
	scanf("%d", &size);

	//Create array
	array = (double *)malloc(size*sizeof(double));
	//Scan array
	for (i = 0; i < size; ++i)
	{
		scanf("%lf", &array[i]);
	}

	//duration = count_sort_serial(array, size);

	duration = count_sort_parallel(array, size, n_threads);

	//duration = count_sort_parallel2(array_parallel2, size, n_threads);

	//Print array
	for (i = 0; i < size; ++i)
	{
		printf("%.2f ", array[i]);
	}
	printf("\n%lf", duration);

	return 0;
}