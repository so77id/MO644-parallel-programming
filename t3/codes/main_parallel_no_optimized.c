#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>

struct histogram_struct {
    double min;
    int *hist;
    int nbins;
    double h_step;
    double *values;
    int nval;
    int thread_id, nthreads;
};

void *histogram_worker(void *arg){

    struct histogram_struct * worker_data = arg;

    int i0 = ceil(((double)worker_data->nval / (double)worker_data->nthreads) * (double)worker_data->thread_id);
    int in = ceil(((double)worker_data->nval / (double)worker_data->nthreads) * ((double)worker_data->thread_id + 1.0));

    int i, j, count;
    double new_min, new_max;
    int *local_hist = (int *)malloc(worker_data->nbins * sizeof(int));
    memset(local_hist, 0, worker_data->nbins * sizeof(int));

    for (i = 0; i < worker_data->nbins; ++i)
    {
        count = 0;
        new_min = worker_data->min + i * worker_data->h_step;
        new_max = worker_data->min + (i+1)* worker_data->h_step;
        for(j = i0; j < in; j++){
            if(worker_data->values[j] <= new_max && worker_data->values[j] > new_min) {
                count++;
            }
        }
        if(count > 0)
            __sync_fetch_and_add(&worker_data->hist[i], count);
    }
}

void histogram_parallel(int nthreads, double min, double max, int* hist, int nbins, double h_step, double *values, int nval){

    int i;

    pthread_t *threads = malloc(nthreads* sizeof(pthread_t));
    struct histogram_struct *threads_data = malloc(nthreads*sizeof(struct histogram_struct));

    for (i = 0; i < nthreads; ++i)
    {
        threads_data[i].min = min;
        threads_data[i].hist = hist;
        threads_data[i].nbins = nbins;
        threads_data[i].h_step = h_step;
        threads_data[i].values = values;
        threads_data[i].nval = nval;
        threads_data[i].thread_id = i;
        threads_data[i].nthreads = nthreads;

        pthread_create(&threads[i], NULL, histogram_worker, &threads_data[i]);
    }

    for (i = 0; i < nthreads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    free(threads_data);
    free(threads);
}


int main(int argc, char const *argv[])
{
    int i, nthreads, nbins, nval, duration;
    int *hist;
    double min, max, h_step, id;
    double *values;
    struct timeval start, end;

    min = DBL_MAX;
    max = DBL_MIN;

    scanf("%d", &nthreads);
    scanf("%d", &nval);
    scanf("%d", &nbins);

    hist = (int *)malloc(nbins * sizeof(int));
    memset(hist, 0, nbins * sizeof(int));

    values = (double *)malloc(nval * sizeof(double));
    for (i = 0; i < nval; ++i)
    {
        scanf("%lf", &values[i]);
        if(min > floor(values[i])) min = floor(values[i]);
        else if(max < ceil(values[i])) max = ceil(values[i]);
    }

    h_step = (max - min)/(nbins * 1.0);

    gettimeofday(&start, NULL);
    histogram_parallel(nthreads, min, max, hist, nbins, h_step, values, nval);
    gettimeofday(&end, NULL);

    i = 0;
    for (id = min; id <= (max+(h_step/10.0)); id = id + h_step)
    {
        if(id == 0){
            printf("%0.2f", id);
        } else {
            printf(" %0.2f", id);
        }
        i++;
    }

    printf("\n");

    for (i = 0; i < nbins; ++i)
    {
        if(i == 0){
            printf("%d", hist[i]);
        } else {
            printf(" %d", hist[i]);
        }
    }

    duration = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

    printf("\n%d", duration);


    return 0;
}
