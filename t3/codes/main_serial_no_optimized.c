#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>



void histogram_serial(double min, double max, int * vet, int nbins, double h, double * val, int nval) {
    int i, j, count;
    double min_t, max_t;

    for(j=0;j<nbins;j++) {
        count = 0;
        min_t = min + j*h;
        max_t = min + (j+1)*h;
        for(i=0;i<nval;i++) {
            if(val[i] <= max_t && val[i] > min_t) {
                count++;
            }
        }
        vet[j] = count;
    }
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
    histogram_serial(min, max, hist, nbins, h_step, values, nval);
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