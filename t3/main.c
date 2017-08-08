#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>


/*
//NOT OPTIMIZED
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

//OPTIMIZED
void histogram_serial(double min, double max, int * vet, int nbins, double h, double * val, int nval) {
    int i, j, count;
    double min_t, max_t;


    for(i=0;i<nval;i++) {

        j = (int)floor((val[i] - min)/(h));

        min_t = floor((val[i] - min)/(h))*h + min;

        if((float)val[i] == (float)min_t || j == nbins) j--;

        if (j >= 0 && j < nbins)
            vet[j]++;
    }
}
*/

// Structure used for pass data to worker
struct histogram_struct {
    double min;
    int *hist;
    int nbins;
    double h_step;
    double *values;
    int nval;
    int thread_id, nthreads;
};

/*
//NOT OPTIMIZED
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
*/


//OPTIMIZED
void *histogram_worker(void *arg){
    // Get data
    struct histogram_struct * worker_data = arg;

    // Get intervals for this thread
    int i0 = ceil(((double)worker_data->nval / (double)worker_data->nthreads) * (double)worker_data->thread_id);
    int in = ceil(((double)worker_data->nval / (double)worker_data->nthreads) * ((double)worker_data->thread_id + 1.0));


    int i;
    // Set local histogram
    int *local_hist = (int *)malloc(worker_data->nbins * sizeof(int));
    memset(local_hist, 0, worker_data->nbins * sizeof(int));

    // For each data belongs to interval
    for(; i0 < in; i0++){
        // Get index in local histogram
        i = (int)floor((worker_data->values[i0] - worker_data->min)/(worker_data->h_step));

        // Get min value of i bin
        double local_min = floor((worker_data->values[i0] - worker_data->min)/(worker_data->h_step))*worker_data->h_step + worker_data->min;

        // if value in i0 is equal to min value of i bin or i is equal to nbins, value in i0 belongs to i-1 bin
        if((float)worker_data->values[i0] == (float)local_min || i == worker_data->nbins) i--;

        // if index is in the range it is added to the local histogram
        if (i >= 0 && i < worker_data->nbins)
            local_hist[i]++;
    }

    // Sum local histogram to final histogram using atomic lock
    for (i = 0; i < worker_data->nbins; i++)
        if(local_hist[i] > 0)
            __sync_fetch_and_add(&worker_data->hist[i], local_hist[i]);

    // free memory of local histogram
    free(local_hist);
}

void histogram_parallel(int nthreads, double min, double max, int* hist, int nbins, double h_step, double *values, int nval){

    int i;
    // Create nthreads
    pthread_t *threads = malloc(nthreads* sizeof(pthread_t));
    struct histogram_struct *threads_data = malloc(nthreads*sizeof(struct histogram_struct));

    // For each thread the function worker is called
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

    // Join to each thread for continue
    for (i = 0; i < nthreads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    // Free memory of dinamic arrays
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

    // Scan inputs
    scanf("%d", &nthreads);
    scanf("%d", &nval);
    scanf("%d", &nbins);

    hist = (int *)malloc(nbins * sizeof(int));
    memset(hist, 0, nbins * sizeof(int));

    values = (double *)malloc(nval * sizeof(double));
    //Scan inputs and get min-max values
    for (i = 0; i < nval; ++i)
    {
        scanf("%lf", &values[i]);
        if(min > floor(values[i])) min = floor(values[i]);
        else if(max < ceil(values[i])) max = ceil(values[i]);
    }

    // Get step between bins
    h_step = (max - min)/(nbins * 1.0);

    // Call histogram function
    gettimeofday(&start, NULL);
    histogram_parallel(nthreads, min, max, hist, nbins, h_step, values, nval);
    gettimeofday(&end, NULL);

    // Print bins values
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

    // Print values
    for (i = 0; i < nbins; ++i)
    {
        if(i == 0){
            printf("%d", hist[i]);
        } else {
            printf(" %d", hist[i]);
        }
    }

    // Print duration
    duration = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

    printf("\n%d", duration);


    return 0;
}


/*

# Cores = 4

Using not optimized algorithm

Serial times
arq1 = 555
arq2 = 58038
arq3 = 436387

--------------------------------------------------------------------
|      | Threads   |     1   |    2   |     4   |     8   |    16  |
--------------------------------------------------------------------
| arq1 | tempo     |    773  |   386  |    366  |    420  |   592  |
|      -------------------------------------------------------------
|      | Speedup   |  0.717  |  1.437 |   1.516 |   1.321 |  0.937 |
|      -------------------------------------------------------------
|      | Eficência |  0.179  |  0.359 |   0.379 |  0.3303 | 0.2343 |
--------------------------------------------------------------------
| arq2 | tempo     |  62565  |  36491 |  21992  |  22225  |  25418 |
|      -------------------------------------------------------------
|      | Speedup   |  0.9276 |  1.590 |  2.639  |  2.611  |  2.283 |
|      -------------------------------------------------------------
|      | Eficência |  0.231  |  0.397 |  0.659  |  0.652  | 0.5708 |
--------------------------------------------------------------------
| arq3 | tempo     |  436387 | 213438 | 156635  |  157581 | 172861 |
|      -------------------------------------------------------------
|      | Speedup   |   1.033 |  2.044 |  2.710  |   2.694 |  2.524 |
|      -------------------------------------------------------------
|      | Eficência |   0.258 | 0.5111 |  0.677  |   0.673 |  0.631 |
--------------------------------------------------------------------

********
ENGLISH
********
The algorithm proposed for the task was used (Line # 11) and it was proposed to perform a parallelization (Line # 60). This parallelization was performed with different input files and different thread numbers, which is summarized in the table displayed. As the number of threads increases the Speed up grows to reach its maximum in 4 threads, after this when increasing the number of threads the speedup decreases being this for the cost overhead of having many threads running and the amount of page faults.

********
ESPANHOL
********
Se utilizo el algoritmo propuesto para la tarea(Linea #11) y se propuso realizar una paralelizacion (Linea #60). Esta paralelizacion fue probrada con distintos archivos de entrada y distitntos numeros de threads, la cual esta resumida en la tabla desplegada. A medida que el numero de threads aumento el speeup crece llegando a su maximo en el numero 4, luego de esto al aumentar el numero de threads el speedup disminuye siendo esto por el costo overhead de tener muchos threads funcionando y la cantidad de page faults.



Using optimized algorithm


Serial times
arq1 = 10.81
arq2 = 121.49
arq3 = 1765.43

--------------------------------------------------------------------
|      | Threads   |     1   |    2   |     4   |     8   |    16  |
--------------------------------------------------------------------
| arq1 | tempo     |    103  |   135  |    659  |    255  |   412  |
|      -------------------------------------------------------------
|      | Speedup   |   0.104 |  0.079 |   0.016 |   0.042 |  0.026 |
|      -------------------------------------------------------------
|      | Eficência |   0.026 |  0.019 |   0.004 |   0.010 |  0.006 |
--------------------------------------------------------------------
| arq2 | tempo     |    238  |    205 |     231 |    302  |    479 |
|      -------------------------------------------------------------
|      | Speedup   |   0.509 |  0.590 |   0.524 |   0.401 |  0.253 |
|      -------------------------------------------------------------
|      | Eficência |   0.127 |  0.147 |   0.131 |   0.100 |  0.063 |
--------------------------------------------------------------------
| arq3 | tempo     |    2030 |  1140  |   1031  |    1134 |   1227 |
|      -------------------------------------------------------------
|      | Speedup   |   0.869 |  1.547 |   1.711 |   1.556 |  1.430 |
|      -------------------------------------------------------------
|      | Eficência |   0.217 |  0.386 |   0.427 |   0.389 |  0.359 |
--------------------------------------------------------------------



********
ENGLISH
********
We was decided to perform an optimization of the function by lowering the complexity of the algorithm (Line # 30) and then this optimized algorithm was parallelized (Line # 91). As with the original algorithm, different tests were run with different input files and different thread numbers. In this it can be seen that for arq1 and arq2 it can be seen that there is no speedup because the serial algorithm is faster than the parallelized due to overhead costs. In the case of arq3 we can see a speedup because the size of data in this input file is so much and that makes the cost benefit of the parallelization of positive results with respect to speedup, but also as in the Algorithm not parallelized the maximum is in the 4 threads, this due to the overhead costs of having more threads running.

********
ESPANHOL
********
Se decidio realizar una optimizacion de la funcion bajando la complejidad del algoritmo(Linea #30) y luego este algoritmo optimizado fue paralelizado (Linea #91). Al igual que con el algoritmo original se corrieron distintas pruebas con distintos archivos de entrada y distintos numeros de threads. En este se puede ver que para el arq1 y arq2 se puede ver que no existe un speedup debido a que el algoritmo serial es mas rapido que el paralelizado debido a los costos overhead. En el caso del arq3 se puede ver un speedup debido a que la cantidad de datos de este archivo de entrada es muy alta y eso hace que el costo beneficio de la paralelizacion de resultados positivos con respecto al speedup, pero tambien al igual que en el algoritmo no paralelizado el maximo esta en los 4 threads, esto debido a los costos overhead de tener mas threads ejecutandose.



********
ENGLISH
********
For all the experiments the time calculation was performed obtaining the average time of more than 1000 instances of the same input.


********
ESPANHOL
********
Para todos los experimentos el calculo de los tiempo se realizo obteniendo el promedio del tiempo de mas de 1000 instancias de la misma entrada.


GPROF

  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
 50.15      0.01     0.01                             histogram_worker
 50.15      0.02     0.01                             main
  0.00      0.02     0.00        1     0.00     0.00  histogram_parallel

********
ENGLISH
********
The percentage of the program that is parallelizable is 100%, because what is done is to divide the vector into N_THREADS portions and independently count each one of them, to then make a general sum.

********
ESPANHOL
********
El porcentaje del programa que es paralelizable es el 100%, debido a que lo que se hace es divir el vector en N_THREADS porciones y contabilizar de manera independiente cada una de estas, para luego realizar una suma general.



*/
