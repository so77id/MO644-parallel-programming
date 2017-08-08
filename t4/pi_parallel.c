#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>


//Global value
long long unsigned int sum = 0;

//Function worker
void *monte_carlo_pi_worker(void *arg) {
    unsigned int tosses = (unsigned int)arg;
    long long unsigned int in = 0, i;
    double x, y, d;
    unsigned int seed = time(NULL);

    for (i = 0; i < tosses; i++) {
        x = ((rand_r(&seed) % 1000000)/500000.0)-1;
        y = ((rand_r(&seed) % 1000000)/500000.0)-1;
        d = ((x*x) + (y*y));
        if (d <= 1) in+=1;
    }

    //Use atomic mutex
    __sync_fetch_and_add(&sum, in);
}


double monte_carlo_pi(unsigned int tosses, int n_threads)
{
    int i;
    //Create #n threads
    pthread_t *threads = malloc(n_threads* sizeof(pthread_t));

    unsigned int thread_tosses_division = tosses/n_threads;
    unsigned int thread_tosses_residual = tosses/n_threads;
    unsigned int thread_tosses;

    //Call workers per thread
    for (i = 0; i < n_threads; ++i)
    {
        thread_tosses = thread_tosses_division + (i < thread_tosses_residual ? 1:0);
        pthread_create(&threads[i], NULL, monte_carlo_pi_worker, (void*)thread_tosses);
    }

    //Join to threads
    for (i = 0; i < n_threads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    //Free memory used by threads
    free(threads);

    return (4*sum/((double)tosses));
}

int main(void) {
    int n_threads;
    unsigned int tosses;
    double pi;
    long long unsigned int duration;
    struct timeval start, end;

    //Scan data
    scanf("%d %u",&n_threads, &tosses);

    //Call function
    gettimeofday(&start, NULL);
    pi = monte_carlo_pi(tosses, n_threads);
    gettimeofday(&end, NULL);

    //Calculation of duration
    duration = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

    printf("%lf\n%llu",pi,duration);

    return 0;
}
