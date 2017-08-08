#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double monte_carlo_pi(unsigned int n) {
    long long unsigned int in = 0, i;
    double x, y, d;

    for (i = 0; i < n; i++) {
        x = ((rand() % 1000000)/500000.0)-1;
        y = ((rand() % 1000000)/500000.0)-1;
        d = ((x*x) + (y*y));
        if (d <= 1) in+=1;
    }

    return (4*in/((double)n));
}

int main(void) {
    int size;
    unsigned int n;
    double pi;
    long unsigned int duration;
    struct timeval start, end;

    scanf("%d %u",&size, &n);

    srand(time(NULL));

    gettimeofday(&start, NULL);
    pi = monte_carlo_pi(n);
    gettimeofday(&end, NULL);

    duration = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

    printf("%lf\n%lu\n",pi,duration);

    return 0;
}
