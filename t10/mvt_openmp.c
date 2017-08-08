#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>

#define GPU 1

#ifdef MEDIUM
  #define N 2048
#elif LARGE
  #define N 4096
#elif EXTRALARGE
  #define N 8192
#endif


double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void init_array(float *A,float *x1,float *x2,float *y1,float *y2){
  int i,j;
  for(i = 0 ; i < N ; i++){
    x1[i] = ((float)i)/N;
    x2[i] = ((float)i + 1)/N;
    y1[i] = ((float)i + 3)/N;
    y2[i] = ((float)i + 4)/N;
    for(j = 0 ; j < N ; j++)
      A[i*N + j] = ((float)i*j)/N;
  }
  return;
}


void mvt_gpu (float *a,float *x1,float *x2,float *y1,float *y2) {
  int i,j;

  #pragma omp target device(GPU) map(to: a[:N*N], y1[:N]) map(tofrom: x1[:N])
  #pragma omp parallel for collapse(GPU)
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      x1[i] += a[i*N + j] * y1[j];

  #pragma omp target device(GPU) map(to: a[:N*N], y2[:N]) map(tofrom: x2[:N])
  #pragma omp parallel for collapse(GPU)
  for(i = 0; i < N ; i++)
    for(j = 0 ; j < N ; j++)
      x2[i] += a[j*N + i] * y2[j];

  return;
}

int main(){
  double t_start, t_end;

  float *A,*x1,*x2,*y1,*y2;
  A = (float*)malloc( N * N * sizeof(float) );
  x1 = (float*)malloc( N * sizeof(float) );
  x2 = (float*)malloc( N * sizeof(float) );
  y1 = (float*)malloc( N * sizeof(float) );
  y2 = (float*)malloc( N * sizeof(float) );

  init_array(A,x1,x2,y1,y2);

  t_start = rtclock();
  mvt_gpu( A , x1 , x2 , y1 , y2 );
  t_end = rtclock();

  float m = 0 , n = 0;
  for(int i = 0 ; i < N ; i++)
    m += x1[i] , n += x2[i];

  fprintf(stdout, "%0.4lf  %0.4lf\n", m, n);
  fprintf(stdout, "%0.4lf\n", t_end - t_start);
}


/*

FLAG=none
--------------------------------------------------------------------------------
|     SIZE    | OPENMP T | SERIAL T | SPEEDUP || Offloading time | Kernel time |
--------------------------------------------------------------------------------
|     DMEDIUM |  0.0203  |   0.0306 |  1.5073 ||       5,391 ms  |    2,973 ms |
|      DLARGE |  0.0454  |   0.2059 |  4.5352 ||      22,693 ms  |    8,334 ms |
| DEXTRALARGE |  0.1281  |   0.9021 |  7.0421 ||      80,723 ms  |   25,330 ms |
--------------------------------------------------------------------------------

FLAG=tile
--------------------------------------------------------------------------------
|     SIZE    | OPENMP T | SERIAL T | SPEEDUP || Offloading time | Kernel time |
--------------------------------------------------------------------------------
|     DMEDIUM |  0.0199  |   0.0306 |  1.5376 ||       5,666 ms  |    2,977 ms |
|      DLARGE |  0.0430  |   0.2059 |  4.7883 ||      22,661 ms  |    8,329 ms |
| DEXTRALARGE |  0.1215  |   0.9021 |  7.4246 ||      79,338 ms  |   25,269 ms |
--------------------------------------------------------------------------------

FLAG=vectorize
--------------------------------------------------------------------------------
|     SIZE    | OPENMP T | SERIAL T | SPEEDUP || Offloading time | Kernel time |
--------------------------------------------------------------------------------
|     DMEDIUM |  0.0195  |   0.0306 |  1.5692 ||       6,526 ms  |    2,986 ms |
|      DLARGE |  0.0418  |   0.2059 |  4.9258 ||      22,015 ms  |    8,324 ms |
| DEXTRALARGE |  0.1154  |   0.9021 |  7.8171 ||      75,857 ms  |   25,133 ms |
--------------------------------------------------------------------------------

The speedups achieved are very similar by size, while more big is the size of data, more speedup is achieved, this because more calculations are parallelized in CUDA. The speedup differ very little between the compilation flags (none, tile and vectorize), but even so small improvements are seen between them, being none < tile < vectorize the order of improvement

The offloading and kernel times are correlated with the size of the data entry, while more big is the input data, more is the the loading time to device  (offloading) and the kernel's execution time. The offloading time will always be greater that the kernel time, because it takes longer to load all the data to device than to run the kernel function.
*/
