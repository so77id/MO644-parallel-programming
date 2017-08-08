#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TW 16

__global__ void matrix_sum(int *C, int *A, int *B, int rows, int cols, int dim) {
    // Get col
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // Get row
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // Get index
    int index = row * cols + col;
    if((row < rows && col < cols) && (index < dim)) {
        // Sum
        C[index] = A[index] + B[index];
    }
}


int main()
{
    // Declaration
    int *A, *B, *C;
    int i, j;
    int *d_a, *d_b, *d_c;

    int rows, cols;

    //Scan values
    scanf("%d", &rows);
    scanf("%d", &cols);

    // Calculation of dimensions
    int dim = rows * cols;
    int size = dim * sizeof(int);

    // Alloc local arrays
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Alloc devise arrays
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Initialize arrays
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            A[i*cols+j] =  B[i*cols+j] = i+j;
        }
    }

    // Copy to devise arrays
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    //Init dimGrid and dimBlocks
    dim3 dimGrid(ceil((float)cols / TW), ceil((float)rows / TW), 1);
    dim3 dimBlock(TW, TW, 1);
    // Call function
    matrix_sum<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, rows, cols, dim);

    // Copy result to local array
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);


    long long int somador=0;

    // Obtain sum
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            somador+=C[i*cols+j];
        }
    }

    // print sum
    printf("%lli\n", somador);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return (0);
}

