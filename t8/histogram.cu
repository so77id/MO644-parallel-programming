#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define HISTOGRAM_SIZE 64
#define TILE_WITDH 16

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}


__global__ void add_to_hist(PPMImage *d_image, float* hist) {
    __shared__ float private_hist[HISTOGRAM_SIZE];
    float size = d_image->y*d_image->x*1.0;


    if(threadIdx.x * TILE_WITDH + threadIdx.y < HISTOGRAM_SIZE) private_hist[threadIdx.x * TILE_WITDH + threadIdx.y] = 0;
    __syncthreads();

    // Get col
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // Get row
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // Get index
    int index = row * d_image->x + col;
    if((row < d_image->y && col < d_image->x) && (index < d_image->x*d_image->y)) {
        // Sum
        atomicAdd(&(private_hist[16*d_image->data[index].red + 4 * d_image->data[index].green + d_image->data[index].blue]), 1);

    }

    __syncthreads();
    if(threadIdx.x * TILE_WITDH + threadIdx.y < HISTOGRAM_SIZE) {
        atomicAdd(&(hist[threadIdx.x * TILE_WITDH + threadIdx.y]), (private_hist[threadIdx.x * TILE_WITDH + threadIdx.y]/size));
    }
}



void Histogram(PPMImage *image, float *h) {

    //Init variables;
    int i;
    unsigned int rows, cols, img_size;
    PPMImage *d_image;
    PPMPixel *d_pixels;
    float *d_hist;

    // CUDA TIMERS
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tbuffer, tenviar, tk, treceber, ttotal;
    */

    // Get data
    cols = image->x;
    rows = image->y;
    img_size = cols * rows;

    //Preprocess data
    for (i = 0; i < img_size; i++) {
        image->data[i].red = floor((image->data[i].red * 4) / 256);
        image->data[i].blue = floor((image->data[i].blue * 4) / 256);
        image->data[i].green = floor((image->data[i].green * 4) / 256);
    }


    //cudaEventRecord(start);
    // Alloc structure to devise
    cudaMalloc((void **)&d_image, sizeof(PPMImage));

    // Alloc image to devise
    cudaMalloc((void **)&d_pixels, sizeof(PPMPixel) * img_size);

    //alloc histogram to devise
    cudaMalloc((void **)&d_hist, HISTOGRAM_SIZE*sizeof(float));

    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tbuffer, start, stop);
    printf("Alloc Time: %f ", tbuffer);
    */


    //cudaEventRecord(start);
    // cpy stucture to devise
    cudaMemcpy(d_image, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixels, image->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_image->data), &d_pixels, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

    // cpy histogram to devise
    cudaMemcpy(d_hist, h, HISTOGRAM_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tenviar, start, stop);
    printf("copy Time: %f ", tenviar);
    */


    //Init dimGrid and dimBlocks
    //cudaEventRecord(start);
    dim3 dimGrid(ceil((float)cols / TILE_WITDH), ceil((float)rows / TILE_WITDH), 1);
    dim3 dimBlock(TILE_WITDH, TILE_WITDH, 1);

    // Call function
    add_to_hist<<<dimGrid, dimBlock>>>(d_image, d_hist);
    /*
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tk, start, stop);
    printf("CUDA KERNEL: %f\n", tk);
    */


    //cudaEventRecord(start);
    // Copy result to local array
    cudaMemcpy(h, d_hist, HISTOGRAM_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&treceber, start, stop);
    printf("receber Time: %f ", treceber);

    ttotal = tbuffer + tenviar + tk + treceber;
    printf("Total: %f\n", ttotal);
    */

    //Free memory
    cudaFree(d_image);
    cudaFree(d_pixels);
    cudaFree(d_hist);
    /*
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    double t_start, t_end;
    int i;
    char *filename = argv[1];

    PPMImage *image = readPPM(filename);

    float *h = (float*)malloc(sizeof(float) * 64);

    //Inicializar h
    for(i=0; i < 64; i++) h[i] = 0.0;

    t_start = rtclock();
    Histogram(image, h);
    t_end = rtclock();

    for (i = 0; i < 64; i++){
        printf("%0.3f ", h[i]);
    }

    printf("\n");
    fprintf(stdout, "%0.6lf\n", t_end - t_start);
    free(h);

    return (0);
}



/*

ts = tempo_serial
tbuffer = tempo_GPU_criar_buffer
tenviar = tempo_GPU_offload_enviar
tk = tempo_kernel
treceber = tempo_GPU_offload_receber
ttotal = GPU_total
speedup = speedup (tempo_serial / GPU_total).


arqX |    ts      |   tbuffer  |   tenviar  |      tk    |  treceber   |  GPU_total   | speedup |
-------------------------------------------------------------------------------------------------
arq1 | 0.320540 s | 1.62240 ms | 0.80217 ms | 0.67052 ms |  0.02444 ms |  3.11955 ms  | 102.751 |
-------------------------------------------------------------------------------------------------
arq2 | 0.585691 s | 1.29168 ms | 1.07040 ms | 1.83283 ms |  0.02112 ms |  4.21603 ms  | 138.920 |
-------------------------------------------------------------------------------------------------
arq3 | 1.676812 s | 1.30262 ms | 3.99257 ms | 7.15699 ms |  0.02096 ms | 12.47315 ms  | 134.433 |
-------------------------------------------------------------------------------------------------


*/
