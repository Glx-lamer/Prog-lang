#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16

/* Kernel for transforming matrix */
__global__ void SumRow(double* DevMatrix, int i, int diag, double k, int n) {
    int j = diag + (blockIdx.x * blockDim.x + threadIdx.x);

    if (j < n) {
        DevMatrix[i * n + j] = fma(k, DevMatrix[diag * n + j], DevMatrix[i * n + j]);
    }
}

/* Kernel for transforming matrix */
__global__ void CompK(double* DevMatrix, int diag, int n) {
    int i = diag + 1 + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < n) {
        double k = (-1.0) * DevMatrix[i * n + diag] / DevMatrix[diag * n + diag];
        dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
        dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
        SumRow<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, i, diag, k, n);
    }
}

/* Main kernel for computing determinator */
__global__ void CompDet(double* DevMatrix, int n, double* det) {
    __shared__ double PartDet[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x; /* Index like in 2-dimension list */
    int tid = threadIdx.x; /* Thread index */

    PartDet[tid] = 1.0;
    if (idx < n) {
        PartDet[tid] = DevMatrix[idx * n + idx];
    }

    __syncthreads();

    /* Reduction */
    for (int interval = blockDim.x / 2; interval > 0; interval /= 2) {
        if (tid < interval) {
            PartDet[tid] += PartDet[tid + interval];
        }
        __syncthreads();
    }

    if (tid == 0) {
        det[blockIdx.x] = PartDet[0];
    }
}

/* Main computing func */
void Gauss(double* DevMatrix, int n, double* det) {
    for (int diag = 0; diag < n - 1; diag++) {
        /* Computing needed quantity of blocks */
        int i = n - (diag + 1);
        dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
        dim3 blocksPerGrid((i + threadsPerBlock.x - 1) / threadsPerBlock.x);

        /* Transforming usual matrix to triangle kernel*/
        CompK<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, diag, n);
    }

    /* Computing needed quantity of blocks */
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    double* DevPatDet;
    cudaMalloc((void**)&DevPatDet, sizeof(double) * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE));

    /* Computing determinator kernel */
    CompDet<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, n, DevPatDet);

    /* Final reduction on device */
    double* HstPatDet = (double*)malloc(sizeof(double) * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    cudaMemcpy(HstPatDet, DevPatDet, sizeof(double) * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE), cudaMemcpyDeviceToHost);

    double ans = 1.0;
    for (int i = 0; i < ((n + BLOCK_SIZE - 1) / BLOCK_SIZE); i++) {
        ans += HstPatDet[i];
    }

    cudaMemcpy(det, &ans, sizeof(double), cudaMemcpyHostToDevice);

    cudaFree(DevPatDet);
    free(HstPatDet);
}

void PrintMatrix(double* HstMatrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f", HstMatrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void FileToHstMatrix(double* HstMatrix, int n) {
    FILE* file;
    file = fopen("../test/cpu_matrix.txt", "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(file, "%lf", &HstMatrix[i * n + j]);
        }
    }
    fclose(file);
}

void Comparision(double HstDet) {
    double Det;
    FILE* file;
    file = fopen("../test/cpu_out.txt", "r");
    fscanf(file, "%lf", &Det);
    fclose(file);
    printf(">>> Expected determinant: %lf\n", Det);
}

void FillMatrix(double* DevMatrix, int n) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());
    curandGenerateUniformDouble(gen, DevMatrix, (n * n));

    curandDestroyGenerator(gen);
}

int main(void) {
    int n;
    int mode;

    printf(">>> Mode:\n > 1 for usual <\n > 2 for test <\n>>> ");
    fflush(stdin);
    fscanf(stdin, "%d", &mode);
    printf(">>> Matrix rank - ");
    fflush(stdin);
    fscanf(stdin, "%d", &n);

    /* Host and device matrix */
    double* HstMatrix;
    double* DevMatrix;

    /* Init host matrix */
    HstMatrix = (double*)calloc(n * n, sizeof(double));
    if (mode == 2) {
        FileToHstMatrix(HstMatrix, n);
    }

    /* Init device matrix */
    cudaMalloc((void**)&DevMatrix, sizeof(double) * n * n);

    /* Fill device matrix from file with host matrix or randomly fill device matrix */
    if (mode == 1) {
        FillMatrix(DevMatrix, n);
    }
    else {
        cudaMemcpy(DevMatrix, HstMatrix, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    }

    if (n <= 10) {
        cudaMemcpy(HstMatrix, DevMatrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
        printf(">>> Generated matrix:\n");
        PrintMatrix(HstMatrix, n);
    }

    /* Host and device determinators */
    double HstDet = 1.0;
    double* DevDet;

    /* Init device determinator */
    cudaMalloc((void**)&DevDet, sizeof(double));
    cudaMemcpy(DevDet, &HstDet, sizeof(double), cudaMemcpyHostToDevice);

    clock_t start = clock();
    /* Main func */
    Gauss(DevMatrix, n, DevDet);
    clock_t end = clock();

    cudaMemcpy(&HstDet, DevDet, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(DevDet);

    cudaMemcpy(HstMatrix, DevMatrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    if (mode == 2) {
        Comparision(HstDet);
    }

    printf(">>> Matrix determinator: %lf\n", HstDet);
    printf(">>> Time to calc - %f msec\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(HstMatrix);
    cudaFree(DevMatrix);

    return 0;
}