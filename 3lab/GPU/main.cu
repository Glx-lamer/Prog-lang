#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16

__global__ void SumRow(float* DevMatrix, int i, int diag, float k, int n) {
	int j = diag + (blockIdx.x * blockDim.x + threadIdx.x);

	if (j < n) {
		DevMatrix[i * n + j] += k * DevMatrix[diag * n + j];
	}
}

__global__ void CompK(float* DevMatrix, int diag, int n) {
	dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	int i = diag + 1 + (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < n) {
		float k = (-1.0) * DevMatrix[i * n + diag] / DevMatrix[diag * n + diag];
		__syncthreads();
		SumRow<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, i, diag, k, n);	
	}
}

__global__ void CompDet(float* DevMatrix, int n, float* det) {
    __shared__ float PartDet[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    PartDet[tid] = 1.0;
    if (idx < n) {
        PartDet[tid] = DevMatrix[idx * n + idx];
    }

    __syncthreads();

    for (int interval = blockDim.x / 2; interval > 0; interval /= 2) {
        if (tid < interval) {
            PartDet[tid] *= PartDet[tid + interval];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicExch(det, *det * PartDet[0]);
    }
}

void Gauss(float* DevMatrix, int n, float* det) {
	for (int diag = 0; diag < n - 1; diag++) {

		int i = n - (diag + 1);
		dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
		dim3 blocksPerGrid((i + threadsPerBlock.x - 1) / threadsPerBlock.x);

		CompK<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, diag, n); 
	}

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
    CompDet<<<blocksPerGrid, threadsPerBlock>>>(DevMatrix, n, det);
}


void PrintMatrix(float* HstMatrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%8.2f", HstMatrix[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void FillMatrix(float* DevMatrix, int n) {
	curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());
	curandGenerateUniform(gen, DevMatrix, (n * n));

	curandDestroyGenerator(gen);
}


int main(void) {
	int n;

	printf("Enter size of square matrix, which will be transformed into a triangular >>> ");
	fflush(stdin);
	fscanf(stdin, "%d", &n);

	float* HstMatrix;
	float* DevMatrix;

	HstMatrix = (float*)calloc(n * n, sizeof(float));
	cudaMalloc((void**)&DevMatrix, sizeof(float) * n * n);

    FillMatrix(DevMatrix, n);

	if (n <= 20) {
		cudaMemcpy(HstMatrix, DevMatrix, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
		printf("Generated matrix:\n");
		PrintMatrix(HstMatrix, n);
	}

    float HstDet = 1.0;
    float *DevDet;

	cudaMalloc((void**)&DevDet, sizeof(float));
    cudaMemcpy(DevDet, &HstDet, sizeof(float), cudaMemcpyHostToDevice);

	clock_t start = clock();
	Gauss(DevMatrix, n, DevDet);
	clock_t end = clock();

    cudaMemcpy(&HstDet, DevDet, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(DevDet);

	cudaMemcpy(HstMatrix, DevMatrix, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	
	if (n <= 20) {
        cudaMemcpy(HstMatrix, DevMatrix, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
		printf("Transformed matrix:\n");
		PrintMatrix(HstMatrix, n);
	}

    printf("Determenator of matrix - %f\n", HstDet);
	float ms_duration = (float)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("Time to execute - %f ms\n", ms_duration);

	free(HstMatrix);
	cudaFree(DevMatrix);

	return 0;
}
