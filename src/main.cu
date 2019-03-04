#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void matrixMult(const float *a, const float *b, int n, float *c)
{
    int bX = blockIdx.x;
    int bY = blockIdx.y;

    int tX = threadIdx.x;
    int tY = threadIdx.y;

    float sum = 0.0f;

    int indexOfA = n * BLOCK_SIZE * bY + n * tY;
    int indexOfB = BLOCK_SIZE * bX + tX;

    int indexOfC = n * BLOCK_SIZE * bY + BLOCK_SIZE * bX;

    for (int k = 0; k < n; k++)
        sum += a[indexOfA + k] * b[indexOfB + k * n];

    c[indexOfC + n * tY + tX] = sum;
}

void multiplyerGPU(const float *A, const float *B, int N, float *C)
{
    int numBytes = N * N * sizeof(float);

    float *deviceA = nullptr;
    float *deviceB = nullptr;
    float *deviceC = nullptr;

    cudaMalloc((void **)&deviceA, numBytes);
    cudaMalloc((void **)&deviceB, numBytes);
    cudaMalloc((void **)&deviceC, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMemcpy(deviceA, A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, numBytes, cudaMemcpyHostToDevice);

    matrixMult<<<blocks, threads>>>(deviceA, deviceB, N, deviceC);

    cudaMemcpy(C, deviceC, numBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("Matrix size = %dx%d\nkernel size = %d\n time spent executing by the GPU: %.2f millseconds\n", N, N, BLOCK_SIZE, gpuTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void printM(float *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            std::cout << arr[i * N + j] << " ";
        std::cout << std::endl;
    }
    std::cout << "\n\n========\n\n";
}

int main(int, char **)
{
    int N;
    std::cin >> N;
    int size = N * N;
    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];
    for (int i = 0; i < size; i++)
    {
        A[i] = (rand() % 1024) / 1024.f;
        B[i] = (rand() % 1024) / 1024.f;
    }
    multiplyerGPU(A, B, N, C);

    return 0;
}
