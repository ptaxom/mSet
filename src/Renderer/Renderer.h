#ifndef INC_RENDERER
#define INC_RENDERER

#include <complex>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace Render
{

using sInt = signed int;
using byte = signed char;
using std::complex;

// template <class T>
// __global__ void kernel(const T x_start, const T y_start, const T dx, const T dy,       //Computed params
//                   const sInt width_, const sInt height_, const sInt max_deep,     //Input params
//                   byte* output
// )
// {
//     int x_index = threadIdx.x + blockDim.x * blockIdx.x;
//     int y_index = threadIdx.y + blockDim.y * blockIdx.y;
//     if (x_index >= width_ || y_index >= height_)        //out of bounds
//         return;

//     T curX = x_start - dx * x_index;
//     T curY = y_start - dy * y_index;
// }

template <class T>
class Renderer
{
public:
  Renderer(complex<T> center, sInt width, sInt height, sInt max_iterations, T zoom = 1.0) : center_(center), zoom_(zoom), width_(width), height_(height), max_iterations_(max_iterations)
  {
  }

  byte *render()
  {

    //Start time measure
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //new deltas based on basic ratio
    T dx = 3.0 / zoom_ / (T)width_;
    T dy = -2.0 / zoom_ / (T)height_;

    T x_start = center_.real() - dx * (width_ / 2);
    T y_start = center_.imag() - dy * (height_ / 2);

    //Kernel preparation

    //Dimenstions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(width_ / threads.x + 1, height_ / threads.y + 1);

    //Memory allocation
    size_t memory_size = width_ * height_ * sizeof(byte);
    byte *devicePixels = nullptr;
    cudaMalloc((void **)&devicePixels, memory_size);

    //Kernel call

    //Host memory allocation
    byte *pixels = (byte *)malloc(memory_size);
    cudaMemcpy(pixels, devicePixels, memory_size, cudaMemcpyDeviceToHost);
    cudaFree(devicePixels);

    //End time measure
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    if (isLogging)
      printf("Time spent executing by the GPU: %.2f millseconds\n", gpuTime);

    return pixels;
  }

private:
  complex<T> center_;
  T zoom_;

  sInt width_;
  sInt height_;
  sInt max_iterations_;

  byte BLOCK_SIZE = 32;
  bool isLogging = false;
};

}; // namespace Render

#endif