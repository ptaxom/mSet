#include "Renderer.cuh"



using sInt = signed int;
using byte = signed char;

template <class T>
__global__ void kernel(const T x_start, const T y_start, const T dx, const T dy,       //Computed params
                  const sInt width_, const sInt height_, const sInt max_deep,     //Input params
                  byte* output
)
{
    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;
    if (x_index >= width_ || y_index >= height_)        //out of bounds
        return;

    T curX = x_start - dx * x_index;
    T curY = y_start - dy * y_index;

    T zRe = 0,    zIm = 0;  
    
    T sqrRe = zRe * zRe,
      sqrIm = zIm * zIm;  

    T buffer;

    sInt iteration = 0;
    while (iteration < max_deep && sqrRe + sqrIm < 4)
    {
        buffer = sqrRe - sqrIm + curX;              //Z = Z * Z + C with optimisations
        zIm = 2 * zRe * zIm + curY;
        zRe = buffer;

        sqrRe = zRe * zRe;
        sqrIm = zIm * zIm;
    }
    
    byte value = (byte)(1.0f * iteration / (float)max_deep * 255.f);         // normalize value in range [0; 255]
    output[y_index * width_ + x_index] = value;


}


template<class T>
byte* cudaWrapper(const T x_start, const T y_start, const T dx, const T dy,       //Computed params
                      const sInt width_, const sInt height_, const sInt max_deep, byte BLOCK_SIZE,    //Input params
                      float *time)
{
    //Start time measure
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    //Kernel preparation

    //Dimenstions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(width_ / threads.x + 1, height_ / threads.y + 1);

    //Memory allocation
    size_t memory_size = width_ * height_ * sizeof(byte);
    byte *devicePixels = nullptr;
    cudaMalloc((void **)&devicePixels, memory_size);

    //Kernel call


    kernel<<<threads, blocks>>>(x_start, y_start, dx, dy, width_, height_, max_deep, devicePixels);
    
    //Host memory allocation
    byte *pixels = (byte *)malloc(memory_size);
    cudaMemcpy(pixels, devicePixels, memory_size, cudaMemcpyDeviceToHost);
    cudaFree(devicePixels);



    //End time measure
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);
    return pixels;

}
