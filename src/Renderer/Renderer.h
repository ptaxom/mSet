#ifndef INC_RENDERER
#define INC_RENDERER

#include <complex>
#include "cuFiles/Renderer.cu"

namespace Render
{

using sInt = signed int;
using byte = signed char;
using std::complex;


template <class T>
class Renderer
{
public:
  Renderer(complex<T> center, sInt width, sInt height, sInt max_iterations, T zoom = 1.0) : center_(center), zoom_(zoom), width_(width), height_(height), max_iterations_(max_iterations)
  {
    std::cout << "dbg";
  }

  byte *render()
  {
    std::cout << "render"; 
    //new deltas based on basic ratio
    T dx = 3.0 / zoom_ / (T)width_;
    T dy = -2.0 / zoom_ / (T)height_;

    T x_start = center_.real() - dx * (width_ / 2);
    T y_start = center_.imag() - dy * (height_ / 2);

    float gpuTime = -1.0f;
    byte *pixels = nullptr;   

    pixels = Render2::cudaWrapper<T>(x_start, y_start, dx, dy, width_, height_, max_iterations_, BLOCK_SIZE, &gpuTime);

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
  bool isLogging = true;
};

}; // namespace Render

#endif