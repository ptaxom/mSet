#ifndef INC_RENDERER_CUH
#define INC_RENDERER_CUH

#include <cuda_runtime.h>
#include <stdlib.h>
#include <complex>

namespace Render2{

    using sInt = signed int;
    using byte = signed char;
    using std::complex;

    template<class T>
    byte* cudaWrapper(const T x_start, const T y_start, const T dx, const T dy,       //Computed params
                      const sInt width_, const sInt height_, const sInt max_deep, byte BLOCK_SIZE,    //Input params
                      float *time);


};

#endif