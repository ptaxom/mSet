#include <iostream>
#include "Renderer/Renderer.h"
#include <complex>

int main(int, char **)
{
    Render::Renderer<double>  r(std::complex<double>(0,0), 1200, 768, 255, 1.0);
    return 0;
}
