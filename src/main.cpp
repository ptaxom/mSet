#include <iostream>
#include "Renderer/Renderer.h"
#include <complex>

int main(int, char **)
{

    Render::Renderer<double>  r(std::complex<double>(0,0), 1200, 768, 255, 1.0);
    auto x = r.render();
    for(int i = 0; i < 100; i++)
        std::cout << (int)x[i] << " ";
    return 0;
}