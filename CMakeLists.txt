cmake_minimum_required(VERSION 3.0.0)
project(mSet VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS "-std=c++0x -O2")
SET(CMAKE_CUDA_FLAGS "-arch=sm_30")

include(CTest)
enable_testing()

find_package(CUDA QUIET REQUIRED)

file(GLOB_RECURSE SRC src/* )

# list(REMOVE_ITEM SRC src/main.cpp)

# add_library(proj_lib STATIC ${SRC})
# target_compile_features(proj_lib PUBLIC cxx_std_11)
# set_target_properties( proj_lib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




cuda_add_executable(mSet ${SRC})

# set_property(TARGET Proj1 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(Proj1 PRIVATE proj)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

