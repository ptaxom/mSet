cmake_minimum_required(VERSION 3.0.0)


project(mSet VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA QUIET REQUIRED)

include(CTest)
enable_testing()
file(GLOB_RECURSE SRC src/*.cpp src/*.h src/*.cu src/*.cuh )
message(STATUS "Source files: " ${SRC})



add_executable(mSet ${SRC})

set_property(TARGET mSet PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(mSet PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)