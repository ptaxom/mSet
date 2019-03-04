cmake_minimum_required(VERSION 3.0.0)
#find_package(CUDA QUIET REQUIRED)

project(mSet VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)

include(CTest)
enable_testing()

#set_target_properties(mSet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mSet src/main.cu)

set_property(TARGET mSet 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
