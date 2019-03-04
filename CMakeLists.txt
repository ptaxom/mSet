cmake_minimum_required(VERSION 3.0.0)

#CUDA
project(mSet VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)

include(CTest)
enable_testing()

file(GLOB_RECURSE SRC src/*.cpp src/*.h src/*.cu src/*.cuh )

#GTK
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})

ADD_DEFINITIONS(${GTK3_CFLAGS_OTHER})
#~~~~~~~~~

add_executable(mSet src/main.cpp src/Renderer/Renderer.h)

TARGET_LINK_LIBRARIES(mSet ${GTK3_LIBRARIES})

set_property(TARGET mSet 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
