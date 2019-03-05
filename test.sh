#!/usr/bin/env bash
nvcc src/main.cpp src/Renderer/Renderer.h src/Renderer/cuFiles/Renderer.cu src/Renderer/cuFiles/Renderer.cuh  -o mSSet