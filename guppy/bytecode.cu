#include "bytecode.h"

Program::~Program() {
  if (_gpu_ptr) {
    cudaFree(_gpu_ptr);
  }
}

void* Program::to_gpu() {
  if (_gpu_ptr) {
    return _gpu_ptr;
  }
  cudaMalloc(&_gpu_ptr, nbytes());
  cudaMemcpy(_gpu_ptr, &_ops[0], nbytes(), cudaMemcpyHostToDevice);
  return _gpu_ptr;
}

