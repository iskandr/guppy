#include "util.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

void check_cuda(const char* file, int line) {
  cudaError_t code = cudaPeekAtLastError();
  if (code != 0) {
    char buf[1024];
    sprintf(buf, "Cuda error: %s\n", cudaGetErrorString(code));
    throw VMException(buf, file, line);
  }
}
