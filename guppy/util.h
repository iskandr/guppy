#ifndef GUPPY_UTIL_H
#define GUPPY_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

static inline int divup(int a, int b) {
  return (int) ceil(float(a) / float(b));
}

static inline void check_cuda(const char* file, int line) {
  cudaError_t code = cudaPeekAtLastError();
  if (code != 0) {
    char buf[1024];
    sprintf(buf, "Cuda error at %s:%d :: %s\n", file, line, cudaGetErrorString(code));
    fprintf(stderr, "%s", buf);
    abort();
    // throw std::runtime_error(buf);
  }
}

#define CHECK_CUDA() check_cuda(__FILE__, __LINE__)

double Now() {
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return tp.tv_sec + 1e-9 * tp.tv_nsec;
}

#define TIMEOP(op)\
{\
  double st = Now();\
  op;\
  double ed = Now();\
  fprintf(stderr, "%s finished in %.f seconds.\n", #op, end - start);\
}

#endif
