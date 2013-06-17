#ifndef GUPPY_UTIL_H
#define GUPPY_UTIL_H

#include <string>

struct VMException {
  std::string what;
  std::string file;
  int line;

  VMException(std::string what, std::string file, int line) {
    this->what = what;
    this->file = file;
    this->line = line;
  }
};

static inline int divup(int a, int b) {
  return (int) ceil(float(a) / float(b));
}

void check_cuda(const char* file, int line);

#define CHECK_CUDA() try {\
  check_cuda(__FILE__, __LINE__);\
} catch (VMException& vm) {\
  fprintf(stderr, "VM Exception: %s at %s:%d\n", vm.what.c_str(), vm.file.c_str(), vm.line);\
  abort();\
}

static inline double Now() {
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
