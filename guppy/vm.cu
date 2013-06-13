#include <stdio.h>
#include <time.h>
#include <vector>
#include <stdint.h>
#include <math.h>

//#include <stdexcept>

/*
 #define BYTECODE_OP static inline __device__

 BYTECODE_OP void load_slice() {

 }

 BYTECODE_OP void add(void* a, void *b) {

 }
 */

#ifndef __NVCC__
#define __global__
#define __kernel__
#define __device__
#define __shared__
#endif

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

enum OP_CODE {
  LOAD_SLICE,
  STORE_SLICE,    // load slice of global arrays into shared vector
  LOAD_SCALAR,
  STORE_SCALAR, // distribute scalar across elements of shared vector
  ADD,
  SUB,
  MUL,
  DIV,        // arithmetic between shared vectors
  BAD
};

struct Op {
  uint32_t code :8;
  uint32_t x :8;
  uint32_t y :8;
  uint32_t z :8;
};

Op make_op(OP_CODE code, int x, int y, int z) {
  Op op = { code, x, y, z };
  return op;
}

struct Program {
  std::vector<Op> _ops;
  Op* _gpu_ptr;

  Program& Add(int x, int y, int z) {
    _ops.push_back(make_op(ADD, x, y, z));
    return *this;
  }
  Program& LoadSlice(int src, int dst) {
    _ops.push_back(make_op(LOAD_SLICE, src, dst, 0));
    return *this;
  }
  Program& StoreSlice(int src, int dst) {
    _ops.push_back(make_op(STORE_SLICE, src, dst, 0));
    return *this;
  }

  int size() {
    return _ops.size();
  }

  int nbytes() {
    return sizeof(Op) * this->size();
  }

  Op* host_ptr() {
    return &_ops[0];
  }
  Op* to_gpu() {
    if (_gpu_ptr) {
      return _gpu_ptr;
    }
    cudaMalloc(&_gpu_ptr, this->nbytes());
    cudaMemcpy(_gpu_ptr, this->host_ptr(), this->nbytes(), cudaMemcpyHostToDevice);
    return _gpu_ptr;
  }

  Program() :
      _gpu_ptr(NULL) {
  }
  ~Program() {
    if (_gpu_ptr) {
      cudaFree(_gpu_ptr);
    }
  }
};

struct Vec {
  int _n;
  int _nbytes;
  float* _host_data;
  float* _gpu_data;
  bool _host_dirty;
  bool _gpu_dirty;

  void init(int n) {
    _n = n;
    _nbytes = sizeof(float) * n;
    cudaMallocHost(&_host_data, this->_nbytes);
    cudaMalloc(&_gpu_data, this->_nbytes);
    _host_dirty = false;
    _gpu_dirty = true;
  }

  Vec(int n) {
    this->init(n);
  }

  Vec(int n, float fill_value) {
    this->init(n);
    for (int i = 0; i < n; ++i) {
      _host_data[i] = fill_value;
    }
  }

  float* get_gpu_data() {
    if (_gpu_dirty) {
      this->copy_to_gpu();
    }
    _host_dirty = true;
    _gpu_dirty = false;
    return _gpu_data;
  }

  float* get_host_data() {
    if (_host_dirty) {
      this->copy_to_host();
    }
    _gpu_dirty = true;
    _host_dirty = false;
    return _host_data;
  }

  void copy_to_host() {
    cudaMemcpy(this->_host_data, this->_gpu_data, this->_nbytes, cudaMemcpyDeviceToHost);
  }

  void copy_to_gpu() {
    cudaMemcpy(this->_gpu_data, this->_host_data, this->_nbytes, cudaMemcpyHostToDevice);
  }

  ~Vec() {
    cudaFree(_gpu_data);
    cudaFreeHost(_host_data);
  }

};

static const int kThreadsX = 8;
static const int kThreadsY = 8;
static const int kOpsPerThread = 8;

static const int kThreadsPerBlock = kThreadsX * kThreadsY;

static const int kRegisterWidth = kThreadsPerBlock * kOpsPerThread;
static const int kNumRegisters = 3;
static const int kProgramSize = 32;

__global__ void run(Op* program,
                    long n_ops,
                    float** values,
                    long n_args,
                    float* constants,
                    long n_consts) {
  __shared__ float registers[kNumRegisters][kRegisterWidth];

  const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
  const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_idx = block_offset * kRegisterWidth;

  for (int pc = 0; pc < n_ops; ++pc) {
    Op op = program[pc];
    switch (op.code) {
    case LOAD_SLICE: {
      float* reg = &registers[op.y][local_idx * kOpsPerThread];
      const float* src = values[op.x] + block_idx + (local_idx * kOpsPerThread);
      for (int i = 0; i < kOpsPerThread; ++i) {
        reg[i] = src[i];
      }
      break;
    }

    case STORE_SLICE: {
      float* reg = &registers[op.y][local_idx * kOpsPerThread];
      float* dst = values[op.x] + block_idx + (local_idx * kOpsPerThread);
      for (int i = 0; i < kOpsPerThread; ++i) {
        dst[i] = reg[i];
      }
      break;
    }

    case ADD: {
      const float* a = &registers[op.x][local_idx * kOpsPerThread];
      const float* b = &registers[op.y][local_idx * kOpsPerThread];
      float *c = &registers[op.z][local_idx * kOpsPerThread];
      for (int i = 0; i < kOpsPerThread; ++i) {
        c[i] = a[i] + b[i];
      }
      break;
    }
    }
  }
}

int main(int argc, const char** argv) {
  int N = 2 << 24;

  Vec a(N, 1.0);
  Vec b(N, 2.0);
  Vec c(N);

  const int n_values = 3;
  float* h_values[n_values];
  h_values[0] = a.get_gpu_data();
  h_values[1] = b.get_gpu_data();
  h_values[2] = c.get_gpu_data();

  float** d_values;
  cudaMalloc(&d_values, sizeof(float*) * n_values);
  cudaMemcpy(d_values, h_values, sizeof(float*) * n_values, cudaMemcpyHostToDevice);

  Program h_program;

  h_program.LoadSlice(0, 0).LoadSlice(1, 1).Add(0, 1, 2).StoreSlice(2, 2);

  for (int i = 1; i <= N; i *= 2) {
    int total_blocks = divup(i, kThreadsPerBlock * kOpsPerThread);
    dim3 blocks;
    blocks.x = int(ceil(sqrt(total_blocks)));
    blocks.y = int(ceil(sqrt(total_blocks)));
    blocks.z = 1;

    dim3 threads;
    threads.x = kThreadsX;
    threads.y = kThreadsY;
    threads.z = 1;

    fprintf(stderr, "%d %d %d; %d %d %d\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y,
            threads.z);
    double st = Now();
    run<<<blocks, threads>>>(h_program.to_gpu(), h_program.size(), d_values, n_values, 0, 0);
    cudaDeviceSynchronize();
    CHECK_CUDA();
    double ed = Now();
    fprintf(stderr, "%d elements in %.5f seconds; %.5f GFLOPS\n", i, ed - st, i * 1e-9 / (ed - st));
  }

  float* ad = a.get_host_data();
  printf("%f %f %f\n", ad[0], ad[10], ad[N - 200]);
  float* bd = b.get_host_data();
  printf("%f %f %f\n", bd[0], bd[10], bd[N - 200]);
  float* cd = c.get_host_data();
  for (int i = 0; i < min(1024, N); ++i) {
    printf("%.0f ", cd[i]);
    if (i % 64 == 63) {
      printf("\n");
    }
  }

  for (int i = 0; i < N; ++i) {
    if (cd[i] == 0) {
      printf("ZERO at %d\n", i);
      break;
    }
  }
  return 0;
}
