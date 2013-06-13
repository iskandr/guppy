#include <stdio.h>
#include <time.h>
#include <vector>
#include <math.h>

#include "bytecode.h"
#include "vec.h"
#include "util.h"

static const int kThreadsX = 4; // 16;
static const int kThreadsY = 4; // 16;
static const int kOpsPerThread = 8;

static const int kThreadsPerBlock = kThreadsX * kThreadsY;

static const int kRegisterWidth = kThreadsPerBlock * kOpsPerThread;
static const int kNumRegisters = 4;
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
  const int global_idx = block_idx + (local_idx * kOpsPerThread);
  const int end_i = min(kOpsPerThread, (2 << 24) - global_idx);

  for (int pc = 0; pc < n_ops; ++pc) {
    Op op = program[pc];
    switch (op.code) {
    case LOAD_SLICE: {
      float* reg = &registers[op.y][local_idx * kOpsPerThread];
      const float* src = &values[op.x][global_idx];
      for (int i = 0; i < end_i; ++i) {
        reg[i] = src[i];
      }
      break;
    }

    case STORE_SLICE: {
      float* reg = &registers[op.y][local_idx * kOpsPerThread];
      float* dst = &values[op.x][global_idx];
      for (int i = 0; i < end_i; ++i) {
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

//  for (int i = 1; i <= N; i *= 2) {
  int i = N;
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
//  }

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
