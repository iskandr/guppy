#include <stdio.h>
#include <time.h>
#include <vector>
#include <math.h>


#include "bytecode.h"
#include "vec.h"
#include "util.h"

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
static const int kThreadsX = 8; // 16;
static const int kThreadsY = 4; // 16;

// seems to give slightly better performance than kOpsPerThread = 8
static const int kOpsPerThread = 9;

static const int kThreadsPerBlock = kThreadsX * kThreadsY;

static const int kRegisterWidth = kThreadsPerBlock * kOpsPerThread;

static const int kNumVecRegisters = 4;

static const int kNumIntRegisters = 10;
static const int kNumFloatRegisters = 10;

static const int kMaxProgramLength = 1000; 

#define PREFETCH_GPU_BYTECODE 

__global__ void run(char* program,
                    long program_nbytes,
                    float** values,
                    long n_args,
                    float* constants,
                    long n_consts) {

  
  // making vector slightly longer seems to minorly improve 
  // performance -- due to bank conflicts? 
  __shared__ float vectors[kNumVecRegisters][kRegisterWidth+1];

  __shared__ int   int_scalars[kNumIntRegisters];
  __shared__ float float_scalars[kNumFloatRegisters];
  

  const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
  const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_start_idx = block_offset * kRegisterWidth;
  const int global_idx = block_start_idx + (local_idx * kOpsPerThread);

 
  #ifdef PREFETCH_GPU_BYTECODE 
    /* preload program so that we don't make 
       repeated global memory requests 
    */  
    __shared__ char  cached_program[kMaxProgramLength];
    for (int i = local_idx; i < program_nbytes; i+=kThreadsPerBlock) {
      cached_program[i] = program[i];      
    }  
  #endif 
  // by convention, the first int register contains the global index
  int_scalars[BlockStart] = block_offset; 
  int_scalars[VecWidth] = kRegisterWidth;
  int_scalars[BlockEltStart] = block_offset * kRegisterWidth; 

  int pc = 0;
  Instruction* instr;
  while (pc < program_nbytes) {
    
    #ifdef PREFETCH_GPU_BYTECODE 
      instr = (Instruction*) &cached_program[pc];
    #else
      instr = (Instruction*) &program[pc]; 
    #endif 
    pc += instr->size;

    switch (instr->code) {
    case LoadVector::op_code: {
      LoadVector* load_slice = (LoadVector*) instr;
      
      float* reg = vectors[load_slice->target_vector]; 
      const float* src = values[load_slice->source_array];
      const int start = int_scalars[load_slice->start_idx] + local_idx;
      int nelts = int_scalars[load_slice->nelts];
      nelts = nelts <= kRegisterWidth ? nelts : kRegisterWidth; 
      for (int i = 0; i < nelts; i += kThreadsPerBlock) { 
        const float elt = src[start+i]; 
        reg[i] = elt;
      }
      break;
    }

    case StoreVector::op_code: {
      StoreVector* store_vector = (StoreVector*) instr;
      const float* reg = vectors[store_vector->source_vector];
      float* dst = values[store_vector->target_array];
      const int start = int_scalars[store_vector->start_idx] + local_idx;
      int nelts = int_scalars[store_vector->nelts];
      nelts = nelts <= kRegisterWidth ? nelts : kRegisterWidth; 
      for (int i = 0; i < nelts; i += kThreadsPerBlock) { 
        const float elt = reg[i]; 
        dst[i+start] = elt; 
      }
      break;
    }

    case Add::op_code: {
      Add* add = (Add*) instr;
      const float* a = vectors[add->arg1];
      const float* b = vectors[add->arg2];
      float *c = vectors[add->result];
      for (int i = local_idx; i < kRegisterWidth; i += kOpsPerThread) {
        c[i] = a[i] + b[i];
      }
      break;
    }
    
    case Map::op_code: {
      Map* map = (Map*) instr;
      /*
      const float* reg = registers[op.x]; 
      for (int i = local_idx; i < kRegisterWidth; i += kOpsPerThread) {
        elt = reg[i];

      }
      */
      break;
    }
    }
  }
}

int main(int argc, const char** argv) {
  int N = 10000 * kRegisterWidth; //2 << 24;

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
  h_program.add(LoadVector(a0,v0,BlockEltStart,VecWidth));
  h_program.add(LoadVector(a1,v1,BlockEltStart,VecWidth));
  h_program.add(Add(v0,v1,v2));
  h_program.add(StoreVector(a2, v2, BlockEltStart,VecWidth));

  printf("%d %d\n", *((uint16_t*)&h_program._ops[0]), *((uint16_t*) &h_program._ops[2]));
  printf("program length: %d\n", h_program.size());
  printf("load size %d\n", sizeof(LoadVector));
  printf("store size %d\n", sizeof(StoreVector));
  printf("add size %d\n", sizeof(Add));

  //  for (int i = 1; i <= N; i *= 2) {
  int total_blocks = divup(N, kRegisterWidth);
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
  run<<<blocks, threads>>>(h_program.to_gpu(),
  		           h_program.size(),
    		           d_values,
    		           n_values, 0, 0);
  cudaDeviceSynchronize();
  CHECK_CUDA();
  double ed = Now();
  fprintf(stderr, "%d elements in %.5f seconds; %.5f GFLOPS\n", N, ed - st, N * 1e-9 / (ed - st));
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
