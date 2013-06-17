#include <stdio.h>
#include <time.h>
#include <vector>
#include <math.h>

#include "config.h"
#include "vec.h"
#include "util.h"
#include "vm.h" 

int main(int argc, const char** argv) {
  int N = 10000 * kVectorWidth;

  Vec a(N, 1.0);
  Vec b(N, 2.0);
  Vec c(N);
  const int n_arrays = 3;
  float* h_arrays[n_arrays];
  h_arrays[0] = a.get_gpu_data();
  h_arrays[1] = b.get_gpu_data();
  h_arrays[2] = c.get_gpu_data();

  size_t h_lengths[n_arrays];
  h_lengths[0] = a.size();
  h_lengths[1] = b.size();
  h_lengths[2] = c.size(); 
  
  float** d_arrays;
  cudaMalloc(&d_arrays, sizeof(float*) * n_arrays);
  cudaMemcpy(d_arrays, h_arrays, sizeof(float*) * n_arrays, 
             cudaMemcpyHostToDevice);
  
  size_t* d_lengths;
  cudaMalloc(&d_lengths, sizeof(size_t) * n_arrays); 
  cudaMemcpy(d_lengths, &h_lengths, sizeof(size_t)*n_arrays, 
             cudaMemcpyHostToDevice); 

  //one version that uses efficient instructions
  Program h_program1;
  h_program1.add(LoadVector2(a0,v0,a1,v1,BlockEltStart,VecWidth));
  h_program1.add(IAdd(v1,v0));
  h_program1.add(StoreVector(a2, v1, BlockEltStart,VecWidth));

  //perform vector operation indirectly via Map 
  Program h_program2; 
  // par {
  //   v0 = a0[BlockEltStart:BlockEltStart+VecWidth] 
  //   v1 = a1[BlockEltStart:BlockEltStart+VecWidth] 
  // } 
  h_program2.add(LoadVector2(a0,v0,a1,v1,BlockEltStart,VecWidth));
  // map2 from (f0 <- v0; f1 <- v1) to (f2 -> v1) {
  //   f2 = f0 + f1
  // }
  h_program2.add(Map2(v0,v1,v1,f0,f1,f1,1));
  h_program2.add(IAdd(f1,f0));
  // a[BlockEltStart:BlockEltStart+VecWidth] = v1
  h_program2.add(StoreVector(a2, v1, BlockEltStart, VecWidth)); 

  int total_blocks = divup(N, kVectorWidth);
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
  run<<<blocks, threads>>>(h_program2.to_gpu(),
  		           h_program2.size(),
    		           d_arrays,
                           d_lengths);
  cudaDeviceSynchronize();
  CHECK_CUDA();
  double ed = Now();
  fprintf(stderr, "%d elements in %.5f seconds; %.5f GFLOPS\n", N, ed - st, N * 1e-9 / (ed - st));

  float* ad = a.get_host_data();
  printf("%f %f %f\n", ad[0], ad[10], ad[N - 200]);
  float* bd = b.get_host_data();
  printf("%f %f %f\n", bd[0], bd[10], bd[N - 200]);
  float* cd = c.get_host_data();
  for (int i = 0; i < min(64, N); ++i) {
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
