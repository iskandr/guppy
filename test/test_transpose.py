#!/usr/bin/env python

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
import numpy as N

from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

import time
def timeit(f):
  st, ed = drv.Event(), drv.Event()
  st.record()
  f()
  ed.record()
  ed.synchronize()
  print st.time_till(ed) * 1e-3

naive = SourceModule('''
__global__ void transpose(float* src, float* dst) {
  const int BSIZE = 32;
  const int N = 10000;
  const int r = blockIdx.x * BSIZE + threadIdx.x;
  const int c = blockIdx.y * BSIZE + threadIdx.y;
  dst[c * N + r] = src[r * N + c];
}
''')


diag = SourceModule('''
static const int TILE_DIM = 32;
static const int BLOCK_ROWS = 32;
__global__ void transpose(float *odata, float *idata, int width, int height) { 
 __shared__ float tile[TILE_DIM][TILE_DIM+1]; 
 int blockIdx_x, blockIdx_y; 
 // diagonal reordering 
 if (width == height) { 
   blockIdx_y = blockIdx.x; 
   blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
 } else { 
   int bid = blockIdx.x + gridDim.x*blockIdx.y; 
   blockIdx_y = bid%gridDim.y; 
   blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x; 
 } 
 int xIndex = blockIdx_x*TILE_DIM + threadIdx.x; 
 int yIndex = blockIdx_y*TILE_DIM + threadIdx.y; 
 int index_in = xIndex + (yIndex)*width; 
 xIndex = blockIdx_y*TILE_DIM + threadIdx.x; 
 yIndex = blockIdx_x*TILE_DIM + threadIdx.y; 
 int index_out = xIndex + (yIndex)*height; 
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { 
   tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width]; 
 } 
__syncthreads(); 
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { 
   odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i]; 
 } 
}

''')

DIM = 20000

src = N.arange(DIM * DIM).reshape((DIM, DIM)).astype(N.float32)
dst = N.empty_like(src)

naive_transpose = naive.get_function('transpose')
diag_transpose = diag.get_function('transpose')
x = to_gpu(src)
y = to_gpu(src)

drv.Context.set_cache_config(drv.func_cache.PREFER_L1)

timeit(lambda: naive_transpose(x, y, block=(32,32,1), grid=(DIM / 32, DIM / 32)))
timeit(lambda: diag_transpose(x, y, N.int32(DIM), N.int32(DIM), block=(32,32,1), grid=(DIM / 32, DIM / 32)))

g_src = x.get()
g_dst = y.get()

