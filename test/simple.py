from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu
import pycuda.autoinit
import pycuda.driver as drv
import numpy as N

src = '''
__global__ void add_array_kernel(float* a,float* b,float* c,int xlen,int ylen) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int i = bx * blockDim.x + tx;
  const int j = by * blockDim.y + ty;

  if (i < xlen && j < ylen) {
    c[1764 * i + j] = a[1764 * i + j] + b[1764 * i + j];
  }
}
'''

module = SourceModule(src)
kernel = module.get_function('add_array_kernel')
block = (1024, 1, 1)

a = to_gpu(N.ones((1764, 1764), dtype=N.float32))
b = to_gpu(N.ones((1764, 1764), dtype=N.float32))
c = to_gpu(N.ones((1764, 1764), dtype=N.float32))
ctx = pycuda.autoinit.context

for i in range(4, 1765, 5):
  grid = (2, i, 1)
  print i, grid
  kernel(a, b, c, N.int32(1764), N.int32(1764),
         block=block,
         grid=grid)
  ctx.synchronize()
