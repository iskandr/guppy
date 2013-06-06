import numpy as N
from pycuda import autoinit, curandom, compiler, gpuarray

mod = compiler.SourceModule
mod = mod('''__global__ void foo(float* x) {
  __shared__ float out[256];
  if (threadIdx.x == 0 && threadIdx.y == 0) { 
    for (int i = 0; i < 256; ++i) { out[i] = 0; }
  }
  out[0] += 1;
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) { 
    for (int i = 0; i < 256; ++i) { x[i] = out[i]; }
  }
}
''')

f = mod.get_function('foo')
tgt = gpuarray.to_gpu(N.ones((256,), dtype=N.float32))
f(tgt, block=(16,16,1))
print tgt
