#!/usr/bin/env python

from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu
import mako.template
import math
import numpy as N
import pycuda
import pycuda.driver as drv
import time


def timeit(f):
  st = time.time()
  f()
  pycuda.autoinit.context.synchronize()
  ed = time.time()
  print 'Operation %s completed in %.3f seconds' % (f, ed - st)


def div_up(a, b):
  return int(math.ceil(float(a) / float(b)))

THREADS_PER_BLOCK = 1024
MAX_DIM = None

def get_block_dims():
  global MAX_DIM
  device = drv.Device(0)
  attr = device.get_attributes()
  MAX_DIM = (attr[drv.device_attribute.MAX_BLOCK_DIM_X],
             attr[drv.device_attribute.MAX_BLOCK_DIM_Y],
             attr[drv.device_attribute.MAX_BLOCK_DIM_Z],)
  MAX_DIM = (512, 64, 4)



class ParFunction(object):
  def __init__(self, name, args, indices, source):
    self.name = name
    self.source = source
    self.args = args

    self.arg_vals = ','.join(args +
                             ['idx_%d' % i for i in range(len(indices))] +
                             ['max_idx_%d' % i for i in range(len(indices))])

    self.global_arg_decl = ','.join([('float* ' + arg) for arg in args] +
                                     ['int max_idx_%d' % i for i in range(len(indices))])
    self.local_arg_decl = ','.join(
                                   [('float* ' + arg) for arg in args] +
                                   [('int ' + idx) for idx in indices] +
                                   ['int max_idx_%d' % i for i in range(len(indices))])

def generate_source(fn):
  tmpl = mako.template.Template('''
__device__ __forceinline__ void ${fn.name}(${fn.local_arg_decl}) {
  ${fn.source}
}
__global__ void ${fn.name}_kernel(${fn.global_arg_decl}) {
  int idx_0 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // const int idx_1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx_1 = 0;
  //const int idx_2 = bz * blockDim.z + tz;
  const int idx_2 = 0;

  //if (idx_1 >= max_idx_1) { return; }
  //if (idx_2 >= max_idx_2) { return; }

  ${fn.name}(${fn.arg_vals});
}
''')
  return tmpl.render(fn=fn)

def pack_block(idxs, lims, threads_per_block):
  '''Try to pack a block with as many threads as possible.'''
  st = min(idxs[0], lims[0])
  block = [st, ]
  block_threads = st

  for i in range(1, len(idxs)):
    dim_pack = min(idxs[i], lims[i], threads_per_block / block_threads)
    block.append(dim_pack)
    block_threads *= dim_pack

  return tuple(block)

def parfor(fn, fixed_args, indices):
  '''Execute `fn` once for every index (i,j,...) present in the cartesian product of indices.'''
  src = generate_source(fn)
  module = SourceModule(src)
  kernel = module.get_function(fn.name + '_kernel')

  # compute a block and grid to execute fn
  block = pack_block(indices, MAX_DIM, THREADS_PER_BLOCK)
  grid = tuple([div_up(indices[0], block[0]),
                div_up(indices[1], block[1]),
                div_up(indices[2], block[2])])

  int_idx = [N.int32(idx) for idx in indices]
  args = list(fixed_args) + int_idx

  print block, grid
  kernel(*args, grid=grid, block=block)
  pycuda.autoinit.context.synchronize()

if __name__ == '__main__':
  import pycuda.autoinit
  get_block_dims()

  fn = ParFunction('add_array',
                   ['a', 'b', 'c'],
                   ['i', 'j', 'k'],
                   'c[i] = a[i] + b[i];')
                   # 'c[max_idx_0 * i + j] = a[max_idx_0 * i + j] + b[max_idx_0 * i + j];')

  print generate_source(fn)

  for dim in N.linspace(1, 4096).astype(N.int):
    dim = int(dim)
    a = to_gpu(N.ones((dim, dim), dtype=N.float32))
    b = to_gpu(N.ones((dim, dim), dtype=N.float32))
    c = to_gpu(N.ones((dim, dim), dtype=N.float32))
    def gpuarray_add():
      c = a + b

    def parfor_add():
      parfor(fn, (a, b, c), [dim * dim, 1, 1])

    timeit(gpuarray_add)
    timeit(parfor_add)


  assert(N.all(c.get() == 2))
