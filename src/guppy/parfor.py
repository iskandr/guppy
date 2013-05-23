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
  ed = time.time()
  print 'Operation %s completed in %.3f seconds' % (f, ed - st)


def roundup(a, b):
  return int(math.ceil(float(a) / float(b)))

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
inline __device__ void ${fn.name}(${fn.local_arg_decl}) {
  ${fn.source}
}
__global__ void ${fn.name}_kernel(${fn.global_arg_decl}) {
  // compute function index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  const int idx_0 = bx * blockDim.x + tx;
  const int idx_1 = by * blockDim.y + ty;
  const int idx_2 = bz * blockDim.z + tz;

  if (idx_0 >= max_idx_0) { return; }
  if (idx_1 >= max_idx_1) { return; }
  if (idx_2 >= max_idx_2) { return; }

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
  device = drv.Device(0)
  attr = device.get_attributes()

  lim = (attr[drv.device_attribute.MAX_BLOCK_DIM_X],
         attr[drv.device_attribute.MAX_BLOCK_DIM_Y],
         attr[drv.device_attribute.MAX_BLOCK_DIM_Z],)

  threads_per_block = attr[drv.device_attribute.MAX_THREADS_PER_BLOCK]
  block = pack_block(indices, lim, threads_per_block)
  grid = tuple([roundup(indices[0], block[0]),
                    roundup(indices[1], block[1]),
                    roundup(indices[2], block[2])])

  print block, grid
  int_idx = [N.int32(idx) for idx in indices]
  args = list(fixed_args) + int_idx


  kernel(*args, grid=grid, block=block)
  pycuda.autoinit.context.synchronize()

if __name__ == '__main__':
  import pycuda.autoinit
  fn = ParFunction('add_array',
                   ['a', 'b', 'c'],
                   ['i', 'j', 'k'],
                   'c[max_idx_0 * i + j] = a[max_idx_0 * i + j] + b[max_idx_0 * i + j];')

  print generate_source(fn)

  for dim in N.linspace(1, 4096).astype(N.int):
    dim = int(dim)
    a = to_gpu(N.ones((dim, dim), dtype=N.float32))
    b = to_gpu(N.ones((dim, dim), dtype=N.float32))
    c = to_gpu(N.ones((dim, dim), dtype=N.float32))
    timeit(lambda: parfor(fn, (a, b, c), [dim, dim, 1]))


  assert(N.all(c.get() == 2))
