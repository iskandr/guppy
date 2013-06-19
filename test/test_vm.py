from honeycomb import bytecode, vm, core
from honeycomb.bytecode import program, divup
from honeycomb.core import *
from math import ceil, sqrt
from pycuda import autoinit, driver, gpuarray
import numpy as np
import time

p = program([
             core.MulScalar(i0, Idx.x, VecSize),
             core.LoadVector2(a0, v0, a1, v1, i0),
             core.IAdd(v1, v0),
             core.StoreVector(a2, v1, i0)])

p1 = program([
              core.MulScalar(i0, Idx.x, VecSize),
              core.LoadVector2(a0, v0, a1, v1, i0),
              core.Map2(v0, v1, v2, f0, f1, f2, 1),
              core.IAdd(f1, f0),
              core.StoreVector(a2, v2, i0)
             ])

# todo -- push these through vm compile
threads_x = 8
threads_y = 8
ops_per_thread = 5
vector_width = threads_x * threads_y * ops_per_thread


def run(program, args):
  ptrs, lens = bytecode.build_descriptor(args)
  vm_kernel = vm.VM().compile()

  total_size = np.prod(args[0].shape)
  total_blocks = divup(total_size, vector_width);
  grid = (int(ceil(sqrt(total_blocks))), int(ceil(sqrt(total_blocks))), 1)

  block = (threads_x, threads_y, 1)

  p = program.code()
  host_bytecodes = np.frombuffer(p, dtype=np.uint8)
  gpu_bytecodes = driver.In(np.frombuffer(p, dtype=np.uint8))

  st = time.time()
  vm_kernel(gpu_bytecodes,
     np.int64(program.size()),
     driver.In(ptrs), driver.In(lens),
     grid=grid, block=block)
  autoinit.context.synchronize()
  ed = time.time()

  return ed - st


N = 10 ** 4 * 320
a = gpuarray.zeros((N,), dtype=np.float32)
b = gpuarray.zeros((N,), dtype=np.float32)

a += 1
b += 2

c = gpuarray.zeros((N,), dtype=np.float32)

elapsed_t = run(p, (a, b, c))

print "Array length:", N
print "Result:", c
print "Time elapsed:", elapsed_t
print "Throughpout:", (N * 4) / (elapsed_t * 1024 * 1024 * 1024), "GFLOP/S"
print "# Wrong: ", (c.get() != (a + b).get()).sum(), "/", N
