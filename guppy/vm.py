#!/usr/bin/env python

from math import ceil, sqrt
from pycuda import autoinit, gpuarray, driver, compiler
import numpy as np 
import os
import time 


try:
  # assert os.system('make --silent -j') == 0
  import sys
  sys.path += ['./bin']

  import vm_wrap as vm
  from vm_wrap import a0, a1, a2, a3
  from vm_wrap import f0, f1, f2, f3
  from vm_wrap import BlockEltStart, VecWidth
  from vm_wrap import v0, v1, v2, v3
except:
  raise

def divup(a, b):
  return int(ceil(float(a) / float(b)))

def load_vm_module():
  import os
  #if debug:
  #  ret = os.system('make --silent -j DBG=1')
  #else:
  #  ret = os.system('make --silent -j')
  #assert ret == 0, "Make failed" 
  cubin = os.path.abspath('./bin/vm_kernel.cubin')
  assert os.path.exists(cubin), cubin
  mod = driver.module_from_file(str(cubin))
  return mod

def load_vm_kernel(_cache = [None]):
  if _cache[0] is None:
    mod = load_vm_module()
    fn = mod.get_function('vm_kernel')
    _cache[0] = fn
  return _cache[0]

def build_descriptor(args):
  for a in args: assert isinstance(a, gpuarray.GPUArray)
  ptrs = np.ndarray(len(args), dtype=np.int64)
  lens = np.ndarray(len(args), dtype=np.int64)
  for i in range(len(args)):
    ptrs[i] = int(args[i].gpudata)
    lens[i] = np.prod(args[i].shape)
  return ptrs, lens 

def run(program, args):
  ptrs, lens = build_descriptor(args)
  vm_kernel = load_vm_kernel()

  total_size = np.prod(args[0].shape)
  total_blocks = divup(total_size, vm.kVectorWidth);
  grid = (int(ceil(sqrt(total_blocks))), int(ceil(sqrt(total_blocks))), 1)

  block = (vm.kThreadsX, vm.kThreadsY, 1)

  p = program.code()
  host_bytecodes = np.frombuffer(p, dtype=np.uint8)
  gpu_bytecodes = driver.In(np.frombuffer(p, dtype=np.uint8))
  vm_kernel(gpu_bytecodes,
            np.int64(program.size()),
            driver.In(ptrs), driver.In(lens), 
            grid=grid, block=block)
  autoinit.context.synchronize()

class program(object):
  def __init__(self, bytecodes):
    self.p = vm.Program()
    for b in bytecodes:
      self.p.add(b)
  
  def __call__(self, *args, **kwargs): 
    #debug = kwargs.get('debug', False)
    load_vm_kernel()
    start_t = time.time()
    run(self.p, args)
    end_t = time.time()
    return end_t - start_t 
  """
    v0, v1 <- load2(a0,a1) 
    v1 += v0
    a2 <- v1
  """ 
 
p = program([
             vm.LoadVector2(a0, v0, a1, v1, BlockEltStart, VecWidth),
             vm.IAdd(v1, v0),
             vm.StoreVector(a2, v1, BlockEltStart, VecWidth)])

p1 = program([
              vm.LoadVector2(a0, v0, a1, v1, BlockEltStart, VecWidth), 
              vm.Map2(v0,v1,v2,f0,f1,f2,1),
              vm.IAdd(f1,f0), 
              vm.StoreVector(a2,v2,BlockEltStart,VecWidth)
             ])

N = 10**4 * 320 
a = gpuarray.zeros((N,), dtype=np.float32)
b = gpuarray.zeros((N,), dtype=np.float32)

a += 1
b += 2

c = gpuarray.zeros((N,), dtype=np.float32)

elapsed_t = p(a, b, c)

print "Array length:", N
print "Result:", c
print "Time elapsed:", elapsed_t
print "Throughpout:", (N*4)/(elapsed_t*1024*1024*1024), "GFLOP/S"
print "# Wrong: ", (c.get() != (a+b).get()).sum(), "/", N 

