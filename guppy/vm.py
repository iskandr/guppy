#!/usr/bin/env python
from math import ceil, sqrt
from pycuda import autoinit, gpuarray, driver, compiler
import numpy as N
import os

try:
  import sys
  sys.path += ['./bin']

  import vm_wrap as VM
  from vm_wrap import a0, a1, a2, a3
  from vm_wrap import f0, f1, f2, f3
  from vm_wrap import BlockEltStart, VecWidth
  from vm_wrap import v0, v1, v2, v3
except:
  raise

def divup(a, b):
  return int(ceil(float(a) / float(b)))

def load_vm_module(debug=False):
  import os
  if debug:
    os.system('make --silent -j DBG=1')
  else:
    os.system('make --silent -j')

  cubin = os.path.abspath('./bin/vm_kernel.cubin')
  assert os.path.exists(cubin), cubin
  mod = driver.module_from_file(str(cubin))
  return mod

def load_vm_kernel(debug=False):
  mod = load_vm_module(debug)
  return mod.get_function('vm_kernel')

def program(bytecodes):
  p = VM.Program()
  for b in bytecodes:
    p.add(b)
  return p

def build_descriptor(args):
  for a in args: assert isinstance(a, gpuarray.GPUArray)
  ptrs = N.ndarray(len(args), dtype=N.int64)
  lens = N.ndarray(len(args), dtype=N.int64)
  for i in range(len(args)):
    ptrs[i] = int(args[i].gpudata)
    lens[i] = N.prod(args[i].shape)

  return gpuarray.to_gpu(ptrs), gpuarray.to_gpu(lens)

def run(program, args, debug=False):
  ptrs, lens = build_descriptor(args)
  vm_kernel = load_vm_kernel(debug)

  total_size = N.prod(args[0].shape)
  total_blocks = divup(total_size, VM.kVectorWidth);
  grid = (int(ceil(sqrt(total_blocks))), int(ceil(sqrt(total_blocks))), 1)

  block = (VM.kThreadsX, VM.kThreadsY, 1)

  p = program.code()
  vm_kernel(driver.In(N.frombuffer(p, dtype=N.uint8)),
            N.int64(program.size()),
            ptrs, lens, grid=grid, block=block)

  
  """
    v0, v1 <- load2(a0,a1) 
    v1 += v0
    a2 <- v1
  """ 
 
p = program([
             VM.LoadVector2(a0, v0, a1, v1, BlockEltStart, VecWidth),
             VM.IAdd(v1, v0),
             VM.StoreVector(a2, v1, BlockEltStart, VecWidth)])

a = gpuarray.zeros((100000,), dtype=N.float32)
b = gpuarray.zeros((100000,), dtype=N.float32)

a += 1
b += 2

c = gpuarray.zeros((100000,), dtype=N.float32)
run(p, [a, b, c], debug=False)

print c
