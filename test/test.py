from pycuda import autoinit, compiler, gpuarray
import numpy as N
import time, math

ARRAY_SIZE = 2 ** 27
OPS_PER_THREAD = 64

g = gpuarray.zeros((ARRAY_SIZE,), dtype=N.float32)
block = (64, 1, 1)
grid_x = int(math.sqrt(ARRAY_SIZE / 64 / OPS_PER_THREAD))
grid = (grid_x, grid_x, 1)

def timeit(f, msg):
  st = time.time()
  
  autoinit.context.synchronize()
  ed = time.time()
  print ed - st, 1e-9 * ARRAY_SIZE / (ed - st), msg

def test_fptr():
  s = compiler.SourceModule('test_fptr.cu')
  kernel= s.get_function('c')
  timeit(lambda: kernel(g, block=block, grid=grid), 'test_fptr')

def test_reg():
  s = compiler.SourceModule(open('test_reg.cu').read())
  timeit(lambda: s.get_function('kernel_reg')(g, block=block, grid=grid), 'test_reg')
  timeit(lambda: s.get_function('kernel_local')(g, block=block, grid=grid), 'test_local')
  timeit(lambda: s.get_function('kernel_shared')(g, block=block, grid=grid), 'test_shared')

test_reg()
