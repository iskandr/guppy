import math
import time

def timeit(f):
  import pycuda.autoinit
  st = time.time()
  f()
  pycuda.autoinit.context.synchronize()
  ed = time.time()
  print 'Operation %s completed in %.3f seconds' % (f, ed - st)

def div_up(a, b):
  return int(math.ceil(float(a) / float(b)))

def memoized(f):
  _cache = {}
  def _f(*args):
    if not args in _cache:
      _cache[args] = f(*args)
    return _cache[args]
  return _f

@memoized
def block_dims(device_num):
  from pycuda import driver
  device = driver.Device(device_num)
  attr = device.get_attributes()
  return (attr[driver.device_attribute.MAX_BLOCK_DIM_X],
          attr[driver.device_attribute.MAX_BLOCK_DIM_Y],
          attr[driver.device_attribute.MAX_BLOCK_DIM_Z],)
