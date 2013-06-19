from core import *
from math import ceil
from pycuda import gpuarray
import core
import numpy as np

# assign opcodes
def instructions():
  for name in dir(core):
    obj = getattr(core, name)
    if hasattr(obj, 'mro') and core.Instruction in obj.mro():
      yield obj

def divup(a, b):
  return int(ceil(float(a) / float(b)))

def build_descriptor(args):
  for a in args: assert isinstance(a, gpuarray.GPUArray)
  ptrs = np.ndarray(len(args), dtype=np.int64)
  lens = np.ndarray(len(args), dtype=np.int64)
  for i in range(len(args)):
    ptrs[i] = int(args[i].gpudata)
    lens[i] = np.prod(args[i].shape)
  return ptrs, lens

class program(object):
  def __init__(self, bytecodes):
    self.p = core.Program()
    for b in bytecodes:
      self.p.add(b)

  def size(self):
    return self.p.size()

  def code(self):
    return self.p.code()
