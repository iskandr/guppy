#!/usr/bin/env python

from guppy import bytecode
from pycuda import autoinit, gpuarray, driver, compiler
import os.path
import time

class CodeGen(object):
  def __init__(self, **kw):
    self.kw = kw

  def _format(self, fmt):
    from mako.template import Template
    tmpl = Template(fmt)
    kw = dict(self.kw)
    kw['this'] = self
    return tmpl.render(**kw)

  def emit(self):
    raw = self._emit()
    if not isinstance(raw, str):
      raw = '\n'.join([str(r) for r in raw])

    return self._format(raw)

class OpGen(CodeGen):
  pass

class LoadVector2(OpGen):
  def _emit(self):
    return '''
    float* reg1 = vectors[op->target_vector1];
    const float* src1 = arrays[op->source_array1];

    float* reg2 = vectors[op->target_vector2];
    const float* src2 = arrays[op->source_array2];
    const int start = int_scalars[op->start_idx];

#pragma unroll
    for (int i = local_idx * kOpsPerThread; i < (local_idx + 1) * kOpsPerThread; ++i) {
      reg1[i] = src1[start + i];
      reg2[i] = src2[start + i];
    }
  '''

class LoadVector(OpGen):
  def _emit(self):
    return '''
    float* reg = vectors[op->target_vector];
    const float* src = arrays[op->source_array];
    const int start = int_scalars[op->start_idx];

#pragma unroll
    for (int i = local_idx; i < kVectorWidth; i += kOpsPerThread) {
      reg[i] = src[start + i];
    }
  '''

INSTRUCTIONS = [LoadVector, LoadVector2]

class BytecodeHeader(CodeGen):
  def _emit(self):
    here = os.path.dirname(os.path.abspath(__file__))
    return open(here + '/bytecode.h').read()

class Enums(CodeGen):
  def _emit(self):
    return ''

class Constants(CodeGen):
  def __init__(self, **kw):
    CodeGen.__init__(self, kw)
    
  def _emit(self):
    return '''
static const int kThreadsX = 8; // 16;
static const int kThreadsY = 8; // 16;

// seems to give slightly better performance than kOpsPerThread = 8
static const int kOpsPerThread = 5;
static const int kThreadsPerBlock = kThreadsX * kThreadsY;
static const int kVectorWidth = kThreadsPerBlock * kOpsPerThread;
static const int kVectorPadding = 0;
static const int kNumVecRegisters = 4;
static const int kNumIntRegisters = 10;
static const int kNumLongRegisters = kNumIntRegisters;
static const int kNumFloatRegisters = 10;
static const int kNumDoubleRegisters = kNumFloatRegisters;
'''

class Registers(CodeGen):
  def __init__(self, scalar_registers_shared=True, **kw):
    CodeGen.__init__(self, scalar_registers_shared=scalar_registers_shared, **kw)

  def _emit(self):
    if self.kw['scalar_registers_shared']:
      return '''
    __shared__ int32_t int_scalars[kNumIntRegisters];
    __shared__ int64_t long_scalars[kNumIntRegisters];
    __shared__ float float_scalars[kNumFloatRegisters];
    __shared__ double double_scalars[kNumFloatRegisters];
    '''
    else:
      return '''
    int32_t int_scalars[kNumIntRegisters];
    int64_t long_scalars[kNumLongRegisters];
    float float_scalars[kNumFloatRegisters];
    double double_scalars[kNumFloatRegisters];
    '''

class SwitchDispatch(CodeGen):
  def _emit(self):
    yield '''
    while (pc < program_nbytes) {
    const Instruction* _op = (const Instruction*) &program[pc];
    pc += _op->size;
    '''
    yield 'switch (_op->tag) {'
    for op_klass in INSTRUCTIONS:
      klass_name = op_klass.__name__
      bytecode_inst = getattr(bytecode, klass_name)
      yield 'case %d: { ' % bytecode_inst.opcode
      yield '%s* op = (%s*)_op;' % (klass_name, klass_name)
      op = op_klass(**self.kw)
      yield op.emit()
      yield '}'
      yield 'break;'
    yield '}'
    yield '}'

class VM(CodeGen):
  def __init__(self, **kw):
    CodeGen.__init__(self, **kw)
    self.enums = Enums(**kw)
    self.constants = Constants(**kw)
    self.registers = Registers(**kw)
    self.dispatch = SwitchDispatch(**kw)
    self.bytecode = BytecodeHeader(**kw)

  def _emit(self):
    return '''
    ${this.bytecode.emit()}
    ${this.enums.emit()}
    ${this.constants.emit()}

extern "C" __global__ void vm_kernel(
     const char* __restrict__ program,
     long program_nbytes,
     float** arrays,
     const size_t* array_lengths) {

  const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
  const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int local_vector_offset = local_idx * kOpsPerThread;

  __shared__ float vectors[kNumVecRegisters][kVectorWidth + kVectorPadding];
  int pc = 0;

  ${this.registers.emit()}

  int_scalars[BlockStart] = block_offset;
  int_scalars[VecWidth] = kVectorWidth;
  int_scalars[BlockEltStart] = block_offset * kVectorWidth;

  ${this.dispatch.emit()}

  }
  '''

  def compile(self):
    mod = compiler.SourceModule(self.emit(), no_extern_c=True)
    return mod.get_function('vm_kernel')

if __name__ == '__main__':
  vm = VM()
  code = vm.emit()
  vm.compile()
