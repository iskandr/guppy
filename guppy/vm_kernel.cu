#include <stdint.h>

#include "vm_kernel.cuh"
#include "config.h"
#include "bytecode.h"

#define EVAL_ARGS \
  int local_idx, \
  float** arrays,\
  const size_t* array_lengths, \
  float** vectors, \
  int32_t* int_scalars, \
  float* float_scalars

#define EVAL_ARG_ACTUALS \
  local_idx, arrays, array_lengths, \
  (float**) vectors, int_scalars, float_scalars

__device__ inline void run_local_instruction(Instruction* instr, EVAL_ARGS) {
  switch (instr->tag) {
    case Add::code: {
      Add* op = (Add*) instr;
      float a = float_scalars[op->arg1];
      float b = float_scalars[op->arg2];
      float_scalars[op->result] = a+b;
      break;
    }
}    }

__device__   inline size_t run_subprogram(char* program, int pc, int n_ops, EVAL_ARGS) {
  for (int i = 0; i < n_ops; ++i) {
    Instruction* instr = (Instruction*) &program[pc];
    pc += instr->size;
    run_local_instruction(instr, EVAL_ARG_ACTUALS);
  }
  return pc;

}

extern "C" {
__global__ void vm_kernel(char* program,
                          long program_nbytes,
                          float** arrays,
                          const size_t* array_lengths) {
  const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
  const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int local_vector_offset = local_idx * kOpsPerThread;
  const int next_vector_offset = local_vector_offset + kOpsPerThread;

  __shared__ float vectors[kNumVecRegisters][kVectorWidth + kVectorPadding];
#if SCALAR_REGISTERS_SHARED
  __shared__ int32_t int_scalars[kNumIntRegisters];
  __shared__ int64_t long_scalars[kNumIntRegisters];
  __shared__ float float_scalars[kNumFloatRegisters];
  __shared__ double double_scalars[kNumFloatRegisters];
#else
  int32_t int_scalars[kNumIntRegisters];
  int64_t long_scalars[kNumLongRegisters];
  float float_scalars[kNumFloatRegisters];
  double double_scalars[kNumFloatRegisters];
#endif

  int_scalars[BlockStart] = block_offset;
  int_scalars[VecWidth] = kVectorWidth;
  int_scalars[BlockEltStart] = block_offset * kVectorWidth;

  int pc = 0;
  Instruction* instr;
  while (pc < program_nbytes) {

    instr = (Instruction*) &program[pc];
    pc += instr->size;

    switch (instr->tag) {
      case LoadVector::code: {
        LoadVector* op = (LoadVector*) instr;
        const int source = op->source_array;
        float* reg = vectors[op->target_vector];
        const float* src = arrays[source];
        const int start = int_scalars[op->start_idx];
#if VECTOR_LOAD_CHECK_BOUNDS
        const int len = array_lengths[source];
        const int nelts = start+kVectorWidth <= len ? kVectorWidth : kVectorWidth ? nelts : kVectorWidth;
        int stop_i = (local_idx+1)*kOpsPerThread;
        stop_i + start_i
        for (int i = local_idx
#else
#pragma unroll 5
            for ( int i = local_idx * kOpsPerThread; i < (local_idx+1) * kOpsPerThread; ++i)
#endif
            {
              reg[i] = src[start+i];
            }

            break;
          }
case LoadVector2::code: {
        LoadVector2* load_slice = (LoadVector2*) instr;

        float* reg1 = vectors[load_slice->target_vector1];
        const float* src1 = arrays[load_slice->source_array1];

        float* reg2 = vectors[load_slice->target_vector2];
        const float* src2 = arrays[load_slice->source_array2];

        const int start = int_scalars[load_slice->start_idx];
#if VECTOR_LOAD_CHECK_BOUNDS
        int nelts = int_scalars[load_slice->nelts];
        nelts = nelts <= kVectorWidth ? nelts : kVectorWidth;
#endif

#pragma unroll 5
        for ( int i = local_idx * kOpsPerThread; i < (local_idx+1) * kOpsPerThread; ++i) {
          //for (int i = local_idx; i < kVectorWidth; i += kOpsPerThread) {
          reg1[i] = src1[start+i];
          reg2[i] = src2[start+i];
        }

        break;
      }

      case StoreVector::code: {
        StoreVector* store = (StoreVector*) instr;
        const float* reg = vectors[store->source_vector];
        float* dst = arrays[store->target_array];
        const int start = int_scalars[store->start_idx];
#if VECTOR_STORE_CHECK_BOUNDS
        int nelts = int_scalars[store->nelts];
        nelts = nelts <= kVectorWidth ? nelts : kVectorWidth;
#endif

#pragma unroll 5
        for ( int i = local_idx * kOpsPerThread; i < (local_idx+1) * kOpsPerThread; ++i) {
          //int i = local_idx * kOpsPerThread; i < (local_idx+1)*kOpsPerThread; ++i) {
          //MEMORY_ACCESS_LOOP {
          dst[i+start] = reg[i];
        }
        break;
      }

      case Add::code: {
        Add* op = (Add*) instr;
        float* a = vectors[op->arg1];
        float* b = vectors[op->arg2];
        float* c = vectors[op->result];
#pragma unroll 5
        for ( int i = local_idx * kOpsPerThread; i < (local_idx+1) * kOpsPerThread; ++i) {
          c[i] = a[i] + b[i];
        }
        break;

      }

      case IAdd::code: {
        IAdd* op = (IAdd*) instr;
        const float* a = vectors[op->arg];
        float* b = vectors[op->result];

        #pragma unroll 5
        for (int i = local_idx * kOpsPerThread; i < (local_idx+1)*kOpsPerThread; ++i) {
          b[i] += a[i];
        }
        break;
      }

      case Map::code: {
        Map* op = (Map*) instr;
        const float* src = vectors[op->source_vector];
        float* dst = vectors[op->target_vector];
        float* in_reg = &float_scalars[op->input_elt_reg];
        float* out_reg = &float_scalars[op->output_elt_reg];
        size_t old_pc = pc;
        for (int i = local_idx * kOpsPerThread; i < (local_idx+1)*kOpsPerThread; ++i) {
          in_reg[0] = src[i];
          pc = run_subprogram(program, old_pc, op->n_ops, EVAL_ARG_ACTUALS);
          dst[i] = out_reg[0];
        }
        break;
      }

      case Map2::code: {
        Map2* op = (Map2*) instr;
        const float* src1 = vectors[op->source_vector1];
        const float* src2 = vectors[op->source_vector2];
        float* dst = vectors[op->target_vector];
        float* in_reg1 = &float_scalars[op->input_elt_reg1];
        float* in_reg2 = &float_scalars[op->input_elt_reg2];
        float* out_reg = &float_scalars[op->output_elt_reg];
        size_t old_pc = pc;

        for (int i = local_idx * kOpsPerThread; i < (local_idx+1)*kOpsPerThread; ++i) {
          *in_reg1 = src1[i];
          //*in_reg2 = src2[i];
          // pc = run_subprogram(program, old_pc, op->n_ops, EVAL_ARG_ACTUALS);
          dst[i] = *out_reg;
        }

        break;
      }
    } // switch
  } // while
} // run

} // extern
