#include <stdint.h>

#include "vm_kernel.cuh"
#include "config.h"
#include "bytecode.h"

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

__device__   inline size_t run_subprogram(const char* __restrict__ program, int pc, int n_ops, EVAL_ARGS) {
  for (int i = 0; i < n_ops; ++i) {
    Instruction* instr = (Instruction*) &program[pc];
    pc += instr->size;
    run_local_instruction(instr, EVAL_ARG_ACTUALS);
  }
  return pc;

}

extern "C" {

  int pc = 0;
  const Instruction* instr;
  while (pc < program_nbytes) {
    instr = (const Instruction*) &program[pc];
    pc += instr->size;
    // dispatch_table[instr->tag](instr, EVAL_ARG_ACTUALS); 
  
    switch (instr->tag) {
      case LoadVector::code: {
        ((const LoadVector*) instr)->eval(EVAL_ARG_ACTUALS); 
        break;
      }
      case LoadVector2::code: {
        ((const LoadVector2*) instr)->eval(EVAL_ARG_ACTUALS); 
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
   
  } //while 
} // run

} // extern
