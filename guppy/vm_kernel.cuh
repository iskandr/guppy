#ifndef GUPPY_VM_H
#define GUPPY_VM_H

/*
 #define CALL_EVAL(t) \
  ((t*)instr)->eval(local_idx, \
           arrays,\
           array_lengths,\
           (float**) vectors, \
           int_scalars, \
           long_scalars, \
           float_scalars, \
           double_scalars)
 */

extern "C" {
__global__ void vm_kernel(const char* __restrict__ program,
                          long program_nbytes,
                          float** arrays,
                          const size_t* array_lengths);
}

#endif 
