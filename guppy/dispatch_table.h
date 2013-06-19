

#ifndef GUPPY_DISPATCH_TABLE_H
#define GUPPY_DISPATCH_TABLE_H

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

struct Instruction; 

typedef void (*EvalFn)(const Instruction*, EVAL_ARGS); 

static const int kMaxBytecodes = 200; 

#ifndef SWIG
__device__ static EvalFn dispatch_table[kMaxBytecodes]; 
#endif

#endif 

