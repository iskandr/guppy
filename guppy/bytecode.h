#ifndef GUPPY_BYTECODE_H
#define GUPPY_BYTECODE_H

#include <stdint.h>
#include <string>
#include <stdio.h> 

#include "config.h"
#include "dispatch_table.h" 


// global idx, by convention is stored in first integer register
enum IntRegisters {
  BlockStart,
  VecWidth,
  BlockEltStart,
  i0,
  i1,
  i2,
  i3
};
enum FloatRegisters {
  f0,
  f1,
  f2,
  f3
};
enum VecRegisters {
  v0,
  v1,
  v2,
  v3
};
enum Arrays {
  a0,
  a1,
  a2,
  a3
};

#ifndef SWIG
struct Instruction {
  /* every instruction must have a unique code and a size in number of bytes */
  const uint16_t tag;
  const uint16_t size;
  Instruction(uint16_t code, uint16_t size) :
      tag(code), size(size) {
  }
};

static int put_into_dispatch_table(int code, EvalFn fn) {
  printf("Initializing code %d with address %p", code, fn);  
  cudaMemcpy(&dispatch_table[code], &fn, sizeof(EvalFn), cudaMemcpyHostToDevice); 
  return 0; 
}

/*
struct DispatchEntry {
  DispatchEntry() { 
    printf("Calling default constructor\n"); 
  }
  DispatchEntry(int code, EvalFn fn) { 
    // dispatch_table[code] = fn;
    printf("Moving fn address %p size %d to location %p\n", &dispatch_table[code], sizeof(EvalFn), 
      &fn);  
    cudaMemcpy(&dispatch_table[code], &fn, sizeof(EvalFn), cudaMemcpyHostToDevice); 
  }
};
*/ 
/* Beware of curiously recurring template pattern!*/
template<class SubType>
struct InstructionT: Instruction {
  // a trick (cute or horrific, depending on your disposition), to
  // register each opcode's eval fn with the device dispatch table 
  static int _garbage; // DispatchEntry put_into_dispatch_table;

  InstructionT() :
      Instruction(SubType::code, sizeof(SubType)) {
  }
  
  /* in a further act of madness, let's give every class a 
     static method which takes an opaque base pointer
     and we'll convert that into an ordinary method call
  */ 
  __device__ static void eval(const Instruction* instr, EVAL_ARGS) { 
    ((const SubType*) instr)->eval_impl(EVAL_ARG_ACTUALS); 
  } 
}; 

template <class SubType>
int InstructionT<SubType>::_garbage = put_into_dispatch_table(SubType::code, &SubType::eval);

#endif

struct LoadVector: public InstructionT<LoadVector> {
  static const int code = 0;
  
  // static DispatchEntry put_into_dispatch_table;
  /* load elements from a global array into a local vector */
  const uint16_t source_array;
  const uint16_t target_vector;
  const uint32_t start_idx;
  const uint16_t nelts;

  LoadVector(int source_array, int target_vector, int start_idx, int nelts) :
      source_array(source_array), target_vector(target_vector), start_idx(start_idx), nelts(nelts) {
  }
  
  __device__ void eval_impl  (EVAL_ARGS) const  { 
    float* reg = vectors[this->target_vector];
    const float* src = arrays[this->source_array];
    const int start = int_scalars[this->start_idx];
    #pragma unroll 5
    for (int i = local_idx; i < kVectorWidth; i += kOpsPerThread)
    {
      reg[i] = src[start+i];
    }
  }
};

struct LoadVector2: public InstructionT<LoadVector2> {
  static const int code = 1;

  /* load elements from a global array into a local vector */
  const uint16_t source_array1;
  const uint16_t target_vector1;
  const uint16_t source_array2;
  const uint16_t target_vector2;

  const uint32_t start_idx;
  const uint16_t nelts;

  LoadVector2(int source_array1,
              int target_vector1,
              int source_array2,
              int target_vector2,
              int start_idx,
              int nelts) :
      source_array1(source_array1),
      target_vector1(target_vector1),
      source_array2(source_array2),
      target_vector2(target_vector2),
      start_idx(start_idx),
      nelts(nelts) {
  }

  __device__ void eval_impl (EVAL_ARGS) const {     
    float* reg1 = vectors[this->target_vector1];
    const float* src1 = arrays[this->source_array1];

    float* reg2 = vectors[this->target_vector2];
    const float* src2 = arrays[this->source_array2];
    const int start = int_scalars[this->start_idx];
     
    #pragma unroll 5
    for ( int i = local_idx * kOpsPerThread; i < (local_idx+1) * kOpsPerThread; ++i) 
    { 
      reg1[i] = src1[start+i];
      reg2[i] = src2[start+i];
    }
    /*
        #pragma unroll 5
    for (int i = local_idx; i < kVectorWidth; i += kOpsPerThread)
        {
          const int shared_idx = local_idx + i*kOpsPerThread;  
          const int global_idx = start + shared_idx; 
          reg1[shared_idx] = src1[global_idx]; 
          reg2[shared_idx] = src2[global_idx];
        }
        //__syncthreads(); 
    */    
  }
};

struct StoreVector: public InstructionT<StoreVector> {
  static const int code = 2;

  /* store elements of a vector into a global array
   * starting from target_array[start_idx] until
   * target_array[start_idx + nelts]
   */
  const uint16_t target_array;
  const uint16_t source_vector;
  const uint32_t start_idx;
  const uint16_t nelts;

  StoreVector(int target_array, int source_vector, int start_idx, int nelts) :
      target_array(target_array), source_vector(source_vector), start_idx(start_idx), nelts(nelts) {
  }
  
  __device__ void eval_impl (EVAL_ARGS) const {

  }
};

struct Add: public InstructionT<Add> {
  static const int code = 3;

  /* for now this will only work as a scalar operation,
   * expecting scalar float registers as arguments x,y,target
   */
  const uint16_t result;
  const uint16_t arg1;
  const uint16_t arg2;

  Add(int result, int arg1, int arg2) :
      result(result), arg1(arg1), arg2(arg2) {
  }
  __device__ void eval_impl (EVAL_ARGS) const {

  }

};

struct IAdd: public InstructionT<IAdd> {
  static const int code = 4;
  /* in-place variant of add: x = x + y */
  const uint16_t arg;
  const uint16_t result;

  IAdd(int result, int arg) :
      result(result), arg(arg) {
  }
  __device__ void eval_impl (EVAL_ARGS) const {

  }

};

struct Sub: public InstructionT<Sub> {
  static const int code = 4;
  __device__ void eval_impl (EVAL_ARGS) const {

  }
};

struct ISub: public InstructionT<ISub> {
  static const int code = 5;
};

struct Mul: public InstructionT<Mul> {
  static const int code = 6;
};

struct IMul: public InstructionT<IMul> {
  static const int code = 7;
};

struct Div: public InstructionT<Div> {
  static const int code = 8;
};

struct IDiv: public InstructionT<IDiv> {
  static const int code = 9;
};

struct MultiplyAdd: public InstructionT<MultiplyAdd> {
  static const int code = 10;
};

struct IMultiplyAdd: public InstructionT<IMultiplyAdd> {
  static const int code = 11;
};

struct Map: public InstructionT<Map> {
  static const int code = 12;

  /* map over elements of source vector 
   * (which are loaded into scalar register input_elt)
   * run given subprogram, 
   * write values of output_elt register into target_vector.
   * The subprogram is just the next n_ops instructions.
   */
  const uint16_t source_vector;
  const uint16_t target_vector;
  const uint16_t input_elt_reg;
  const uint16_t output_elt_reg;

  const uint16_t n_ops;

  Map(int source_vector, int target_vector, int input_elt, int output_elt, int n_ops) :
      source_vector(source_vector),
      target_vector(target_vector),
      input_elt_reg(input_elt),
      output_elt_reg(output_elt),
      n_ops(n_ops) {
  }
  __device__ void eval_impl (EVAL_ARGS) const {

  }
};

struct Map2: public InstructionT<Map2> {
  static const int code = 13;

  /* map over elements of two source vectors 
   * (which are loaded into scalar register input_elt) 
   * run given subprogram, 
   * write values of output_elt register into target_vector.
   * The subprogram is just the next n_ops instructions.
   */
  const uint32_t source_vector1;
  const uint32_t source_vector2;
  const uint32_t target_vector;
  const uint32_t input_elt_reg1;

  const uint32_t input_elt_reg2;
  const uint32_t output_elt_reg;
  const uint32_t n_ops;

  Map2(int source_vector1,
       int source_vector2,
       int target_vector,
       int input_elt_reg1,
       int input_elt_reg2,
       int output_elt_reg,
       int n_ops) :
      source_vector1(source_vector1),
      source_vector2(source_vector2),
      target_vector(target_vector),
      input_elt_reg1(input_elt_reg1),
      input_elt_reg2(input_elt_reg2),
      output_elt_reg(output_elt_reg),
      n_ops(n_ops) {
  }
  __device__ void eval_impl (EVAL_ARGS) const {

  }
};

class Subroutine { 
  /* scalar-oriented program that runs at the thread level */ 
};

class Kernel { 
  /* vector-oriented program that runs at the thread-block level */ 
};


class Program {
  /* TODO: Make this a global/host level program that initiates kernel launches */ 

private:
  std::string _ops;
  char* _gpu_ptr;
public:
  void add(const Instruction& instr) {
    const char* data = (const char*) &instr;
    int len = instr.size;

    _ops += std::string(data, len);
  }
  int size() {
    return _ops.size();
  }

  int nbytes() {
    return this->size();
  }

  std::string code() {
    return _ops;
  }

  void* to_gpu();

  Program() :
      _gpu_ptr(NULL) {
  }
  ~Program();
};

#endif
