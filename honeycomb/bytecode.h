#ifndef GUPPY_BYTECODE_H
#define GUPPY_BYTECODE_H

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <stdint.h>
#include <string>
#include <stdio.h>

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

  Instruction(int t, int s) :
      tag(t), size(s) {
  }
};

template<class SubClass>
struct InstructionT: public Instruction {
  InstructionT() :
      Instruction(SubClass::opcode, sizeof(SubClass)) {
  }
};
#endif

struct LoadVector: public InstructionT<LoadVector> {
  static const int opcode = 0;
  const uint16_t source_array;
  const uint16_t target_vector;
  const uint32_t start_idx;
  const uint16_t nelts;

  LoadVector(int source_array, int target_vector, int start_idx, int nelts) :
      source_array(source_array), target_vector(target_vector), start_idx(start_idx), nelts(nelts) {
  }
};

struct LoadVector2: public InstructionT<LoadVector2> {
  static const int opcode = 1;
  /* load elements from a global array into a local vector */
  const uint16_t source_array1;
  const uint16_t target_vector1;
  const uint16_t source_array2;
  const uint16_t target_vector2;

  const uint32_t start_idx;
  const uint16_t nelts;

  /*
   void to_ptx_buffer(Buffer& b) {
   for (int i = 0; i < kOpsPerThread; i++) {
   b.add("ld.global ...")
   }

   }
   */
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
};

struct StoreVector: public InstructionT<StoreVector> {
  static const int opcode = 2;
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
};

struct Add: public InstructionT<Add> {
  static const int opcode = 3;
  /* for now this will only work as a scalar operation,
   * expecting scalar float registers as arguments x,y,target
   */
  const uint16_t result;
  const uint16_t arg1;
  const uint16_t arg2;

  Add(int result, int arg1, int arg2) :
      result(result), arg1(arg1), arg2(arg2) {
  }
};

struct IAdd: public InstructionT<IAdd> {
  static const int opcode = 4;
  /* in-place variant of add: x = x + y */
  const uint16_t arg;
  const uint16_t result;

  IAdd(int result, int arg) :
      result(result), arg(arg) {
  }
};

struct Sub: public InstructionT<Sub> {
  static const int opcode = 5;
};

struct ISub: public InstructionT<ISub> {
  static const int opcode = 6;
};

struct Mul: public InstructionT<Mul> {
  static const int opcode = 7;
};

struct IMul: public InstructionT<IMul> {
  static const int opcode = 8;
};

struct Div: public InstructionT<Div> {
  static const int opcode = 9;
};

struct IDiv: public InstructionT<IDiv> {
  static const int opcode = 10;
};

struct MultiplyAdd: public InstructionT<MultiplyAdd> {
  static const int opcode = 11;
};

struct IMultiplyAdd: public InstructionT<IMultiplyAdd> {
  static const int opcode = 12;
};

struct Map: public InstructionT<Map> {
  static const int opcode = 13;
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
};

struct Map2: public InstructionT<Map2> {
  static const int opcode = 14;
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
};

class Subroutine {
  /* scalar-oriented program that runs at the thread level */
};

class Kernel {
  /* vector-oriented program that runs at the thread-block level */
};

class Program {
private:
  std::string _ops;
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
};

#endif
