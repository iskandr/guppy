#ifndef GUPPY_BYTECODE_H
#define GUPPY_BYTECODE_H

#include <stdint.h>
#include <string>


// global idx, by convention is stored in first integer register

enum IntRegisters { BlockStart, VecWidth, BlockEltStart, i0, i1, i2, i3};
enum FloatRegisters { f0, f1, f2, f3 };
enum VecRegisters { v0, v1, v2, v3 };
enum Arrays { a0, a1, a2, a3 };

struct Instruction {
/* every instruction must have a unique code and a size in number of bytes */
	const uint16_t code;
  	const uint16_t size ; 
  	Instruction(uint16_t code, uint16_t size) : code(code), size(size) {}
};

/* Beware of curiously recurring template pattern!*/
template <class SubType>
struct InstructionT : Instruction {
	InstructionT() : Instruction(SubType::op_code,  sizeof(SubType)) {}
};

struct LoadVector : public InstructionT<LoadVector> {
	static const int op_code = 0;

	/* load elements from a global array into a local vector */
	const uint16_t source_array;
	const uint16_t target_vector;
	const uint32_t start_idx;
	const uint16_t nelts;

	LoadVector(int source_array, int target_vector, int start_idx, int nelts)
	  : source_array(source_array),
	    target_vector(target_vector),
	    start_idx(start_idx),
	    nelts(nelts) {}
};


struct LoadVector2 : public InstructionT<LoadVector2> {
	static const int op_code = 1;

	/* load elements from a global array into a local vector */
	const uint16_t source_array1;
	const uint16_t target_vector1;
	const uint16_t source_array2;
	const uint16_t target_vector2;
      
	const uint32_t start_idx;
	const uint16_t nelts;

	LoadVector2(int source_array1, int target_vector1,
                    int source_array2, int target_vector2, 
                    int start_idx, int nelts)
	  : source_array1(source_array1),
            target_vector1(target_vector1),
            source_array2(source_array2),
            target_vector2(target_vector2),
            start_idx(start_idx),
	    nelts(nelts) {}

};

struct StoreVector : public InstructionT<StoreVector> {
	static const int op_code = 2;

    /* store elements of a vector into a global array
	 * starting from target_array[start_idx] until
	 * target_array[start_idx + nelts]
	 */
	const uint16_t target_array;
	const uint16_t source_vector;
	const uint32_t start_idx;
	const uint16_t nelts;

	StoreVector(int target_array, int source_vector, int start_idx, int nelts)
      : target_array(target_array),
	source_vector(source_vector),
	start_idx(start_idx),
	nelts(nelts) {}
};


struct Add : public InstructionT<Add> {
	static const int op_code = 3;

	/* for now this will only work as a scalar operation,
	 * expecting scalar float registers as arguments x,y,target
	 */
	const uint16_t result;
	const uint16_t arg1;
	const uint16_t arg2;

	Add(int result, int arg1, int arg2) : result(result), arg1(arg1), arg2(arg2) {}
};


struct IAdd : public InstructionT<IAdd> { 
  static const int op_code = 4;
  /* in-place variant of add: x = x + y */ 
  const uint16_t arg;
  const uint16_t result;

  IAdd(int result, int arg) : result(result), arg(arg) {}
};

struct Sub : public InstructionT<Sub> { 
  static const int op_code = 4; 
};


struct ISub : public InstructionT<ISub> { 
  static const int op_code = 5; 
};


struct Mul : public InstructionT<Mul> { 
  static const int op_code = 6; 
};


struct IMul : public InstructionT<IMul> { 
  static const int op_code = 7; 
};

struct Div : public InstructionT<Div> { 
  static const int op_code = 8; 
};

struct IDiv : public InstructionT<IDiv> { 
  static const int op_code = 9; 
};

struct MultiplyAdd : public InstructionT<MultiplyAdd> { 
  static const int op_code = 10;
};

struct IMultiplyAdd : public InstructionT<IMultiplyAdd> { 
  static const int op_code = 11;
};

struct Map : public InstructionT<Map> {
  static const int op_code = 12;

  /* map over elements of source vector 
   * (which are loaded into scalar register input_elt)
   * run given subprogram, 
   * write values of output_elt register into target_vector.
   * The subprogram is just the next n_ops instructions.
   */ 
  const uint64_t source_vector :16;
  const uint64_t target_vector :16;
  const uint64_t input_elt_reg :16;
  const uint64_t output_elt_reg :16;
 
  const uint16_t n_ops;

  Map(int source_vector, int target_vector, int input_elt, int output_elt, int n_ops)
  	: source_vector(source_vector),
    	  target_vector(target_vector),
    	  input_elt_reg(input_elt),
    	  output_elt_reg(output_elt),
    	  n_ops(n_ops) {}
};


struct Map2 : public InstructionT<Map2> {
  static const int op_code = 13;

  /* map over elements of two source vectors 
   * (which are loaded into scalar register input_elt) 
   * run given subprogram, 
   * write values of output_elt register into target_vector.
   * The subprogram is just the next n_ops instructions.
   */
  const uint64_t source_vector1  :16;
  const uint64_t source_vector2  :16;
  const uint64_t target_vector   :16;
  const uint64_t input_elt_reg1  :16;

  const uint64_t input_elt_reg2 :16; 
  const uint64_t output_elt_reg  :16;
  const uint64_t n_ops           :16; 

  Map2(int source_vector1, int source_vector2, 
       int target_vector, 
       int input_elt_reg1, 
       int input_elt_reg2, 
       int output_elt_reg, 
       int n_ops)
    	: source_vector1(source_vector1),
          source_vector2(source_vector2),
    	  target_vector(target_vector),
    	  input_elt_reg1(input_elt_reg1),
          input_elt_reg2(input_elt_reg2), 
    	  output_elt_reg(output_elt_reg),
    	  n_ops(n_ops) {}
};

struct Program {
  std::string _ops;
  char* _gpu_ptr;
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

  char* host_ptr() {
    return &_ops[0];
  }

  char* to_gpu() {
    if (_gpu_ptr) {
      return _gpu_ptr;
    }
    cudaMalloc(&_gpu_ptr, this->nbytes());
    cudaMemcpy(_gpu_ptr, this->host_ptr(), this->nbytes(), cudaMemcpyHostToDevice);
    return _gpu_ptr;
  }

  Program() : _gpu_ptr(NULL) { }
  ~Program() {
    if (_gpu_ptr) {
      cudaFree(_gpu_ptr);
    }
  }
};

#endif
