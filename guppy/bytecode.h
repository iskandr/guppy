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
	const uint16_t code :8;
  	const uint16_t size :8;
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
	const uint16_t source_array :8;
	const uint16_t target_vector :8;
	const uint32_t start_idx;
	const uint16_t nelts;

	LoadVector(int source_array, int target_vector, int start_idx, int nelts)
	  : source_array(source_array),
		target_vector(target_vector),
		start_idx(start_idx),
		nelts(nelts) {}

};

struct StoreVector : public InstructionT<StoreVector> {
	static const int op_code = 1;

    /* store elements of a vector into a global array
	 * starting from target_array[start_idx] until
	 * target_array[start_idx + nelts]
	 */
	const uint16_t target_array :8;
	const uint16_t source_vector :8;
	const uint32_t start_idx;
	const uint16_t nelts;

	StoreVector(int target_array, int source_vector, int start_idx, int nelts)
      : target_array(target_array),
	    source_vector(source_vector),
		start_idx(start_idx),
	    nelts(nelts) {}
};

struct Map : public InstructionT<Map> {
	static const int op_code = 2;

	/* map over element of source vector (which are loaded into scalar register input_elt)
	 * run given subprogram, write values of output_elt register into target_vector.
	 * The subprogram is just the next n_ops instructions.
	 */
	const uint16_t source_vector;
	const uint16_t target_vector;
	const uint16_t input_elt;
	const uint16_t output_elt;
	const uint16_t n_ops;

	Map(int source_vector, int target_vector, int input_elt, int output_elt, int n_ops)
    	: source_vector(source_vector),
    	  target_vector(target_vector),
    	  input_elt(input_elt),
    	  output_elt(output_elt),
    	  n_ops(n_ops) {}
};

struct Add : public InstructionT<Add> {
	static const int op_code = 3;

	/* for now this will only work as a scalar operation,
	 * expecting scalar float registers as arguments x,y,target
	 */
	const uint32_t arg1 :8;
	const uint32_t arg2 :8;
	const uint32_t result :16;

	Add(int arg1, int arg2, int result) : arg1(arg1), arg2(arg2), result(result) {}
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
