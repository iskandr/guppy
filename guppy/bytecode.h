#ifndef GUPPY_BYTECODE_H
#define GUPPY_BYTECODE_H

#include <stdint.h>




enum OP_CODE {

  LOAD_SLICE,       // load/store global 1D array
  STORE_SLICE,
  LOAD_ROW_SLICE,   // load/store rows of global 2D array
  STORE_ROW_SLICE,
  LOAD_COL_SLICE,   // load/store cols of global 2D array
  STORE_COL_SLICE,
  LOAD_SCALAR,      // distribute scalar across elements of shared vector
  STORE_SCALAR,     // write first element of shared vector to a single global location
  ADD,              // arithmetic between shared vectors
  SUB,
  MUL,
  DIV
};

//#define PACKED 64

#if PACKED == 32
  struct Op {
    uint32_t code :8;
    uint32_t x :8;
    uint32_t y :8;
    uint32_t z :8;
  };
#elif PACKED == 64
  struct Op {
    uint64_t code :16;
	uint64_t x :16;
	uint64_t y :16;
	uint64_t z :16;
};
#else
  struct Op {
    uint32_t code;
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
#endif

Op make_op(OP_CODE code, int x, int y, int z) {
  Op op = { code, x, y, z };
  return op;
}

struct Program {
  std::vector<Op> _ops;
  Op* _gpu_ptr;

  Program& Add(int x, int y, int z) {
    _ops.push_back(make_op(ADD, x, y, z));
    return *this;
  }
  Program& LoadSlice(int src, int dst) {
    _ops.push_back(make_op(LOAD_SLICE, src, dst, 0));
    return *this;
  }
  Program& StoreSlice(int src, int dst) {
    _ops.push_back(make_op(STORE_SLICE, src, dst, 0));
    return *this;
  }

  int size() {
    return _ops.size();
  }

  int nbytes() {
    return sizeof(Op) * this->size();
  }

  Op* host_ptr() {
    return &_ops[0];
  }
  Op* to_gpu() {
    if (_gpu_ptr) {
      return _gpu_ptr;
    }
    cudaMalloc(&_gpu_ptr, this->nbytes());
    cudaMemcpy(_gpu_ptr, this->host_ptr(), this->nbytes(), cudaMemcpyHostToDevice);
    return _gpu_ptr;
  }

  Program() :
      _gpu_ptr(NULL) {
  }
  ~Program() {
    if (_gpu_ptr) {
      cudaFree(_gpu_ptr);
    }
  }
};

#endif
