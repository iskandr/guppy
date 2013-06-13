#ifndef GUPPY_BYTECODE_H
#define GUPPY_BYTECODE_H

#include <stdint.h>

#define PACKED

enum OP_CODE {
  LOAD_SLICE,
  STORE_SLICE,    // load slice of global arrays into shared vector
  LOAD_SCALAR,
  STORE_SCALAR, // distribute scalar across elements of shared vector
  ADD,
  SUB,
  MUL,
  DIV,        // arithmetic between shared vectors
  BAD
};

#ifdef PACKED
  struct Op {
    uint32_t code :8;
    uint32_t x :8;
    uint32_t y :8;
    uint32_t z :8;
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
