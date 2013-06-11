#include <cuda.h>

#define BYTECODE_OP static inline __device__

BYTECODE_OP void load_slice() {

}

BYTECODE_OP void add(void* a, void *b) {

}

struct VMArgs {

};

__global__ void execute(VMArgs* args) {

}