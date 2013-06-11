#include <stdio.h> 
#include <cuda.h>

#define BYTECODE_OP static inline __device__



BYTECODE_OP void load_slice() {

}

BYTECODE_OP void add(void* a, void *b) {

}

enum OP_CODE { ADD_VV, ADD_VS, ADD_SV, PUSH_V, PUSH_S, POP, BAD }; 

struct Op {
  Op() : code(BAD), arg(0) {}
  Op(OP_CODE c, int a) : code(c), arg(a) {}	
  OP_CODE code; 
  int arg;   
};

template <int N_VALUES, int N_OPS>
struct VMArgs {
  float* values[N_VALUES]; 
  static const int n_values = N_VALUES;  
  Op program[N_OPS]; 
  static const int n_ops = N_OPS; 
};

enum VMResult { OK, Error };

const int STACK_DEPTH = 100; 

template <int N_VALUES, int N_OPS>
__global__ void run(VMArgs<N_VALUES, N_OPS>* args) {
  
  void* stack[STACK_DEPTH];
  int stack_pos = -1; 
  int startIdx = blockIdx.x * blockDim.x; 
  int stopIdx = startIdx + blockDim.x; 

  for (int pc = 0; pc < args->n_ops; ++pc) { 
    
    Op op = args->program[pc];	  
    switch (op.code) { 
	    case PUSH_V: {
	      stack_pos++;
	      stack[stack_pos] = args->values[op.arg] + sizeof(float)*startIdx;
	    }
	    case POP: {
	      stack_pos--; 
	    }
	    case ADD_VV: {
	      int output_idx = op.arg; 	  
	      float* c = args->values[output_idx];
              float* b = stack[stack_pos-1];
              float* a = stack[stack_pos-2];
              
	      c[threadIdx.x] = b[threadIdx.x] + a[threadIdx.x];	      
              
	      stack_pos--;
	      stack[stack_pos] = c; 
            }
    }  
  }
}

const int THREADS_PER_BLOCK = 512; 

int main(char** argv, int argc) { 

  const int N = 400 * THREADS_PER_BLOCK;
  float a[N] = {1.0}; 
  float b[N] = {2.0};
  float* d_a;
  float* d_b; 
  float* d_c; 
  cudaMalloc(&d_a, sizeof(float)*N); 
  cudaMalloc(&d_b, sizeof(float)*N); 
  cudaMalloc(&d_c, sizeof(float)*N); 
  
  cudaMemcpy(&d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice); 
  cudaMemcpy(&d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice); 
  
  VMArgs<3,3> args; 
  
  args.values[0] = d_a; 
  args.values[1] = d_b; 
  args.values[2] = d_c; 

  args.program[0] = Op(PUSH_V, 0); 
  args.program[1] = Op(PUSH_V, 1); 
  args.program[2] = Op(ADD_VV, 2); 
  VMArgs<3,3>* d_args; 
  cudaMalloc(&d_args, sizeof(VMArgs<3, 3>)); 
  cudaMemcpy(&d_args, &args, sizeof(VMArgs<3,3>), cudaMemcpyHostToDevice); 

  run<3,3><<400>>(d_args);

  float c[N]; 
  cudaMemcpy(&c, d_c, sizeof(float)*N, cudaMemcpyDeviceToHost); 
  printf("%f %f %f\n", c[0], c[1], c[2]); 
  return 0; 
}
