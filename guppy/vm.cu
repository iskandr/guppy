#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define BYTECODE_OP static inline __device__

BYTECODE_OP void load_slice() {

}

BYTECODE_OP void add(void* a, void *b) {

}

double Now() {
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return tp.tv_sec + 1e-9 * tp.tv_nsec;
}

#define TIMEOP(op)\
{\
  double st = Now();\
  op;\
  double ed = Now();\
  fprintf(stderr, "%s finished in %.f seconds.\n", #op, end - start);\
}


enum OP_CODE { ADD_VV, ADD_VS, ADD_SV, BAD };

struct Op {
  Op() : code(BAD), x(0), y(0), dest(0) {}
  Op(OP_CODE code, int x, int y, int dest) : code(code), x(x), y(y), dest(dest)  {}

  OP_CODE code; 
  int x, y;
  int dest;
};

struct Vec {
  int _n;
  int _nbytes;
  float* _host_data;
  float* _gpu_data;
  bool _host_dirty;
  bool _gpu_dirty;

  void init(int n) {
    _n = n;
    _nbytes = sizeof(float) * n;
    _host_data = new float[n];
	  cudaMalloc(&_gpu_data, this->_nbytes);
    _host_dirty = false;
    _gpu_dirty = true;
  }

  Vec(int n) {
    this->init(n);
  }

  Vec (int n, float fill_value) {
    this->init(n);
    for (int i = 0; i < n; ++i) {
      _host_data[i] = fill_value;
    }
  }

  float* get_gpu_data() {
     if (_gpu_dirty) { this->copy_to_gpu(); }
     _host_dirty = true;
     _gpu_dirty = false;
     return _gpu_data;
  }

  float* get_host_data() {
	  if (_host_dirty) { this->copy_to_host();}
	  _gpu_dirty = true;
	  _host_dirty = false;
	  return _host_data;
  }

  void copy_to_host() {
	  cudaMemcpy(this->_host_data, this->_gpu_data, this->_nbytes, cudaMemcpyDeviceToHost);
  }


  void copy_to_gpu() {
	  cudaMemcpy(this->_gpu_data, this->_host_data, this->_nbytes, cudaMemcpyHostToDevice);
  }

  ~Vec() {
	  cudaFree(this->_gpu_data);
	  delete[] this->_host_data;
  }

};


const int STACK_DEPTH = 100; 


__global__ void run(float** values, int n_args, Op* program, int n_ops) {
  void* stack[STACK_DEPTH];
  int stack_pos = -1; 
  int startIdx = blockIdx.x * blockDim.x; 
  int stopIdx = startIdx + blockDim.x; 

  for (int pc = 0; pc < n_ops; ++pc) {
    Op op = program[pc];
    switch (op.code) { 
	    case ADD_VV: {
	      float* x = values[op.x] + startIdx + threadIdx.x;
	      float* y = values[op.y] + startIdx + threadIdx.x;
        float* z = values[op.dest] + startIdx + threadIdx.x;
        *z = *x + *y;
      }
	    break;
    }  
  }
}

#define THREADS_PER_BLOCK 512

int main(int argc, const char** argv) { 
  int N = 400 * THREADS_PER_BLOCK;
  if (argc > 1) {
    N = strtol(argv[1], NULL, 10);
  }
    
  Vec a(N, 1.0);
  Vec b(N, 2.0);
  Vec c(N);
  
  const int n_values = 3;
  float* h_values[n_values];
  h_values[0]= a.get_gpu_data();
  h_values[1] = b.get_gpu_data();
  h_values[2] = c.get_gpu_data();

  float** d_values;
  cudaMalloc(&d_values, sizeof(float*) * n_values);
  cudaMemcpy(d_values, h_values, sizeof(float*) * n_values, cudaMemcpyHostToDevice);

  int n_ops = 1;
  Op h_program[] = {Op(ADD_VV, 0, 1, 2)};
  Op* d_program;
  cudaMalloc(&d_program, sizeof(Op) * n_ops);
  cudaMemcpy(d_program, h_program, sizeof(Op) * n_ops, cudaMemcpyHostToDevice );

  double st = Now();
  run<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_values, n_values, d_program, n_ops);
  cudaDeviceSynchronize();
  double ed = Now();
  fprintf(stderr, "%.5f seconds\n", ed -st);

  float* ad = a.get_host_data();
  printf("%f %f %f\n", ad[0], ad[1], ad[2]);
  float* cd = c.get_host_data();
  printf("%f %f %f\n", cd[0], cd[1], cd[2]);
  return 0; 
}
