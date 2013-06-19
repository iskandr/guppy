static const int kOpsPerThread = 1;

__global__ void kernel_local(float *out, int input) {
  float reg[8] = { 1 };
  for (int i = 0; i < kOpsPerThread; ++i) {
    reg[threadIdx.x] += threadIdx.y;
  }
  out[0] = reg[0];
}

__global__ void kernel_reg(float *out) {
  float reg = threadIdx.x % 2;
  const int block_size = blockDim.y * blockDim.x;
  for (int i = 0; i < kOpsPerThread; ++i) {
    reg = reg + i % 7 + i % 8;
  }
  out[0] = reg;
}

__global__ void kernel_shared(float *out) {
  __shared__ float reg[128];
  reg[threadIdx.y * blockDim.x + threadIdx.x] = 0;
  int offset = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < kOpsPerThread; ++i) {
    reg[offset + i % 7] += reg[i % 8];
  }
  out[0] = reg[0];
}
