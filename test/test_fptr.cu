struct VMState {
  int a;
  float *b;
};

static const int kOpsPerThread = 1;

__device__ void a0(const VMState& vm) { vm.b[vm.a] = 0.; }
__device__ void a1(const VMState& vm) { vm.b[vm.a] = 1.; }
__device__ void a2(const VMState& vm) { vm.b[vm.a] = 2.; }
__device__ void a3(const VMState& vm) { vm.b[vm.a] = 3.; }
__device__ void a4(const VMState& vm) { vm.b[vm.a] = 4.; }
__device__ void a5(const VMState& vm) { vm.b[vm.a] = 5.; }
__device__ void a6(const VMState& vm) { vm.b[vm.a] = 6.; }

typedef void (*Fn)(const VMState&);

__device__ static Fn table[7] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
__global__ void c(float *out) {
  const int block_size = blockDim.y * blockDim.x;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * block_size * kOpsPerThread;

  int registers[8];
  // for (int i = 0; i < kOpsPerThread; ++i) {
  // }

  registers[threadIdx.x] = 0;

  VMState vm;
  vm.b = out;
  for (int i = 0; i < kOpsPerThread; ++i) {
    vm.a = offset + (i * block_size) + threadIdx.x + registers[threadIdx.x];
    // vm.a = offset + threadIdx.x * kOpsPerThread + i;
    table[threadIdx.x % 7](vm);
  }

}
