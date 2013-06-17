#ifndef GUPPY_CONFIG_H
#define GUPPY_CONFIG_H

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
static const int kThreadsX = 8; // 16;
static const int kThreadsY = 8; // 16;

// seems to give slightly better performance than kOpsPerThread = 8
static const int kOpsPerThread = 5;

static const int kThreadsPerBlock = kThreadsX * kThreadsY;

static const int kVectorWidth = kThreadsPerBlock * kOpsPerThread;

static const int kVectorPadding = 0;

static const int kNumVecRegisters = 4;

static const int kNumIntRegisters = 10;
static const int kNumLongRegisters = kNumIntRegisters; 
static const int kNumFloatRegisters = 10;
static const int kNumDoubleRegisters = kNumFloatRegisters;



#ifndef PREFETCH_GPU_BYTECODE 
  #define PREFETCH_GPU_BYTECODE 0
#endif 

#if PREFETCH_GPU_BYTECODE
  static const int kMaxProgramLength = 500; 
#endif 

#ifndef VECTOR_LOAD_CONTIGUOUS
  #define VECTOR_LOAD_CONTIGUOUS 1
#endif 

#ifndef VECTOR_LOAD_CHECK_BOUNDS
  #define VECTOR_LOAD_CHECK_BOUNDS 1
#endif 

#ifndef VECTOR_STORE_CONTIGUOUS
  #define VECTOR_STORE_CONTIGUOUS 1
#endif

#ifndef VECTOR_STORE_CHECK_BOUNDS
  #define VECTOR_STORE_CHECK_BOUNDS 1
#endif 

#ifndef VECTOR_OPS_CONTIGUOUS
  #define VECTOR_OPS_CONTIGUOUS 1 
#endif

#ifndef SCALAR_REGISTERS_SHARED
  #define SCALAR_REGISTERS_SHARED 0
#endif 

 
#if VECTOR_LOAD_CONTIGUOUS 
  #define MEMORY_ACCESS_LOOP \
    int i = local_idx*kOpsPerThread;\
    (local_idx+1)*kOpsPerThread <= nelts ? (local_idx+1)*kOpsPerThread : nelts;\
    ++i
#else 
  #define MEMORY_ACCESS_LOOP \
    int i = local_idx; i < nelts; i += kOpsPerThread; ++i 
#endif
 
#if VECTOR_OPS_CONTIGUOUS  
  #define VECTOR_OP_LOOP \
     int i = local_idx*kOpsPerThread;\
     i < (local_idx+1)*kOpsPerThread; \
     ++i
#else 
  #define VECTOR_OP_LOOP \
     int i = local_idx; i < kVectorWidth; i += kOpsPerThread
#endif 

#endif // GUPPY_CONFIG_H      
