ifndef REALBUILD

export REALBUILD := 1
export DBG

ifdef DBG
all:
	mkdir -p build/dbg
	cd build/dbg && $(MAKE) -f ../../Makefile
	ln -sf build/dbg bin
else
all: 
	mkdir -p build/opt
	cd build/opt &&  $(MAKE) -f ../../Makefile
	ln -sf build/opt bin
endif

clean:
	rm -rf build/
else

SRCDIR := ../../
VPATH := $(SRCDIR)

SWIG := swig
CXX := g++
CUDA_SDK?=5

INCLUDES =  -I"$(SRCDIR)" 
LIBS =  -lrt -lstdc++ -lpthread -lcuda -lcudart -lcublas -lcufft

ARCH := -arch=sm_20
PTXASFLAGS := $(ARCH) -v

NVCCFLAGS := $(ARCH) $(INCLUDES) --ptxas-options='-v' -ccbin=gcc-4.4 -Xcompiler '-fPIC'
CXXFLAGS := -fPIC $(INCLUDES)

ifdef DBG
CXXFLAGS := $(CXXFLAGS) -O0 -ggdb2
NVCCFLAGS := $(NVCCFLAGS) -O1 -G --generate-line-info 
else
CXXFLAGS := $(CXXFLAGS) -O3
NVCCFLAGS := $(NVCCFLAGS) -O3
endif

all : vm_kernel.cubin test_main _vm_wrap.so

NVCC ?= nvcc
PTXAS ?= ptxas

SWIG_SOURCES := vm_wrap.i
CU_SOURCES := vm_kernel.cu util.cu bytecode.cu
 
HEADERS := $(wildcard $(SRCDIR)/*.h)

CPP_OBJS := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS := $(patsubst %.cu, %.o, $(CU_SOURCES))
SWIG_OBJS := $(patsubst %.i, _%.o, $(SWIG_SOURCES))

.PRECIOUS: %.ptx

_%.o : %.i
	$(SWIG) -O -python -c++ -o _vm_wrap.cu $<
	$(NVCC) $(NVCCFLAGS) -I/usr/include/python2.7 -c _vm_wrap.cu -o _vm_wrap.o

_vm_wrap.so: $(SWIG_OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) -shared -I/usr/include/python2.7 $^ -o _vm_wrap.so
	
%.ptx : %.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

%.o : %.cu %.ptx  $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

vm_kernel.cubin: vm_kernel.cu
	$(NVCC) $(NVCCFLAGS) -cubin -o $@ $^
	
test_main: $(CPP_OBJS) $(CU_OBJS) test_main.o
	$(NVCC) $(NVCCFLAGS) -link -o $@ $^ $(LIBS)
endif
