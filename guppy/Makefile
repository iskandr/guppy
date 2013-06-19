ifndef REALBUILD

export REALBUILD := 1
export DBG

ifdef DBG
all:
	mkdir -p build/dbg
	cd build/dbg && $(MAKE) -f ../../Makefile
	ln -sf build/dbg bin
	cp bin/core.py .
else
all: 
	mkdir -p build/opt
	cd build/opt &&  $(MAKE) -f ../../Makefile
	ln -sf build/opt bin
	cp bin/core.py .
endif

clean:
	rm -rf build/
	rm -f bin
else

SRCDIR := ../../
VPATH := $(SRCDIR)

SWIG := swig
CXX := g++
CUDA_SDK?=5

INCLUDES =  -I"$(SRCDIR)" -I/usr/include/python2.7
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

all : core.py _core.so

NVCC ?= nvcc
PTXAS ?= ptxas

SWIG_SOURCES := core.i
CU_SOURCES := 
 
HEADERS := $(wildcard $(SRCDIR)/*.h)

CPP_OBJS := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS := $(patsubst %.cu, %.o, $(CU_SOURCES))
SWIG_OBJS := $(patsubst %.i, _%.o, $(SWIG_SOURCES))

.PRECIOUS: %.ptx

_%.cpp %.py : %.i
	$(SWIG) -O  -python -c++ -outdir . -o $(patsubst %.i, _%.cpp, $(notdir $<)) $<

_core.so: $(SWIG_OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -shared  $^ -o _core.so
	
%.ptx : %.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

%.o : %.cu %.ptx  $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<


endif
