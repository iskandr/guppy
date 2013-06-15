BIN               := vm 
CXX               := gcc-4.6
CXXFLAGS          := -O3 -g

NVCC ?= nvcc
PTXAS ?= ptxas

INCD =  -I"./"
LIBS =  -lrt -lstdc++ -lpthread -lcuda -lcudart -lcublas -lcufft
CUDA_SDK?=5

ARCH := -arch=sm_13
NVCCFLAGS := $(ARCH) --ptxas-options='-v' -O3 -g -ccbin=gcc-4.4#--compiler-bindir=./nvcc-bindir
PTXASFLAGS := $(ARCH) -v

# files
CPP_SOURCES       :=  
CU_SOURCES        := vm.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

all: $(BIN) vm.ptx 

%.ptx : %.cu
	$(NVCC) $(NVCCFLAGS) $(INCD) -ptx -o $@ $<

%.o : %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCD) -c -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -link -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS) 

clean:
	rm -f $(BIN) *.o *.ptx
