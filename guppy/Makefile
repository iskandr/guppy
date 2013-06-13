BIN               := vm 
CXXFLAGS          := -O3 -g

NVCC ?= nvcc
INCD =  -I"./"
LIBS =  -lrt -lstdc++ -lpthread
CUDA_SDK?=3
NVCCFLAGS := --ptxas-options=-v -O3 -G -g -arch compute_20


# files
CPP_SOURCES       :=  
CU_SOURCES        := vm.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -link -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu.o
