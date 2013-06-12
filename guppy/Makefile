BIN               := vm 
CXXFLAGS          := -O3 -g

NVCC ?= nvcc
INCD =  -I"./"
LIBS =  -lcuda  -lcudart -lcublas -lcufft -lstdc++ -lpthread
CUDA_SDK?=3
NVCCFLAGS := --ptxas-options=-v -O3 -G -g 


# files
CPP_SOURCES       :=  
CU_SOURCES        := vm.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o