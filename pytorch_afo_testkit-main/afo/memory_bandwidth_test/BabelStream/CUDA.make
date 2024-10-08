CXXFLAGS=-O3
CUDA_CXX=nvcc

ifdef TBSIZE
CXXFLAGS+=-DTBSIZE=$(TBSIZE)
endif

cuda-stream: main.cpp CUDAStream.cu
	$(CUDA_CXX) -std=c++11 $(CXXFLAGS) -DCUDA $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f cuda-stream
