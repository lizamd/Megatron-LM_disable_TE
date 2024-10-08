HIP_PATH?= /opt/rocm
HIPCC=$(HIP_PATH)/bin/hipcc

ifdef TBSIZE
CXXFLAGS+=-DTBSIZE=$(TBSIZE)
endif

ifdef ELTS_PER_LANE
$(warning ELTS_PER_LANE environment variable is depricated, please use DWORDS_PER_LANE instead)
DWORDS_PER_LANE=$(ELTS_PER_LANE)
endif

ifdef DWORDS_PER_LANE
CXXFLAGS+=-DDWORDS_PER_LANE=$(DWORDS_PER_LANE)
endif

ifdef CHUNKS_PER_BLOCK
CXXFLAGS+=-DCHUNKS_PER_BLOCK=$(CHUNKS_PER_BLOCK)
endif

srcs := main.cpp HIPStream.cpp
deps := main.o HIPStream.o

hip-stream: main.cpp HIPStream.cpp
	$(HIPCC) $(CXXFLAGS) -std=c++17 -O3 -DHIP $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hip-stream
