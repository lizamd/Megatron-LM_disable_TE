
default: kokkos-stream

ifndef KOKKOS_PATH
$(error Must set KOKKOS_PATH to point to kokkos install)
endif

ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU INTEL

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_GNU = g++
COMPILER_INTEL = icpc -qopt-streaming-stores=always
COMPILER_HIP = hipcc
COMPILER_CUDA = $(NVCC_WRAPPER)
CXX = $(COMPILER_$(COMPILER))

ifndef TARGET
define target_help
Set TARGET to change to offload device. Defaulting to CPU.
Available targets are:
  CPU (default)
  CUDA
  HIP
endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ($(TARGET), CUDA)
CXX = $(COMPILER_CUDA)
else ifeq ($(TARGET), HIP)
CXX = $(COMPILER_HIP)
endif

OBJ = main.o KokkosStream.o

kokkos-stream: $(OBJ) $(KOKKOS_CPP_DEPENDS)
	$(CXX) -L$(KOKKOS_PATH)/lib -L$(KOKKOS_PATH)/lib64 -ldl -Wl,--enable-new-dtags -DKOKKOS -O3 -std=c++14 $(EXTRA_FLAGS) $(OBJ) -lkokkoscore -lkokkoscontainers -o $@

%.o: %.cpp
	$(CXX) -I$(KOKKOS_PATH)/include -DKOKKOS -O3 -std=c++14 $(EXTRA_FLAGS) -c $<

.PHONY: clean
clean:
	rm -f kokkos-stream main.o KokkosStream.o
