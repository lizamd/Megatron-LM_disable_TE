FROM   rocm/pytorch-nightly:2024-03-01-rocm6.0.2 as base_rocm  
#6.0.2-115

# General build overrides
ARG GFX_COMPILATION_ARCH="gfx941:xnack-,gfx942:xnack-,gfx90a"
ENV WORKSPACE_DIR=/rocm
WORKDIR $WORKSPACE_DIR

COPY . /rocm

#  BUILD BabelStream
RUN cp -r memory_bandwidth_test/BabelStream BabelStream
RUN cd BabelStream/ && "/opt/rocm/llvm/bin/clang"  -isystem "/opt/rocm-6.0.2/include" --offload-arch="$GFX_COMPILATION_ARCH" -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false --driver-mode=g++ --hip-link --rtlib=compiler-rt -unwindlib=libgcc  -std=c++17 -O3 -DHIP -x hip main.cpp -x hip HIPStream.cpp -o "hip-stream"
RUN cp /rocm/BabelStream/hip-stream /rocm/

# add locale en_US.UTF-8
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8

# record configuration for posterity
RUN pip3 list

CMD ["bash", "-l"]


