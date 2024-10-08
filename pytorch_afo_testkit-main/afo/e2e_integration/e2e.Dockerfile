FROM   rocm/pytorch-nightly:2024-03-01-rocm6.0.2 as base_rocm  
#6.0.2-115

# General build overrides
ARG GFX_COMPILATION_ARCH="gfx941:xnack-,gfx942:xnack-,gfx90a"

# ---------------------------------------------------------------------------------------------------------------
#  BUILD BabelStream
FROM base_rocm AS rocm_babelstream
ARG GFX_COMPILATION_ARCH
WORKDIR /rocm

COPY memory_bandwidth_test/BabelStream BabelStream
RUN cd BabelStream/ && "/opt/rocm/llvm/bin/clang"  -isystem "/opt/rocm-6.0.2/include" --offload-arch="$GFX_COMPILATION_ARCH" -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false --driver-mode=g++ --hip-link --rtlib=compiler-rt -unwindlib=libgcc  -std=c++17 -O3 -DHIP -x hip main.cpp -x hip HIPStream.cpp -o "hip-stream"

# BabelStream export for output tar generation
FROM scratch AS export-rocm_babelstream
COPY --from=rocm_babelstream /rocm/BabelStream/hip-stream /

# ---------------------------------------------------------------------------------------------------------------
# FINAL
FROM base_rocm AS final
WORKDIR /rocm

COPY . /rocm


RUN mkdir build
RUN ls

RUN /bin/sh -c 'if [ `ls * | grep "rocm_.*.tar" | wc -l` -gt 0 ]; then for tar_file in rocm_*.tar; do tar -xvf $tar_file -C build ; done; fi'
RUN cp /rocm/build/hip-stream /rocm/

