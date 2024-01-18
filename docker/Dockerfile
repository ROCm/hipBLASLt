FROM ubuntu:20.04

ARG ROCM_VERSION=5.4
ARG AMDGPU_VERSION=5.4
ARG KEEP_BUILD_FOLDER="False"
RUN echo "KEEP_BUILD_FOLDER $KEEP_BUILD_FOLDER"

# Register the ROCM package repository, and install rocm-dev package, then build and install hipBLASLt from source
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
  && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu focal main" | tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  git \
  libelf1 \
  kmod \
  file \
  python3\
  python3-pip \
  python3-venv \
  rocm-dev \
  hipblas \
  hipblas-dev \
  build-essential \
  && cd /root/ \
  && git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt \
  && cd hipBLASLt \
  && ./install.sh -idc \
  && if [ "$KEEP_BUILD_FOLDER" = "False" ] ; then \
  echo "Removing hipBLASLt build folder"; \
  rm -rf /root/hipBLASLt/build; \
  fi \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 109 render
