FROM ubuntu:22.04

ARG KEEP_BUILD_FOLDER="False"
RUN echo "KEEP_BUILD_FOLDER $KEEP_BUILD_FOLDER"

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCMVERSION=6.0
ARG compiler_version=""
ARG compiler_commit=""

RUN set -xe

ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
RUN useradd -rm -d /home/jenkins -s /bin/bash -u 1004 jenkins
# Add rocm repository
RUN chmod 1777 /tmp
RUN apt-get update && apt-get install -y --allow-unauthenticated apt-utils wget gnupg2 curl && apt-get clean

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
RUN curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

RUN wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/focal/amdgpu-install_6.0.60000-1_all.deb  --no-check-certificate
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ./amdgpu-install_6.0.60000-1_all.deb && rm /amdgpu-install_6.0.60000-1_all.deb && apt-get clean

RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    sh -c "echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] $DEB_ROCM_REPO focal main > /etc/apt/sources.list.d/rocm.list" && \
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCMVERSION/ubuntu focal main > /etc/apt/sources.list.d/amdgpu.list'

RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu focal main universe | tee -a /etc/apt/sources.list"
RUN apt-get update && amdgpu-install -y --usecase=rocm --no-dkms && apt-get clean

# Install hipBLASLt dependent packages, then build and install hipBLASLt from source
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
  sudo \
  git \
  libelf1 \
  kmod \
  file \
  cmake \
  vim \
  python3\
  python3-pip \
  python3-venv \
  python3-joblib \
  hipblas \
  hipblas-dev \
  build-essential \
  && cd /root/ \
  && git clone https://github.com/ROCm/hipBLASLt \
  && cd hipBLASLt \
  && ./install.sh -idc \
  && if [ "$KEEP_BUILD_FOLDER" = "False" ] ; then \
  echo "Removing hipBLASLt build folder"; \
  rm -rf /root/hipBLASLt/build; \
  fi \
  && apt-get clean

RUN groupadd -f render

