# Build env for onnxruntime-gpu with sm_50 (Maxwell) kernels: CUDA 11.8 from NVIDIA apt, no .run/.deb files needed
FROM python:3.11-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 ninja-build git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compiler-11-8 cuda-libraries-dev-11-8 libcudnn8=8.9.7.29-1+cuda11.8 libcudnn8-dev=8.9.7.29-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sfn /usr/local/cuda-11.8 /usr/local/cuda \
    && ln -sf /usr/include/x86_64-linux-gnu/cudnn*.h /usr/local/cuda-11.8/include/ \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-11.8/lib64/
RUN pip install --no-cache-dir cmake==3.26.4 numpy==1.26.4 wheel packaging setuptools coloredlogs flatbuffers protobuf sympy
ENV CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 PATH=/usr/local/cuda-11.8/bin:$PATH
