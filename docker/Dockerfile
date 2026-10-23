# Frigate + onnxruntime-gpu built for sm_50 (Maxwell), with CUDA 11.8 / cuDNN 8 runtime libs.
ARG FRIGATE_TAG=stable

FROM debian:bookworm-slim AS cuda
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
       cuda-cudart-11-8 libcublas-11-8 libcufft-11-8 libcurand-11-8 libcudnn8=8.9.7.29-1+cuda11.8 \
    && mkdir /libs \
    && cp -a /usr/local/cuda-11.8/lib64/libcudart.so.* /usr/local/cuda-11.8/lib64/libcublas.so.* \
             /usr/local/cuda-11.8/lib64/libcublasLt.so.* /usr/local/cuda-11.8/lib64/libcufft.so.* \
             /usr/local/cuda-11.8/lib64/libcurand.so.* /usr/lib/x86_64-linux-gnu/libcudnn*.so.8* /libs/ \
    && rm -f /libs/*_train*

FROM ghcr.io/blakeblackshear/frigate:${FRIGATE_TAG}
COPY --from=cuda /libs/ /usr/local/cuda-11.8/lib64/
COPY wheel/*.whl /tmp/
# patchelf: the ORT .so is flagged as needing an executable stack, which glibc >= 2.41 refuses to load.
RUN echo /usr/local/cuda-11.8/lib64 > /etc/ld.so.conf.d/cuda-11-8.conf && ldconfig \
    && pip3 uninstall -y --break-system-packages onnxruntime onnxruntime-gpu \
    && pip3 install --no-deps --break-system-packages /tmp/onnxruntime_gpu-*.whl \
    && pip3 install --break-system-packages patchelf \
    && find /usr/local/lib/python3.11/dist-packages/onnxruntime/capi -name '*.so*' -exec patchelf --clear-execstack {} + \
    && pip3 uninstall -y --break-system-packages patchelf \
    && rm /tmp/onnxruntime_gpu-*.whl
LABEL org.opencontainers.image.source=https://github.com/snooptheone/frigate-maxwell
