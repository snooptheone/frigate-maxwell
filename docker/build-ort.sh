#!/usr/bin/env bash
# Runs inside the builder container. Mounts: /work (source+build tree), /out (results).
set -euo pipefail
ORT_TAG=${ORT_TAG:-v1.18.1}
JOBS=${JOBS:-$(nproc)}
cd /work
[ -d ort ] || git clone --recursive --depth 1 --branch "$ORT_TAG" https://github.com/microsoft/onnxruntime.git ort
cd ort

# GitLab refuses scripted downloads of the eigen archive (HTTP 403) and regenerated it, so the SHA1 pinned in
# deps.txt is stale anyway. Fetch the same commit from the GitHub mirror and pin the hash of that file.
if grep -qE '^eigen;https://gitlab.com' cmake/deps.txt; then
  sha=$(grep -E '^eigen;' cmake/deps.txt | grep -oE '[0-9a-f]{40}' | head -1)
  curl -fsSL -o /tmp/eigen.zip "https://github.com/eigen-mirror/eigen/archive/${sha}.zip"
  sed -i -E "s#^eigen;[^;]+;[0-9a-f]+#eigen;file:///tmp/eigen.zip;$(sha1sum /tmp/eigen.zip | cut -d' ' -f1)#" cmake/deps.txt
fi
grep '^eigen' cmake/deps.txt

# sm_50 (< 530) stub in moe_kernel.cu declares the last param `const int`; the real kernel takes `int`.
# nvcc then emits a mismatching device stub.
sed -i '/__CUDA_ARCH__ < 530/,/^#else/ s/int, const int) {/int, int) {/' \
  onnxruntime/contrib_ops/cuda/moe/ft_moe/moe_kernel.cu

./build.sh --config Release --build_shared_lib --parallel "$JOBS" --nvcc_threads 1 \
  --use_cuda --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/local/cuda-11.8 \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=50 CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
  --build_wheel --skip_tests --allow_running_as_root --compile_no_warning_as_error

mkdir -p /out
cp build/Linux/Release/dist/onnxruntime_gpu-*.whl /out/
