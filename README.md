# frigate-maxwell

[Frigate](https://frigate.video) image with GPU object detection on Maxwell GPUs (GTX 750 Ti, 9xx, 960M and similar, compute capability 5.0/5.2).

The official `stable-tensorrt` image ships ONNX Runtime kernels that start at a newer architecture. On a GTX 960M, inference fails with `cudaErrorNoKernelImageForDevice`. This repo builds `onnxruntime-gpu` with CUDA 11.8 and cuDNN 8 for `sm_50` and layers it on top of the regular `ghcr.io/blakeblackshear/frigate` image.

Measured on a GTX 960M with `yolov9-t` at 320x320: 57-100 ms per frame on the CPU, 17 ms on the GPU.

## Use

```yaml
services:
  frigate:
    image: ghcr.io/snooptheone/frigate-maxwell:stable   # or a version, e.g. :0.18.2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      NVIDIA_DRIVER_CAPABILITIES: all
```

Use the `onnx` detector with a model such as YOLOv9. ONNX Runtime picks CUDA on its own:

```yaml
detectors:
  onnx:
    type: onnx
model:
  model_type: yolo-generic
  path: /config/yolov9-t.onnx
  width: 320
  height: 320
  input_tensor: nchw
  input_dtype: float
  labelmap_path: /labelmap/coco-80.txt
```

Check it works: `/api/stats` should show an `inference_speed` well below the CPU figure, and `nvidia-smi` should list the Frigate detector process.

## How it is built

| Workflow | When | What |
|---|---|---|
| `build-wheel` | manual | Compiles `onnxruntime-gpu` for `sm_50` and attaches the wheel to a release (`ort-<tag>-sm50`). |
| `build-image` | weekly, on Dockerfile changes, manual | Installs the wheel and the CUDA 11.8 / cuDNN 8 runtime libraries over the Frigate image, then pushes to GHCR. |

Run `build-wheel` once before the first `build-image`. The compile takes a long time on a hosted runner.

Patches applied during the build (see `docker/build-ort.sh` and `docker/Dockerfile`):

- `moe_kernel.cu`: the pre-`sm_53` stub declares its last parameter `const int`, the real kernel `int`. nvcc rejects the mismatch.
- GitLab answers 403 to scripted eigen downloads and the archive no longer matches the SHA1 in ORT's `deps.txt`. The build fetches the same commit from the GitHub mirror and pins that file's hash.
- `patchelf --clear-execstack` on the ORT libraries. glibc 2.41 refuses to load them otherwise.
- Flash and memory-efficient attention are disabled. They are slow to compile and YOLO does not use them.

## Limits

- The CI cannot test the GPU. It checks that the module imports and that the CUDA provider has no unresolved libraries. Verify on real hardware after an update.
- The wheel is `cp311`. If Frigate moves off Python 3.11, rerun `build-wheel`.
- CUDA 11.8 and cuDNN 8 are downloaded from NVIDIA during the build and are covered by NVIDIA's licenses.
- GHCR packages start private. Set the package to public in its settings.

## Local build

```bash
docker build -f docker/Dockerfile.builder -t ort-builder docker
mkdir -p work out
docker run --rm -v "$PWD/work:/work" -v "$PWD/out:/out" \
  -v "$PWD/docker/build-ort.sh:/build-ort.sh:ro" ort-builder bash /build-ort.sh
mkdir -p wheel && cp out/*.whl wheel/
docker build -f docker/Dockerfile -t frigate-maxwell .
```
