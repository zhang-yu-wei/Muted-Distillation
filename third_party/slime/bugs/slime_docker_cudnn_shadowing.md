# slimerl/slime:latest ships a stale apt cuDNN that shadows the pip cuDNN torch needs

**Status:** confirmed in `slimerl/slime:latest` (digest `sha256:a7317182c71d...`, pulled
2026-07-10), worked around in `examples/tau-bench/docker_run.sh`. Not yet reported to
slime.

## Summary

The image contains **two cuDNN installations**:

| location | version | origin |
|---|---|---|
| `/usr/lib/x86_64-linux-gnu/libcudnn.so.9*` | **9.10.2** | apt, inherited from the sglang base image |
| `/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/` | **9.17.1** | pip `nvidia-cudnn-cu12`, the version torch 2.11.0+cu129 was built against |

The image's default `LD_LIBRARY_PATH`
(`/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`) contains neither,
so `libcudnn.so.9` resolves through the ldconfig cache to the **apt 9.10.2** copy.

## Symptoms

1. Any code path that initializes torch's cuDNN bindings fails immediately:

   ```
   RuntimeError: cuDNN version incompatibility: PyTorch was compiled against (9, 17, 1)
   but found runtime version (9, 10, 2). ...
   ```

   Triggered by e.g. `torch.backends.cudnn.version()`, convolutions, or TE attention —
   i.e. the image cannot run this workload as shipped.

2. The "obvious" fix — prepending the pip cuDNN dir to `LD_LIBRARY_PATH` — makes it
   worse, not better:

   ```
   cuDNN Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
   cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
   ```

   cuDNN 9 is split into a main lib plus sublibraries (`libcudnn_graph`,
   `libcudnn_engines_precompiled`, ...). In this configuration the process ends up with
   a **mixed set** (9.17 main lib, 9.10 sublibs or vice versa) because some components
   were already resolved via ldconfig before the env override took effect.

   Note this is also the error mode anyone on the host venv hits if they try to fix the
   host's cuDNN by prepending a newer pip cuDNN — same root pattern.

3. Consequence worth knowing even after fixing: cuDNN 9.17.1 does **not** fix the
   fused-attention unwritten-dk/dv bug (see
   `bugs/te_cudnn_fused_attn_unwritten_dkdv.md`) — both cuDNN versions in this image
   exhibit it. The packaging bug and the kernel bug are independent.

## Reproduction / verification

```bash
# Broken as shipped:
sudo docker run --rm --gpus device=0 slimerl/slime:latest \
  python3 -c "import torch; print(torch.backends.cudnn.version())"
# -> RuntimeError: cuDNN version incompatibility

# Show the duplicate installs:
sudo docker run --rm slimerl/slime:latest bash -c \
  'find / -name "libcudnn.so*" -not -path "/proc/*" 2>/dev/null'

# Fixed:
sudo docker run --rm --gpus device=0 slimerl/slime:latest bash -c \
  'rm -f /usr/lib/x86_64-linux-gnu/libcudnn*; ldconfig;
   python3 -c "import torch; print(torch.backends.cudnn.version())"'
# -> 91701
```

## Workaround / fix

- Runtime workaround (what `docker_run.sh` does at container start): delete the apt
  copies and refresh the linker cache:

  ```bash
  rm -f /usr/lib/x86_64-linux-gnu/libcudnn*; ldconfig
  ```

  Deleting is safer than `LD_LIBRARY_PATH` because it eliminates the mixed-sublibrary
  failure mode for every process, including ray workers that rebuild their own env.

- Proper fix for the image (report to slime / their sglang base): `apt-get remove -y
  'libcudnn9*'` in the Dockerfile, or align the apt version with the pip/torch one.

## How we ran into it

First attempt to run the NaN acceptance test in the freshly pulled image failed with
symptom 1; the `LD_LIBRARY_PATH` attempt then failed with symptom 2. Since torch, TE,
and sglang all sit on cuDNN in this stack, the image is unusable for training until one
of the two fixes above is applied — which suggests upstream CI either patches this in a
wrapper we don't see or doesn't exercise a cuDNN-initializing path on this image tag.
