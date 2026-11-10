# Bug Report: Transformer Engine PyTorch Extension Fails To Build In uv Environment

## Status: RESOLVED (2026-07-09)

TE 2.16.1 now builds from source and imports cleanly. The fix combined three
pieces:

1. **Pin TE to 2.16.1 instead of 2.17.0.** The missing header
   `torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp` is only needed by
   `csrc/extensions/ep.cpp`, which was added in TE 2.17.0. The 2.16.1 sdist has
   no reference to it. (Also: no prebuilt wheel could ever have worked here —
   TE publishes wheels only for cp312, and only against torch 2.8.0+cu129 or
   NGC torch builds; this venv is cp311 + torch 2.9.1+cu128.)

2. **Build against a matching CUDA 12.8 toolkit** instead of the system CUDA
   13.2. Installed side-by-side, toolkit-only, no root needed:

   ```bash
   sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit \
     --toolkitpath=/mnt/localssd/cuda-12.8 --no-man-page --override
   export CUDA_HOME=/mnt/localssd/cuda-12.8
   export PATH=${CUDA_HOME}/bin:${PATH}
   ```

3. **Force the cu12 core wheel.** The `transformer-engine-torch` sdist's
   metadata resolved to `transformer-engine-cu13`, whose
   `libtransformer_engine.so` needs `libcublasLt.so.13`
   (`undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`). Swap it:

   ```bash
   uv pip uninstall transformer-engine-cu13
   uv pip install --no-deps transformer-engine-cu12==2.16.1
   ```

The full build script is the same env shape as Attempt 2 below but with
`CUDA_HOME=/mnt/localssd/cuda-12.8` and `MAX_JOBS=64` (the machine has 208
CPUs / 1.8 TB RAM; `MAX_JOBS=1` was unnecessary), then:

```bash
uv pip install --no-build-isolation -v "transformer_engine[pytorch]==2.16.1"
```

Verified:

```text
python -c "import transformer_engine.pytorch, transformer_engine_torch"  # ok, 2.16.1
te.LayerNormLinear GPU forward pass                                      # ok
```

Note: importing TE requires the venv's pip-installed NVIDIA lib dirs on
`LD_LIBRARY_PATH` (the launch script already exports this).

Follow-ups applied after TE worked:

- Checkpoints were reconverted with the TE transformer impl to
  `/mnt/Qwen3-4B-Instruct-2507_torch_dist_te` and
  `/mnt/Qwen3-4B-Instruct-2507_slime_te` (the old local-impl checkpoints are
  kept for rollback).
- `run_qwen3_4B.sh` now uses the `_te` checkpoint paths, restores
  `--sequence-parallel`, and drops `--transformer-impl local`,
  `--no-rope-fusion`, `--no-persist-layer-norm`.
- `--gradient-accumulation-fusion` needs Apex (`fused_weight_gradient_mlp_cuda`),
  not TE — Apex was built separately from source against the same CUDA 12.8
  toolkit (`APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --no-build-isolation
  /mnt/localssd/apex`).

The original report follows.

## Summary

Installing `transformer_engine[pytorch]` in the `uv` managed Slime environment
fails while building `transformer-engine-torch` from source. The install first
failed because TE could not find `cudnn.h`. After explicitly adding the venv's
NVIDIA include/library paths, the build progressed farther but failed on a
missing PyTorch internal header:

```text
torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp
```

The current workaround is to run Slime/Megatron without Transformer Engine:

```bash
--transformer-impl local
--no-rope-fusion
--no-persist-layer-norm
--no-gradient-accumulation-fusion
```

and to disable:

```bash
--sequence-parallel
```

because torch LayerNorm fallback does not support sequence parallel in this
Megatron version.

## Environment

Machine:

```text
GCP H100 instance
```

Repo/env paths:

```bash
SLIME_REPO=/mnt/slime
VENV=/mnt/localssd/slime-py311-venv
MEGATRON=/mnt/Megatron-LM-slime
```

Python/PyTorch:

```text
Python 3.11.15
torch 2.9.1+cu128
torch CUDA 12.8
```

System CUDA compiler:

```text
/usr/local/cuda-13.2/bin/nvcc
Cuda compilation tools, release 13.2, V13.2.78
```

Important mismatch:

```text
PyTorch wheel: CUDA 12.8
System nvcc:   CUDA 13.2
```

Pip-installed NVIDIA headers/libs exist in the venv:

```bash
/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/nvidia/cudnn/include/cudnn.h
/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
```

## Attempt 1

Command shape:

```bash
source /mnt/localssd/slime-py311-venv/bin/activate
cd /mnt/slime

export CUDA_PATH=/usr/local/cuda-13.2
export CUDNN_PATH=/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/nvidia/cudnn
export CUDNN_HOME=${CUDNN_PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDNN_PATH}/lib:/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export MAX_JOBS=1
export NVTE_BUILD_THREADS_PER_JOB=1

uv pip install --no-build-isolation -v "transformer_engine[pytorch]"
```

Observed:

```text
Resolved 44 packages
Installed 2 packages
Building transformer-engine-torch==2.17.0
```

TE tried to use a prebuilt wheel:

```text
https://github.com/NVIDIA/TransformerEngine/releases/download/v2.17.0/transformer_engine_torch-2.17.0+cu12torch2.9.1+cu128cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

but GitHub returned:

```text
HTTP Error 404: Not Found
```

so it fell back to source build.

Failure:

```text
/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/torch/include/ATen/cudnn/cudnn-wrapper.h:3:10:
fatal error: cudnn.h: No such file or directory
    3 | #include <cudnn.h>
      |          ^~~~~~~~~
compilation terminated.
```

The header existed in the venv, but it was not on the compiler include path.

## Attempt 2

Retried with explicit include/library search paths from all pip-installed NVIDIA
packages:

```bash
source /mnt/localssd/slime-py311-venv/bin/activate
cd /mnt/slime

NVIDIA_ROOT=/mnt/localssd/slime-py311-venv/lib/python3.11/site-packages/nvidia
NVIDIA_LIB_DIRS=$(find ${NVIDIA_ROOT} -type d -name lib | paste -sd: -)
NVIDIA_INCLUDE_DIRS=$(find ${NVIDIA_ROOT} -type d -name include | paste -sd: -)

export CUDA_PATH=/usr/local/cuda-13.2
export CUDNN_PATH=${NVIDIA_ROOT}/cudnn
export CUDNN_HOME=${CUDNN_PATH}
export CPATH=${NVIDIA_INCLUDE_DIRS}:${CUDA_PATH}/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=${NVIDIA_INCLUDE_DIRS}:${CUDA_PATH}/include:${CPLUS_INCLUDE_PATH:-}
export LIBRARY_PATH=${NVIDIA_LIB_DIRS}:${CUDA_PATH}/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIRS}:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export MAX_JOBS=1
export NVTE_BUILD_THREADS_PER_JOB=1

uv pip install --no-build-isolation -v "transformer_engine[pytorch]"
```

This got past the `cudnn.h` error and compiled several objects, but failed at
`[10/33]`:

```text
/mnt/localssd/uv/sdists-v9/pypi/transformer-engine-torch/2.17.0/.../src/csrc/extensions/ep.cpp:23:10:
fatal error: torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp: No such file or directory
   23 | #include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

The source build then failed with:

```text
RuntimeError: Error compiling objects for extension
```

## Partial Install Cleanup

After the failed build, `transformer_engine` was partially importable, but the
torch extension was missing:

```text
transformer_engine      present
transformer_engine_torch None
```

Import failed with:

```text
FileNotFoundError: Could not find shared object file for Transformer Engine torch lib.
```

The partial install was removed:

```bash
uv pip uninstall --python /mnt/localssd/slime-py311-venv/bin/python \
  transformer-engine transformer-engine-torch transformer-engine-cu12 transformer-engine-cu13
```

Final clean state:

```text
transformer_engine None
transformer_engine_torch None
```

## Impact

Without TE/Apex, Megatron uses the local torch fallback. That means:

- `--sequence-parallel` cannot be used because torch LayerNorm asserts:

  ```text
  AssertionError: sequence parallel not supported by torch LayerNorm
  ```

- `--persist-layer-norm` cannot be used.
- `--gradient-accumulation-fusion` cannot be used.
- `--rope-fusion` cannot be used.

The tau-bench launch script has been patched to use the local fallback path.

## Current Workaround

In `/mnt/slime/examples/tau-bench/run_qwen3_4B.sh`, keep:

```bash
--transformer-impl local
--no-rope-fusion
--no-persist-layer-norm
--no-gradient-accumulation-fusion
```

and do not include:

```bash
--sequence-parallel
```

## Likely Root Causes

There are likely two independent issues:

1. No prebuilt TE wheel exists for the exact combo:

   ```text
   transformer-engine-torch 2.17.0
   Python cp311
   torch 2.9.1+cu128
   CUDA 12.x wheel tag
   cxx11abiTRUE
   linux_x86_64
   ```

   The guessed wheel URL returned 404.

2. Source build compatibility is fragile in this environment:

   ```text
   PyTorch is built for CUDA 12.8
   nvcc is CUDA 13.2
   TE source expects PyTorch internal header torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp
   ```

   That header is not present in the installed PyTorch wheel's include tree.

## Possible Fixes To Try Later

1. Use an NVIDIA NGC PyTorch container where TE is prebuilt with matching
   PyTorch/CUDA/cuDNN.

2. Use a PyTorch version for which TE publishes a matching prebuilt wheel.

3. Use a CUDA 12.8 toolkit instead of CUDA 13.2 when building against
   `torch==2.9.1+cu128`.

4. Try an older TE version that does not include/require:

   ```text
   torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp
   ```

5. Try building from TE source at a commit known to support the installed
   PyTorch wheel.

## Verification Commands

Check current no-TE state:

```bash
/mnt/localssd/slime-py311-venv/bin/python -c \
  "import importlib.util; print('transformer_engine', importlib.util.find_spec('transformer_engine')); print('transformer_engine_torch', importlib.util.find_spec('transformer_engine_torch'))"
```

Expected:

```text
transformer_engine None
transformer_engine_torch None
```

Check PyTorch/CUDA:

```bash
/mnt/localssd/slime-py311-venv/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda)"

nvcc --version
```
