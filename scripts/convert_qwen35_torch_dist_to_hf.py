#!/usr/bin/env python3
"""Convert a Qwen3.5 Megatron torch.distributed checkpoint to HF format.

This is the Qwen3.5-only, host-runnable counterpart of Slime's generic
``convert_torch_dist_to_hf.py``.  It deliberately avoids importing Slime's
serving/quantization stack, which is unnecessary for BF16 evaluation
checkpoints and otherwise requires the much larger training container.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import shutil
import time
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp


class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, mod_name: str, name: str):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        # Training checkpoints may pickle configuration-only classes from the
        # training stack. Conversion only reads their attributes, so loading
        # them as attribute-bearing dummies avoids requiring Slime/Megatron on
        # otherwise lightweight evaluation hosts.
        if mod_name.startswith(("megatron", "glm", "slime")):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for key, value in metadata.state_dict_metadata.items():
            if "optimizer" in key or "_state" in key:
                continue
            print(f"find {key} in torch_dist ckpt")
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_qwen_converter():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "third_party/slime/slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py"
    )
    spec = importlib.util.spec_from_file_location("officeqa_qwen3_5_converter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Qwen3.5 converter from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_qwen3_5_to_hf


CONVERT_QWEN35_TO_HF = _load_qwen_converter()


def _get_expert_params(args, name, param):
    if ".experts." not in name:
        yield name, param
        return
    match = re.search(r"mlp.experts.(.+).weight(\d+)", name)
    if match:
        yield name, param
        return
    assert param.shape[0] == args.num_experts
    for expert_id in range(args.num_experts):
        yield name.replace(".experts.experts.", ".experts.") + str(expert_id), param[expert_id]


def _get_layer_params(args, name, param):
    if ".layers." not in name or re.search(r".layers.(\d+).", name):
        yield from _get_expert_params(args, name, param)
        return
    assert param.shape[0] == args.num_layers
    for layer_id in range(args.num_layers):
        layer_name = name.replace(".layers.", f".layers.{layer_id}.")
        yield from _get_expert_params(args, layer_name, param[layer_id])


def _named_params(args, state_dict):
    for name, param in state_dict.items():
        yield from _get_layer_params(args, f"module.module.{name}", param)


def _save_tensors(args, state_dict, output_dir: Path, chunk_size: int, origin_hf_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    current_size = 0
    total_size = 0
    shards = [{}]
    converted_names: set[str] = set()
    vocab_size = getattr(args, "vocab_size", None)

    for name, param in _named_params(args, state_dict):
        if vocab_size and name.removeprefix("module.module.") in {
            "embedding.word_embeddings.weight",
            "output_layer.weight",
        }:
            param = param[:vocab_size]
        for converted_name, converted_param in CONVERT_QWEN35_TO_HF(args, name, param):
            tensor_size = converted_param.numel() * converted_param.element_size()
            if shards[-1] and tensor_size + current_size > chunk_size:
                shards.append({})
                current_size = 0
            shards[-1][converted_name] = converted_param
            converted_names.add(converted_name)
            current_size += tensor_size
            total_size += tensor_size

    for filename in sorted(origin_hf_dir.glob("*.safetensors")):
        with safetensors.safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in converted_names:
                    continue
                tensor = handle.get_tensor(key)
                tensor_size = tensor.numel() * tensor.element_size()
                if shards[-1] and tensor_size + current_size > chunk_size:
                    shards.append({})
                    current_size = 0
                shards[-1][key] = tensor
                converted_names.add(key)
                current_size += tensor_size
                total_size += tensor_size

    weight_map = {}
    for index, tensors in enumerate(shards):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        weight_map.update({key: filename for key in tensors})
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    )
    for index, tensors in enumerate(shards):
        filename = output_dir / f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        started = time.time()
        safetensors.torch.save_file(tensors, filename)
        print(f"{filename.name} saved in {time.time() - started:.2f} sec")

    for source in origin_hf_dir.iterdir():
        if not source.is_file() or source.suffix == ".safetensors" or source.name == "model.safetensors.index.json":
            continue
        shutil.copy(source, output_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")

    print(f"loading model from {args.input_dir}")
    started = time.time()
    megatron_args = torch.load(args.input_dir / "common.pt", weights_only=False)["args"]
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(args.input_dir)),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time() - started:.2f} sec")
    _save_tensors(megatron_args, state_dict, args.output_dir, args.chunk_size, args.origin_hf_dir)


if __name__ == "__main__":
    main()
