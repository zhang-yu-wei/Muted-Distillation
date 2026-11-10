#!/usr/bin/env python3
"""Host-side Gemma4 export using Slime's Gemma mapping and existing DCP reader.

Unlike a permissive origin-fill conversion, require every language tensor to
come from the trained checkpoint. Only frozen non-language tensors may be
copied from the original HF model. Validate every name and shape before save.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors import safe_open
from safetensors.torch import save_file

from convert_qwen35_torch_dist_to_hf import (
    EmptyStateDictLoadPlanner, WrappedStorageReader, _named_params,
)


def gemma_converter():
    path = Path(__file__).resolve().parents[1] / 'third_party/slime/slime/backends/megatron_utils/megatron_to_hf/gemma4.py'
    spec = importlib.util.spec_from_file_location('officeqa_gemma4_mapping', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_gemma4_to_hf


def export(args, state, origin, output, chunk_size=5 * 1024**3):
    config = json.loads((origin / 'config.json').read_text())
    if config.get('model_type') != 'gemma4_unified':
        raise ValueError('Expected the Gemma4 Unified origin checkpoint')
    args.hf_checkpoint = str(origin)
    vocab_size = config['text_config']['vocab_size']
    convert = gemma_converter()
    tensors = {}
    for name, tensor in _named_params(args, state):
        if name in ('module.module.embedding.word_embeddings.weight', 'module.module.output_layer.weight'):
            tensor = tensor[:vocab_size]
        for key, value in convert(args, name, tensor):
            if key in tensors:
                if not torch.equal(tensors[key], value):
                    raise ValueError(f'Conflicting tied/duplicate tensor: {key}')
                continue
            tensors[key] = value.contiguous()
    trained_keys = set(tensors)
    expected = set()
    for shard in sorted(origin.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                expected.add(key)
                if key not in tensors:
                    if key.startswith('model.language_model.'):
                        raise ValueError(f'Missing trained language tensor: {key}')
                    tensors[key] = handle.get_tensor(key)
                if list(tensors[key].shape) != handle.get_slice(key).get_shape():
                    raise ValueError(f'Converted shape differs from origin: {key}')
    if set(tensors) != expected:
        raise ValueError(f'Unexpected converted keys: {set(tensors) - expected}')
    output.mkdir(parents=True, exist_ok=False)
    shards, size = [{}], 0
    for key, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if shards[-1] and size + nbytes > chunk_size:
            shards.append({})
            size = 0
        shards[-1][key] = tensor
        size += nbytes
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f'model-{i+1:05d}-of-{len(shards):05d}.safetensors'
        save_file(shard, output / name)
        weight_map.update({key: name for key in shard})
        print(f'Saved {name}', flush=True)
    for source in origin.iterdir():
        if source.is_file() and source.suffix != '.safetensors' and source.name != 'model.safetensors.index.json':
            shutil.copy2(source, output / source.name)
    (output / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': sum(v.numel() * v.element_size() for v in tensors.values())},
        'weight_map': weight_map,
    }, indent=2) + '\n')
    return dict(trained_tensors=len(trained_keys), frozen_tensors=len(expected - trained_keys),
                origin_config_sha256=hashlib.sha256((origin / 'config.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--origin-hf-dir', type=Path, required=True)
    cli = parser.parse_args()
    if cli.output_dir.exists():
        raise ValueError('Refusing to overwrite an existing export')
    args = torch.load(cli.input_dir / 'common.pt', weights_only=False)['args']
    state = {}
    print(f'Loading trained Gemma weights: {cli.input_dir}', flush=True)
    dist_cp.state_dict_loader._load_state_dict(
        state, storage_reader=WrappedStorageReader(str(cli.input_dir)),
        planner=EmptyStateDictLoadPlanner(), no_dist=True,
    )
    report = export(args, state, cli.origin_hf_dir, cli.output_dir)
    report.update(source=str(cli.input_dir), origin=str(cli.origin_hf_dir), mapping='slime gemma4.py')
    (cli.output_dir / 'conversion_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Validated Gemma export', report, flush=True)


if __name__ == '__main__':
    main()
