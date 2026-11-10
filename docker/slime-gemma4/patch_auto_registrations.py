"""Make backfilled HF auto-registrations idempotent.

sglang and megatron.bridge both pin transformers 5.8.1 and backfill configs
that version lacks. transformers 5.13 (needed to parse gemma4_unified) ships
those configs natively, so each backfilled register() raises "already used by
a Transformers config". Bounded set: 3 calls in megatron.bridge, 2 in sglang.
"""

import pathlib
import re

TARGETS = [
    "/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py",
    "/sgl-workspace/sglang/python/sglang/srt/configs/qwen3_asr.py",
]

PATTERN = re.compile(
    r"(Auto(?:Config|Model|Processor|ModelForCausalLM)\.register\((?:[^()]|\([^()]*\))*?)\)(\s*)$",
    re.M,
)

total = 0
for target in TARGETS:
    path = pathlib.Path(target)
    if not path.exists():
        print(f"skip (absent): {target}")
        continue
    source = path.read_text()
    patched = PATTERN.sub(r"\1, exist_ok=True)\2", source)
    path.write_text(patched)
    count = patched.count("exist_ok=True")
    total += count
    print(f"patched {count} registrations in {target}")

assert total >= 5, f"expected >=5 patched registrations, got {total}"
