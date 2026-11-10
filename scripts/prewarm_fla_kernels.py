"""Pre-warm the Triton autotune cache for Qwen3.5's full FLA GDN stack.

The training path is:

    ShortConvolution -> gated delta rule (including Q/K L2 norm)
                     -> FusedRMSNormGated

Several kernels in that path have different autotune keys.  The gated-delta
rule sweep is effectively sequence-length independent, but ShortConvolution
keys on the number of packed sequences, and the L2/fused-norm kernels key on
a bucket derived from the flattened token count.  A single two-sequence 8k
pass therefore does not cover the shapes used by the 64k training recipe.

The default plan first runs that small pass so the memory-hungry GDR sweep is
cached, then covers the smoke-run packed counts (1-3), the full-run counts
observed with GLOBAL_BS=64 (32-65, including a trailing pad segment), and all
normalization buckets from half to full context plus one CP/padding overflow
bucket.

Run inside the training container with nothing else on the GPU:

    sudo docker exec -e TRITON_CACHE_DIR=/mnt/officeqa-rl/results/triton_cache \
      -e CUDA_VISIBLE_DEVICES=0 -w /mnt/officeqa-rl officeqa-train \
      python3 scripts/prewarm_fla_kernels.py

Inspect the cases without touching CUDA with ``--dry-run``. Re-run after
wiping the cache or changing the FLA, Triton, model, TP, or CP versions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


AUTOTUNE_ROWS_PER_BUCKET = 2048 * 32
DEFAULT_PACKED_SEQUENCE_COUNTS = "1-3,32-65"
PREWARM_MANIFEST = ".officeqa_fla_prewarm_complete.json"


@dataclass(frozen=True)
class LinearAttentionGeometry:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    activation: str
    norm_epsilon: float

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


@dataclass(frozen=True)
class PrewarmCase:
    total_tokens: int
    packed_sequences: int
    norm_bucket: int


def linear_attn_geometry(hf_checkpoint: str) -> LinearAttentionGeometry:
    with open(f"{hf_checkpoint}/config.json") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    geometry = LinearAttentionGeometry(
        num_k_heads=int(text_config["linear_num_key_heads"]),
        num_v_heads=int(text_config["linear_num_value_heads"]),
        head_k_dim=int(text_config["linear_key_head_dim"]),
        head_v_dim=int(text_config["linear_value_head_dim"]),
        conv_kernel_size=int(text_config["linear_conv_kernel_dim"]),
        activation=str(text_config["hidden_act"]),
        norm_epsilon=float(text_config["rms_norm_eps"]),
    )
    if geometry.num_v_heads % geometry.num_k_heads:
        raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
    return geometry


def linear_attn_dims(hf_checkpoint: str) -> tuple[int, int, int]:
    """Retain the original helper's public result for callers that use it."""
    geometry = linear_attn_geometry(hf_checkpoint)
    return geometry.num_v_heads, geometry.head_k_dim, geometry.head_v_dim


def parse_int_ranges(spec: str) -> list[int]:
    """Parse a comma-separated list containing integers and inclusive ranges."""
    values: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            fields = item.split("-")
            if len(fields) != 2:
                raise argparse.ArgumentTypeError(f"invalid integer range: {item!r}")
            try:
                start, end = (int(field) for field in fields)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid integer range: {item!r}") from exc
            if start > end:
                raise argparse.ArgumentTypeError(f"descending integer range: {item!r}")
            values.update(range(start, end + 1))
        else:
            try:
                values.add(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid integer: {item!r}") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("packed-sequence counts must be positive")
    return sorted(values)


def norm_bucket(total_tokens: int, num_v_heads: int) -> int:
    """Return the FLA L2/fused-norm NB autotune key for this token count."""
    flattened_rows = total_tokens * num_v_heads
    return (flattened_rows + AUTOTUNE_ROWS_PER_BUCKET - 1) // AUTOTUNE_ROWS_PER_BUCKET


def token_lengths_for_norm_buckets(
    min_tokens: int,
    max_tokens: int,
    num_v_heads: int,
    padding_buckets: int,
) -> list[int]:
    """Choose the smallest token count that exercises each relevant NB key."""
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("expected 1 <= min_tokens <= max_tokens")
    if padding_buckets < 0:
        raise ValueError("padding_buckets must be non-negative")

    first_bucket = norm_bucket(min_tokens, num_v_heads)
    last_bucket = norm_bucket(max_tokens, num_v_heads) + padding_buckets
    lengths = []
    for bucket in range(first_bucket, last_bucket + 1):
        # NB=ceil(T*H/65536). Pick the first T belonging to this bucket;
        # retain min_tokens for the first bucket so the intended lower bound
        # is represented rather than a shorter equivalent shape.
        first_token_in_bucket = ((bucket - 1) * AUTOTUNE_ROWS_PER_BUCKET) // num_v_heads + 1
        lengths.append(max(min_tokens, first_token_in_bucket))
    return lengths


def build_case_plan(
    token_lengths: list[int],
    packed_sequence_counts: list[int],
    num_v_heads: int,
) -> list[PrewarmCase]:
    """Pair two independent autotune key spaces without taking a cross product."""
    if not token_lengths or not packed_sequence_counts:
        raise ValueError("token lengths and packed-sequence counts must be non-empty")
    case_count = max(len(token_lengths), len(packed_sequence_counts))
    return [
        PrewarmCase(
            total_tokens=token_lengths[index % len(token_lengths)],
            packed_sequences=packed_sequence_counts[index % len(packed_sequence_counts)],
            norm_bucket=norm_bucket(token_lengths[index % len(token_lengths)], num_v_heads),
        )
        for index in range(case_count)
    ]


def balanced_cu_seqlens(total_tokens: int, packed_sequences: int) -> list[int]:
    """Build monotonically increasing packed boundaries with no empty segment."""
    if packed_sequences < 1 or total_tokens < packed_sequences:
        raise ValueError("total_tokens must be at least packed_sequences")
    base_length, longer_sequences = divmod(total_tokens, packed_sequences)
    boundaries = [0]
    for index in range(packed_sequences):
        boundaries.append(boundaries[-1] + base_length + (index < longer_sequences))
    return boundaries


def run_case(torch, chunk_gated_delta_rule, conv, norm, geometry, case, label) -> None:
    """Run the exact nonlinear kernel stack once in forward and backward."""
    T = case.total_tokens
    H = geometry.num_v_heads
    cu_seqlens = torch.tensor(
        balanced_cu_seqlens(T, case.packed_sequences),
        device="cuda",
        dtype=torch.int32,
    )
    print(
        f"[{label}] T={T} packed={case.packed_sequences} NB={case.norm_bucket}",
        flush=True,
    )

    mixed_qkv = torch.randn(
        1,
        T,
        geometry.conv_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    conv_out, _ = conv(x=mixed_qkv, cu_seqlens=cu_seqlens)
    query, key, value = torch.split(
        conv_out,
        [geometry.key_dim, geometry.key_dim, geometry.value_dim],
        dim=-1,
    )
    query = query.reshape(1, T, geometry.num_k_heads, geometry.head_k_dim)
    key = key.reshape(1, T, geometry.num_k_heads, geometry.head_k_dim)
    value = value.reshape(1, T, H, geometry.head_v_dim)
    head_ratio = H // geometry.num_k_heads
    if head_ratio > 1:
        query = query.repeat_interleave(head_ratio, dim=2)
        key = key.repeat_interleave(head_ratio, dim=2)

    g = (-torch.rand(1, T, H, device="cuda", dtype=torch.float32) - 0.1).requires_grad_()
    beta = torch.rand(
        1,
        T,
        H,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    core, _ = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    z = torch.randn(
        1,
        T,
        H,
        geometry.head_v_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = norm(core.reshape(-1, geometry.head_v_dim), z.reshape(-1, geometry.head_v_dim))
    output.float().sum().backward()
    torch.cuda.synchronize()

    del beta, conv_out, core, cu_seqlens, g, key, mixed_qkv, output, query, value, z
    conv.zero_grad(set_to_none=True)
    norm.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--hf-checkpoint", default="/mnt/Qwen3.5-4B")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "TRITON_CACHE_DIR",
            str(Path(__file__).resolve().parents[1] / "results" / "triton_cache"),
        ),
        help="shared Triton cache directory (default: TRITON_CACHE_DIR or results/triton_cache)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=65536,
        help="maximum unpadded training context length (default: 65536)",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=None,
        help="lower token-count coverage bound (default: half of --seq-len)",
    )
    parser.add_argument(
        "--padding-buckets",
        type=int,
        default=1,
        help="extra norm buckets above --seq-len for CP/packing padding (default: 1)",
    )
    parser.add_argument(
        "--packed-sequence-counts",
        type=parse_int_ranges,
        default=parse_int_ranges(DEFAULT_PACKED_SEQUENCE_COUNTS),
        metavar="RANGES",
        help=f"counts/ranges to cover (default: {DEFAULT_PACKED_SEQUENCE_COUNTS})",
    )
    parser.add_argument(
        "--bootstrap-seq-len",
        type=int,
        default=8192,
        help="small first pass that caches the memory-heavy GDR sweep (default: 8192)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print geometry and coverage plan without importing CUDA/FLA",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    geometry = linear_attn_geometry(args.hf_checkpoint)
    min_seq_len = args.min_seq_len if args.min_seq_len is not None else args.seq_len // 2
    token_lengths = token_lengths_for_norm_buckets(
        min_seq_len,
        args.seq_len,
        geometry.num_v_heads,
        args.padding_buckets,
    )
    cases = build_case_plan(token_lengths, args.packed_sequence_counts, geometry.num_v_heads)
    bootstrap_tokens = min(args.bootstrap_seq_len, min_seq_len)
    bootstrap_sequences = min(2, bootstrap_tokens)
    bootstrap = PrewarmCase(
        total_tokens=bootstrap_tokens,
        packed_sequences=bootstrap_sequences,
        norm_bucket=norm_bucket(bootstrap_tokens, geometry.num_v_heads),
    )

    print(
        "geometry: "
        f"K_heads={geometry.num_k_heads} V_heads={geometry.num_v_heads} "
        f"K_dim={geometry.head_k_dim} V_dim={geometry.head_v_dim} "
        f"conv_dim={geometry.conv_dim} kernel={geometry.conv_kernel_size}"
    )
    print(
        f"bootstrap: T={bootstrap.total_tokens} packed={bootstrap.packed_sequences} "
        f"NB={bootstrap.norm_bucket}"
    )
    print(
        f"coverage: {len(cases)} cases; packed={args.packed_sequence_counts}; "
        f"NB={sorted({case.norm_bucket for case in cases})}; "
        f"T={sorted({case.total_tokens for case in cases})}"
    )
    print(f"cache: {args.cache_dir}")
    if args.dry_run:
        return 0

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # FLA reads the cache path while its Triton modules are imported.
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    import torch
    import triton
    import fla
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --dry-run is used")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free: {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    if free < 0.9 * total:
        print(
            "WARNING: GPU is not idle; Triton benchmark sweeps may OOM. "
            "Stop resident ray/sglang/training processes first."
        )

    torch.manual_seed(0)
    conv = ShortConvolution(
        hidden_size=geometry.conv_dim,
        kernel_size=geometry.conv_kernel_size,
        bias=False,
        activation=geometry.activation,
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm = FusedRMSNormGated(
        geometry.head_v_dim,
        eps=geometry.norm_epsilon,
        activation=geometry.activation,
        device="cuda",
        dtype=torch.bfloat16,
    )

    run_case(torch, chunk_gated_delta_rule, conv, norm, geometry, bootstrap, "bootstrap")
    for index, case in enumerate(cases, start=1):
        run_case(torch, chunk_gated_delta_rule, conv, norm, geometry, case, f"{index}/{len(cases)}")
    manifest = {
        "format_version": 1,
        "hf_checkpoint": str(Path(args.hf_checkpoint).resolve()),
        "geometry": asdict(geometry),
        "bootstrap": asdict(bootstrap),
        "cases": [asdict(case) for case in cases],
        "versions": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "fla": getattr(fla, "__version__", "unknown"),
        },
    }
    manifest_path = cache_dir / PREWARM_MANIFEST
    temporary_manifest_path = manifest_path.with_suffix(".tmp")
    with open(temporary_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary_manifest_path, manifest_path)
    print(f"full FLA forward/backward autotune coverage is warm; wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
