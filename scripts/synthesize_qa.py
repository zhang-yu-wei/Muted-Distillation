"""Author synthetic QA pairs for SFT from the Treasury Bulletin corpus.

Implements the first two steps of the SFT data design: stratified table-region
sampling and Gemini question authoring. This is deliberately author-only:
answer validation happens later by retaining only teacher traces (GLM-5.2 on
the local server) whose submitted result agrees with Gemini's answer. Outputs use the benchmark CSV schema plus
a provenance JSONL and manifest.

Example (pilot):
    uv run scripts/synthesize_qa.py --n-questions 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from officeqa_rl import synthesis
from officeqa_rl.dataset import read_csv_rows
from officeqa_rl.gemini_author import GeminiAuthor

PROMPT_VERSION = "authoring-v1"
CSV_COLUMNS = ("uid", "question", "answer", "source_docs", "source_files", "difficulty")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/treasury_bulletins_parsed/transformed"),
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        action="append",
        help="Benchmark CSV(s) supplying style exemplars; default: "
        "data/officeqa_full.csv (which contains the pro subset)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--prefix", default="officeqa_synth_v0")
    parser.add_argument("--n-questions", type=int, default=60, help="Candidates to author")
    parser.add_argument(
        "--tier-mix",
        default="T1=0.3,T2=0.45,T3=0.25",
        help="Authored-candidate mix over difficulty tiers",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--author-model", default="gemini-3.5-flash")
    parser.add_argument("--author-temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-regions-per-file", type=int, default=2)
    parser.add_argument("--n-exemplars", type=int, default=4)
    return parser.parse_args(argv)


def parse_tier_mix(spec: str) -> dict[str, float]:
    mix: dict[str, float] = {}
    for part in spec.split(","):
        tier, _, fraction_text = part.partition("=")
        tier = tier.strip()
        if tier not in synthesis.DIFFICULTY_BY_TIER:
            raise ValueError(f"Unknown tier in mix: {tier!r}")
        mix[tier] = float(fraction_text)
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"Tier mix must sum to 1.0: {spec!r}")
    return mix


def tier_assignments(n: int, mix: dict[str, float], rng: random.Random) -> list[str]:
    tiers = sorted(mix)
    counts = {tier: int(mix[tier] * n) for tier in tiers}
    while sum(counts.values()) < n:
        counts[rng.choice(tiers)] += 1
    assignments = [tier for tier in tiers for _ in range(counts[tier])]
    rng.shuffle(assignments)
    return assignments


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CandidateJob:
    candidate_id: int
    tier: str
    regions: list[synthesis.TableRegion]

    def source_files(self) -> list[str]:
        return [region.source_file for region in self.regions]


@dataclass
class CandidateResult:
    job: CandidateJob
    accepted: bool = False
    rejection: str | None = None
    authored: dict[str, Any] | None = None
    uid: str | None = None
    author_status: dict[str, Any] = field(default_factory=dict)
    author_raw: str = ""


class Synthesizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        benchmark_csvs = args.benchmark_csv or [Path("data/officeqa_full.csv")]
        self.benchmark_csvs = benchmark_csvs
        self.benchmark_rows: list[dict[str, str]] = []
        for csv_path in benchmark_csvs:
            self.benchmark_rows.extend(read_csv_rows(csv_path))
        self.author = GeminiAuthor(model=args.author_model)
        self._file_text_cache: dict[str, str] = {}

    # ---- corpus access -----------------------------------------------------

    def file_text(self, source_file: str) -> str:
        if source_file not in self._file_text_cache:
            path = self.args.corpus_dir / source_file
            self._file_text_cache[source_file] = path.read_text(errors="replace")
        return self._file_text_cache[source_file]

    # ---- region sampling (Step 1) -------------------------------------------

    def sample_jobs(self) -> list[CandidateJob]:
        files = sorted(path.name for path in self.args.corpus_dir.glob("*.txt"))
        if not files:
            raise SystemExit(f"No corpus files under {self.args.corpus_dir}")
        by_decade: dict[str, list[str]] = {}
        for name in files:
            by_decade.setdefault(synthesis.decade_of(name), []).append(name)
        # Oversample files: not every file yields usable regions.
        picked_files = synthesis.round_robin_sample(
            by_decade, min(len(files), self.args.n_questions * 3), self.rng
        )

        pool: list[synthesis.TableRegion] = []
        regions_by_decade: dict[str, list[synthesis.TableRegion]] = {}
        for name in picked_files:
            regions = synthesis.extract_table_regions(self.file_text(name), name)
            self.rng.shuffle(regions)
            for region in regions[: self.args.max_regions_per_file]:
                pool.append(region)
                regions_by_decade.setdefault(synthesis.decade_of(name), []).append(region)

        if len(pool) < self.args.n_questions:
            raise SystemExit(
                f"Only {len(pool)} qualifying table regions found for "
                f"{self.args.n_questions} questions; increase file oversampling"
            )
        primaries = synthesis.round_robin_sample(
            regions_by_decade, self.args.n_questions, self.rng
        )
        tiers = tier_assignments(
            self.args.n_questions, parse_tier_mix(self.args.tier_mix), self.rng
        )

        jobs = []
        for candidate_id, (region, tier) in enumerate(zip(primaries, tiers), start=1):
            regions = [region]
            if tier == "T3":
                pair = synthesis.find_cross_year_pair(region, pool)
                if pair is None:
                    tier = "T2"  # no same-family table in another year; degrade
                else:
                    regions.append(pair)
            jobs.append(CandidateJob(candidate_id=candidate_id, tier=tier, regions=regions))
        return jobs

    # ---- authoring (Step 2) --------------------------------------------------

    def exemplars(self) -> list[str]:
        # Style exemplars come from full-minus-pro only (never pro), so the
        # authored style cannot drift toward the held-out eval set.
        pro_uids = {
            (row.get("uid") or "").strip()
            for row in read_csv_rows(Path("data/officeqa_pro.csv"))
        }
        eligible = [
            row["question"]
            for row in self.benchmark_rows
            if (row.get("uid") or "").strip() not in pro_uids
        ]
        return self.rng.sample(eligible, min(self.args.n_exemplars, len(eligible)))

    def run_candidate(self, job: CandidateJob, exemplars: list[str]) -> CandidateResult:
        result = CandidateResult(job=job)
        excerpts = [(region.source_file, region.excerpt()) for region in job.regions]
        prompt = synthesis.build_authoring_prompt(excerpts, job.tier, exemplars)
        try:
            result.author_raw = self.author.generate(
                prompt, temperature=self.args.author_temperature
            )
            authored = synthesis.parse_authored_json(result.author_raw)
        except Exception as error:  # noqa: BLE001 - recorded, not fatal
            result.rejection = "author_parse_error"
            result.author_status = {"parsed": False, "detail": str(error)}
            return result
        if authored.get("skip"):
            result.rejection = "author_skip"
            result.author_status = {
                "parsed": True,
                "accepted": False,
                "detail": "author declined region",
            }
            return result
        result.authored = authored
        result.author_status = {"parsed": True, "accepted": True}
        result.accepted = True
        return result

    # ---- output (Step 2 outputs) ---------------------------------------------

    def write_outputs(self, results: list[CandidateResult]) -> dict[str, Any]:
        import csv as csv_module

        out_dir = self.args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        accepted = [result for result in results if result.accepted]

        csv_path = out_dir / f"{self.args.prefix}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv_module.writer(file)
            writer.writerow(CSV_COLUMNS)
            for index, result in enumerate(accepted, start=1):
                authored = result.authored or {}
                uid = f"SYN{index:04d}"
                result.uid = uid
                regions = result.job.regions
                writer.writerow(
                    [
                        uid,
                        authored["question"],
                        authored["answer"],
                        ";".join(
                            f"synthetic:{region.source_file}"
                            f"#L{region.start_line}-L{region.end_line}"
                            for region in regions
                        ),
                        "\n".join(result.job.source_files()),
                        synthesis.DIFFICULTY_BY_TIER[result.job.tier],
                    ]
                )

        provenance_path = out_dir / f"{self.args.prefix}_provenance.jsonl"
        rejections: dict[str, int] = {}
        with provenance_path.open("w", encoding="utf-8") as file:
            for result in results:
                if result.rejection:
                    rejections[result.rejection] = rejections.get(result.rejection, 0) + 1
                record = {
                    "candidate_id": result.job.candidate_id,
                    "uid": result.uid,
                    "accepted": result.accepted,
                    "rejection": result.rejection,
                    "tier": result.job.tier,
                    "source_files": result.job.source_files(),
                    "regions": [region.span() for region in result.job.regions],
                    "authored": result.authored,
                    "author_status": result.author_status,
                    "author_model": self.args.author_model,
                    "prompt_version": PROMPT_VERSION,
                    "author_raw": result.author_raw[:2000],
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest = {
            "prefix": self.args.prefix,
            "seed": self.args.seed,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_dir": str(self.args.corpus_dir.resolve()),
            "benchmark_csvs": {
                str(path): sha256_of_file(path) for path in self.benchmark_csvs
            },
            "author_model": self.args.author_model,
            "prompt_version": PROMPT_VERSION,
            "mode": "author-only",
            "tier_mix": self.args.tier_mix,
            "authored_candidates": len(results),
            "accepted_rows": len(accepted),
            "rejections": dict(sorted(rejections.items())),
            "accepted_by_tier": {
                tier: sum(1 for r in accepted if r.job.tier == tier)
                for tier in sorted(synthesis.DIFFICULTY_BY_TIER)
            },
            "outputs": {
                "csv": {"file": csv_path.name, "rows": len(accepted), "sha256": sha256_of_file(csv_path)},
                "provenance": {
                    "file": provenance_path.name,
                    "rows": len(results),
                    "sha256": sha256_of_file(provenance_path),
                },
            },
        }
        manifest_path = out_dir / f"{self.args.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def run(self) -> dict[str, Any]:
        jobs = self.sample_jobs()
        exemplars = self.exemplars()
        print(
            f"Authoring {len(jobs)} candidates with {self.args.author_model} "
            "(author-only; answer validation deferred to Step 6)"
        )
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
            results = list(
                executor.map(lambda job: self.run_candidate(job, exemplars), jobs)
            )
        manifest = self.write_outputs(results)
        print(
            f"accepted {manifest['accepted_rows']}/{manifest['authored_candidates']} "
            f"(by tier: {manifest['accepted_by_tier']}); rejections: {manifest['rejections']}"
        )
        print(f"outputs in {self.args.output_dir}/{self.args.prefix}*")
        return manifest


def main(argv: list[str] | None = None) -> None:
    Synthesizer(parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
