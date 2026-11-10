"""Failure-pattern analysis over full-dataset eval result directories.

Classifies every attempt (``UID*.s*.json``, falling back to ``UID*.json``
when no multi-sample files exist) by outcome, format-error turns, command
mix, and the numeric distance of wrong submissions, then prints per-run
aggregates and cross-run question overlap.

Usage:
    uv run scripts/analyze_eval_failures.py \
        results/vllm-Qwen3.5-4B results/vllm-fanoutfix-iter139 \
        results/gemini35-flash-keypool
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_num(s):
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace("%", "").strip()
    if not _NUM_RE.fullmatch(s.replace(" ", "")):
        return None
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def classify_wrong(label, submission):
    lv, sv = parse_num(label), parse_num(submission)
    if sv is None:
        return "non_numeric_submission"
    if lv is None:
        return "label_not_numeric"
    if lv == 0:
        return "other_numeric"
    if abs(sv - lv) / abs(lv) <= 0.05:
        return "close_miss_le5pct"
    if sv != 0 and abs(sv + lv) / abs(lv) <= 0.02:
        return "sign_flip"
    for power in (1e6, 1e3, 100, 10):
        for ratio in (power, 1 / power):
            if sv != 0 and abs(sv / lv - ratio) / ratio <= 0.02:
                return "scale_off_by_power_of_10"
    return "other_numeric"


def cmd_kind(command):
    head = command.strip().split()[0] if command.split() else ""
    if head == "submit":
        return "submit"
    if head in ("grep", "rg", "zgrep"):
        return "grep"
    if head in ("cat", "head", "tail", "sed", "awk", "less"):
        return "read"
    if head in ("ls", "find", "wc", "file", "stat"):
        return "browse"
    if head.startswith("python"):
        return "python"
    return "other"


def analyze_attempt(path):
    r = json.load(open(path))
    msgs = r["messages"]
    fmt_errs = sum(
        1
        for m in msgs
        if m["role"] == "user"
        and str(m.get("content", "")).startswith("Format error:")
    )
    commands = [
        c["command"] for c in r["metadata"]["environment"]["command_history"]
    ]
    dup = 0.0
    if commands:
        counts = Counter(commands)
        dup = sum(n - 1 for n in counts.values()) / len(commands)
    kinds = Counter(cmd_kind(c) for c in commands)
    single_kw_greps = 0
    for c in commands:
        if cmd_kind(c) == "grep":
            pats = re.findall(r'["\x27]([^"\x27]+)["\x27]', c)
            if len(pats[0].split()) <= 1 if pats else True:
                single_kw_greps += 1
    sources = [
        s.strip()
        for s in str(r["metadata"]["environment"].get("source_files", "")).split(",")
        if s.strip()
    ]
    touched_src = (
        any(any(s in c for s in sources) for c in commands) if sources else None
    )
    turns = len(r["metadata"].get("generated_turns") or []) or r["steps"]
    return dict(
        uid=r["uid"],
        status=r["status"],
        score=r["score"],
        steps=r["steps"],
        fmt_errs=fmt_errs,
        n_turns=turns,
        dup=dup,
        kinds=kinds,
        single_kw_greps=single_kw_greps,
        touched_src=touched_src,
        wrong_kind=(
            classify_wrong(r["ground_truth"], r["submission"])
            if r["status"] == "submitted" and r["score"] == 0.0
            else None
        ),
    )


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def report(name, rows):
    n = len(rows)
    submitted = [r for r in rows if r["status"] == "submitted"]
    limited = [r for r in rows if r["status"] != "submitted"]
    wrong = [r for r in submitted if r["score"] == 0.0]
    correct = [r for r in submitted if r["score"] == 1.0]
    print(f"\n===== {name}  attempts={n}")
    print(
        f" submitted {len(submitted)} ({pct(len(submitted), n)}); "
        f"correct {len(correct)}; wrong {len(wrong)}; "
        f"step-limit {len(limited)} ({pct(len(limited), n)})"
    )
    wrong_kinds = Counter(r["wrong_kind"] for r in wrong)
    print(f" wrong-answer breakdown: {dict(wrong_kinds.most_common())}")
    kinds = Counter()
    for r in rows:
        kinds.update(r["kinds"])
    total_cmds = sum(kinds.values())
    print(
        f" command mix ({total_cmds} commands): "
        f"{ {k: pct(v, total_cmds) for k, v in kinds.most_common()} }"
    )
    greps = kinds["grep"]
    if greps:
        single = sum(r["single_kw_greps"] for r in rows)
        print(f" greps: {pct(greps, total_cmds)} of commands, single-keyword {pct(single, greps)}")
    for group, gname in ((rows, "all"), (limited, "step-limit"), (wrong, "wrong"), (correct, "correct")):
        if not group:
            continue
        fmt_attempts = sum(1 for r in group if r["fmt_errs"] > 0)
        fmt_turns = sum(r["fmt_errs"] for r in group)
        turns = sum(r["n_turns"] for r in group)
        dup_high = sum(1 for r in group if r["dup"] > 0.3)
        median_steps = sorted(r["steps"] for r in group)[len(group) // 2]
        touched = [r["touched_src"] for r in group if r["touched_src"] is not None]
        print(
            f"  [{gname:10}] n={len(group):4} "
            f"fmt-err attempts {pct(fmt_attempts, len(group))}, "
            f"fmt-err turns {pct(fmt_turns, turns)}, "
            f"dup>30% attempts {pct(dup_high, len(group))}, "
            f"median steps {median_steps}, "
            f"touched-src {pct(sum(touched), len(touched)) if touched else '-'}"
        )
    per_question = defaultdict(list)
    for r in rows:
        per_question[r["uid"]].append(r["score"])
    never = {q for q, v in per_question.items() if sum(v) == 0}
    always = sum(1 for v in per_question.values() if sum(v) == len(v))
    print(
        f" questions: never solved {len(never)}/{len(per_question)}, "
        f"solved every attempt {always}/{len(per_question)}"
    )
    return never


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+")
    args = parser.parse_args()

    never_by_run = {}
    for dir_ in args.result_dirs:
        files = sorted(glob.glob(os.path.join(dir_, "UID*.s*.json"))) or sorted(
            glob.glob(os.path.join(dir_, "UID*.json"))
        )
        rows = [analyze_attempt(f) for f in files]
        never_by_run[dir_] = report(dir_, rows)

    if len(never_by_run) > 1:
        print("\n===== cross-run (questions never solved)")
        names = list(never_by_run)
        common = set.intersection(*never_by_run.values())
        print(f" unsolved in every run: {len(common)}")
        for a in names:
            for b in names:
                if a != b:
                    recovered = never_by_run[a] - never_by_run[b]
                    print(f" unsolved in {a} but solved at least once in {b}: {len(recovered)}")


if __name__ == "__main__":
    main()
