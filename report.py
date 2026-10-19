"""Count the verdicts in one or more results files and print them as tables.

Run after run.py. Each answer is judged fresh by tribunal/judge.py at report time, so the
tables always reflect the current heuristics, even for old runs.

    python report.py results/*.csv
    python report.py results/*.csv --markdown     # paste-ready tables
    python report.py results/<file>.csv --annotate  # write a copy with the verdicts filled in, for review
"""

import argparse
import collections
import csv
import io
import statistics
from pathlib import Path

from tribunal.judge import VERDICTS, verdict


def load(path):
    """Return (metadata dict, row dicts, raw header lines) for one results CSV."""
    with open(path, newline="") as f:
        lines = f.readlines()
    # The run metadata is the block of '# key: value' lines at the top. Stop at the first line
    # that isn't one, rather than dropping every '#' line, which would also eat any answer
    # that spans several lines and has one starting with '#' (a markdown heading, say).
    n = next(i for i, line in enumerate(lines) if not line.startswith("# "))
    meta = dict(line[2:].rstrip("\n").split(": ", 1) for line in lines[:n])
    rows = list(csv.DictReader(io.StringIO("".join(lines[n:]))))
    return meta, rows, lines[:n]


def judge_all(meta, rows):
    """Attach (verdict, detail) to every row. Files from before max_new_tokens was recorded skip truncation checks."""
    cap = meta.get("max_new_tokens")
    for row in rows:
        row["prelim_verdict"], row["verdict_detail"] = verdict(row, cap)


def table(headers, body, markdown):
    """Render rows as an aligned plain-text table, or as a markdown table."""
    cells = [headers] + [[str(c) for c in r] for r in body]
    if markdown:
        lines = ["| " + " | ".join(r) + " |" for r in cells]
        lines.insert(1, "|" + "---|" * len(headers))
        return "\n".join(lines)
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    # First column (labels) left-aligned, numbers right-aligned.
    fmt = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([fmt(cells[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]])


def model_name(meta):
    """The model's repo id when known, else whatever --model was (files from before model_repo was recorded)."""
    repo = meta.get("model_repo", "unknown")
    return meta["model"] if repo == "unknown" else repo


def describe(meta):
    """One line naming the run: model, prompt, and machine."""
    machine = meta.get("gpu") or meta.get("cpu", "?")
    return (f"{model_name(meta)} · {meta.get('format', '?')}-{meta['prompt_style']} · "
            f"{machine} ({meta['device']}, {meta['dtype']})")


def report_one(path, meta, rows, markdown):
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        counts[row["category"]][row["prelim_verdict"]] += 1
    body = [[cat] + [c[v] or "·" for v in VERDICTS] for cat, c in counts.items()]
    total = sum(counts.values(), collections.Counter())
    body.append(["all"] + [total[v] or "·" for v in VERDICTS])

    details = collections.Counter(r["verdict_detail"] for r in rows if r["verdict_detail"])
    detail_line = ", ".join(f"{n} {d}" for d, n in sorted(details.items())) or "none"
    if "max_new_tokens" not in meta:
        detail_line += " (max_new_tokens not recorded in this file, so truncation wasn't checked)"

    print(f"\n{'### ' if markdown else ''}{describe(meta)}")
    print(f"{path.name} · {meta['samples']} sample(s) per question · {float(meta['median_tok_s']):.1f} tok/s median · "
          f"{float(meta['total_seconds']) / 60:.1f} min\n")
    print(table(["category", *VERDICTS], body, markdown))
    print(f"\nSub-labels: {detail_line}")


def pct(n, total):
    # A --limit run can hold no fictional questions at all; show n/a rather than divide by zero.
    return f"{100 * n / total:.0f}%" if total else "n/a"


def summary_row(meta, rows):
    """Headline numbers for one run: how often it knew real answers, how often it caught fake questions."""
    real = [r for r in rows if r["kind"] == "real"]
    fake = [r for r in rows if r["kind"] == "fictional"]
    knows = sum(r["prelim_verdict"] == "knows" for r in real)
    caught = sum(r["prelim_verdict"] in ("says-unknown", "rejects-premise") for r in fake)
    toks = statistics.median(int(r["new_tokens"]) for r in rows)
    return [model_name(meta).split("/")[-1], f"{meta.get('format', '?')}-{meta['prompt_style']}",
            pct(knows, len(real)), pct(caught, len(fake)), f"{toks:.0f}",
            f"{float(meta['median_tok_s']):.1f}", f"{float(meta['total_seconds']) / 60:.1f}"]


def annotate(path, header_lines, rows):
    """Write results/judged/<name> with prelim_verdict and verdict_detail just before human_verdict."""
    out = path.parent / "judged" / path.name
    out.parent.mkdir(exist_ok=True)
    fields = [k for k in rows[0] if k not in ("prelim_verdict", "verdict_detail", "human_verdict")]
    fields += ["prelim_verdict", "verdict_detail", "human_verdict"]
    with open(out, "w", newline="") as f:
        f.writelines(header_lines)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="results CSVs written by run.py")
    p.add_argument("--markdown", action="store_true", help="print markdown tables")
    p.add_argument("--annotate", action="store_true", help="also write a copy of each file with verdicts filled in")
    args = p.parse_args()

    summaries = []
    for path in args.files:
        meta, rows, header_lines = load(path)
        judge_all(meta, rows)
        report_one(path, meta, rows, args.markdown)
        summaries.append(summary_row(meta, rows))
        if args.annotate:
            annotate(path, header_lines, rows)

    if len(summaries) > 1:
        print(f"\n{'### ' if args.markdown else ''}Summary\n")
        print(table(["model", "prompt", "knows (real)", "caught (fake)", "median tok", "tok/s", "min"],
                    summaries, args.markdown))


if __name__ == "__main__":
    main()
