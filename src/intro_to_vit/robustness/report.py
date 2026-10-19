"""Save benchmark results as JSON and as a Markdown table."""

import json
from dataclasses import asdict
from pathlib import Path

from intro_to_vit.robustness.benchmark import BenchmarkResult
from intro_to_vit.robustness.perturbations import PERTURBATIONS


def result_to_dict(result: BenchmarkResult) -> dict:
    """Return the result as a JSON-friendly dict (per-image correctness is left out)."""
    data = asdict(result)
    data.pop("correct")
    return data


def format_p_value(p: float) -> str:
    """Format a p-value compactly, e.g. '<0.001' or '0.27'."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def format_markdown(result: BenchmarkResult) -> str:
    """Render accuracy per severity as a Markdown table, with the paired comparison if present."""
    perturbation = PERTURBATIONS[result.perturbation]
    names = [m["display_name"] for m in result.models]
    header = [perturbation.severity_label, *names]
    if result.paired:
        header += [f"bara {names[0]} rätt", f"bara {names[1]} rätt", "p (McNemar)"]
    lines = [
        f"{perturbation.title} på {result.dataset} ({result.num_images:,} bilder, seed {result.seed})",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row, severity in enumerate(result.severities):
        cells = [f"{severity:g}"] + [f"{result.accuracy[m['alias']][row]:.1%}" for m in result.models]
        if result.paired:
            pair = result.paired[row]
            cells += [str(pair["only_first"]), str(pair["only_second"]), format_p_value(pair["p_value"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_result(result: BenchmarkResult, out_dir: str | Path) -> Path:
    """Write results.json and results.md to `out_dir` and return the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result_to_dict(result), indent=2))
    (out / "results.md").write_text(format_markdown(result) + "\n")
    return out
