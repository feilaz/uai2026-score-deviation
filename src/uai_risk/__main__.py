import argparse
from pathlib import Path

from .experiments import run_panel, summarize, write_csv


def main():
    # The installed `uai-risk` command and `python -m uai_risk` enter here.
    # Defaults are relative to the shell's current directory, unlike the runner.
    parser = argparse.ArgumentParser(description="Offline UAI 2026 fact-selection experiments")
    parser.add_argument("panel", choices=("main", "gamma", "heterogeneous", "all"), nargs="?", default="main")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Numeric snapshot directory")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--include-legacy", action="store_true", help="Also run the earlier calibration rule, including its invalid fallback")
    parser.add_argument("--thresholds", type=int, default=200,
                        help="Candidate count: 200 (original experiment scripts), 100 (paper specification)")
    args = parser.parse_args()
    if args.thresholds < 2:
        parser.error("--thresholds must be at least 2")
    if not (args.data / "manifest.json").is_file():
        parser.error("Dataset not found. Run from the repository root or pass --data /path/to/data.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.include_legacy:
        print("Legacy mode runs the earlier calibration rule, including uncertified selections.")
    panels = ("main", "gamma", "heterogeneous") if args.panel == "all" else (args.panel,)
    for panel in panels:
        rows = run_panel(panel, args.data, args.include_legacy, args.thresholds)
        summary = summarize(rows)
        write_csv(args.output / f"{panel}_splits.csv", rows)
        write_csv(args.output / f"{panel}_summary.csv", summary)
        print(f"{panel}: {len(rows)} split results, {len(summary)} summary rows → {args.output}")


if __name__ == "__main__":
    main()
