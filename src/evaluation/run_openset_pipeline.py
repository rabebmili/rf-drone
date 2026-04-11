"""Orchestrator for the full open-set evaluation pipeline.

Runs the four stages in order:
  1. run_openmax_only     — compute OpenMax for all models × datasets × holdouts,
                            fill any known MSP/Energy/Mahalanobis gaps
  2. generate_openmax_plots — distribution PNGs for selected thesis cases
  3. summarize_openset_with_openmax — print AUROC tables to stdout
  4. plot_openset_thesis_figures    — generate thesis chapter figures (fig1/fig2/fig3)

Usage:
    python -m src.evaluation.run_openset_pipeline
"""

from src.evaluation.run_openmax_only import main as run_openmax
from src.evaluation.generate_openmax_plots import main as generate_plots
from src.evaluation.summarize_openset_with_openmax import main as summarize
from src.evaluation.plot_openset_thesis_figures import main as plot_thesis_figures


def main():
    print("\n" + "=" * 70)
    print("  OPEN-SET EVALUATION PIPELINE  (step 1/4: OpenMax + gap fill)")
    print("=" * 70)
    run_openmax()

    print("\n" + "=" * 70)
    print("  OPEN-SET EVALUATION PIPELINE  (step 2/4: distribution plots)")
    print("=" * 70)
    generate_plots()

    print("\n" + "=" * 70)
    print("  OPEN-SET EVALUATION PIPELINE  (step 3/4: AUROC summary tables)")
    print("=" * 70)
    summarize()

    print("\n" + "=" * 70)
    print("  OPEN-SET EVALUATION PIPELINE  (step 4/4: thesis figures)")
    print("=" * 70)
    plot_thesis_figures()

    print("\n" + "=" * 70)
    print("  Open-set pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
