# Score Deviation for Risk-Controlled Fact Selection

Code for [Controlling Uncertainty and Hallucination Risk in Multi-Agent Fact Verification](https://proceedings.mlr.press/v337/kostka26a.html) (UAI 2026).

## Quickstart

From the repository root, with Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uai-risk main
```

Saved-data experiments run on a CPU without API keys or model downloads.

| Command | Experiment |
|---|---|
| `uai-risk main` | Compare raw-score, entropy, U-Shape and Score Deviation scoring |
| `uai-risk gamma` | Sweep the Score Deviation penalty weight |
| `uai-risk heterogeneous` | Compare heterogeneous and matched homogeneous ensembles |
| `uai-risk all` | Run all three experiments |

Included numerical data are in [`data/`](data/). Commands write split-level and summary CSVs to `outputs/`. Use `--data PATH` or `--output PATH` to change these locations. The released snapshot contains 299 completed records from the original input set.

Corrected calibration is the default. `--include-legacy` additionally runs the historical rule, including its uncertified fallback selections.

## Generate model outputs

[`run_experiment.py`](run_experiment.py) generates five answers per query, extracts claims, obtains weak-verifier scores and strong-judge labels, and computes response entropy with MiniLM embeddings.

```sh
python -m pip install -e '.[generation]'
cp .env.example .env
# Set your provider and API key in .env, then run:
python run_experiment.py generate --limit 1
```

Configure Gemini or OpenRouter, models and the estimated spend limit in `.env`, using [`.env.example`](.env.example) as a template. Inputs come from `data/inputs.json.gz`. Cached calls, annotated records and numerical exports are saved under `outputs/generation/`. Increase `--limit` to process more queries. Generation supports the homogeneous ensemble.

## License and data

Code is licensed under [MIT](LICENSE). The input dataset is adapted from Google's FACTS Grounding public examples under CC BY 4.0; see [data attribution](data/README.md).

## Citation

```bibtex
@InProceedings{pmlr-v337-kostka26a,
  title     = {Controlling Uncertainty and Hallucination Risk in Multi-Agent Fact Verification},
  author    = {Kostka, Adam and Chudziak, Jaroslaw A.},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages     = {3143--3161},
  year      = {2026},
  volume    = {337},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v337/kostka26a.html}
}
```
