# Future Work

Ideas we wanted to pursue but cut, with the reason for each.

- **Investigator feedback loop and selection bias (gap G10).** Only alerted transactions get timely labels, so the training labels depend on the previous model. Simulating this needs a closed-loop simulator (model → alerts → labels → retrain), which is beyond the scope of v0.1. Related work: `dalpozzolo2018realistic`, `krutikov2024booking`, `sculley2015debt`.
- **Drift × label delay interaction.** RQ2 and RQ4 isolate each factor (DECISIONS D010). Production systems face both at once, and the combined effect could be worse than additive.
- **Calibrating generator distributions to published aggregate statistics.** For example, matching the amount and hour-of-day histograms reported in public papers. v0.1 uses stylised parameters (DATA_CARD limitation 1).
- **Seasonality and legitimate-behaviour drift** (weekly cycles, holidays, customer life events).
- **Adaptive adversary:** fraud whose mix responds to which scenarios the deployed model catches.
- **TabPFN arm of RQ1.** Implemented and wired into `experiments/configs/rq1.yaml`, but its weights need a license token the automated run didn't have (DECISIONS D019). Rerun with `uv sync --extra tabpfn && TABPFN_TOKEN=… make reproduce`.
- **Delay-aware validation for RQ4.** Under delayed labels, the most recent validation window has almost no matured labels, so model selection is close to blind. A matured-window or prequential validation scheme (`leborgne2022handbook`) would separate "fewer training labels" from "no validation signal".
- **Neural hyperparameter search.** The neural models use fixed, literature-default architectures. An Optuna budget equal to LightGBM's would make RQ1 fairer to FT-Transformer, at roughly 10× the compute.
- **LLM features via a local model (gap G8).** Cut for time. Ollama is installed, but a seeded, cached ≤4B-parameter feature extractor wouldn't fit the 90-minute budget alongside the core suite.
- **PyG-based GraphSAGE / GAT / CARE-GNN.** Blocked by missing arm64 wheels for `pyg-lib` (D019). The pure-PyTorch GraphSAGE covers the RQ3 question. Fraud-specific GNNs remain untested here.
