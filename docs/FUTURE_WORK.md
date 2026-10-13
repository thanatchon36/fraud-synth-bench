# Future Work

Ideas we wanted to pursue but cut, with the reason for each.

- **Investigator feedback loop and selection bias (gap G10).** Only alerted transactions get timely labels, so the training labels depend on the previous model. Simulating this needs a closed-loop simulator (model → alerts → labels → retrain), which is beyond the scope of v0.1. Related work: `dalpozzolo2018realistic`, `krutikov2024booking`, `sculley2015debt`.
- **Drift × label delay interaction.** RQ2 and RQ4 isolate each factor (DECISIONS D010). Production systems face both at once, and the combined effect could be worse than additive.
