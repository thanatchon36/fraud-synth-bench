# Literature Review: Machine Learning for Transaction Fraud Detection, 2015–2026

**Scope.** This review covers transaction and payment fraud plus closely related financial-crime tasks (anti-money laundering, account-opening fraud, review/spam fraud used as GNN benchmarks). It runs from 2015 to the present, with a few earlier foundational papers. Every reference was checked against OpenAlex (title, authors, year, venue, DOI/arXiv ID) on 2026-10-13. Keys refer to [`refs.bib`](refs.bib).

**How to read an entry.** *Summary*: what the paper does. *Data*: what it was evaluated on. *Metrics*: what it reports. *Claim*: the headline finding, as the authors state it. *Gap*: what it leaves open that matters for this project. Where we couldn't confirm an exact number from the paper, we describe the result qualitatively instead of quoting a figure.

**A note on evidence.** Most of the real-world fraud literature uses proprietary data (bank partners, eBay, Alipay, Booking.com) or one public dataset (the ULB/Kaggle European card dataset, 284,807 transactions, 492 frauds, PCA-anonymised features). That makes cross-paper comparisons weak and is the root of several gaps listed below.

---

## 1. Classical supervised learning on tabular transactions

### 1.1 Dal Pozzolo et al. (2014). *Learned lessons in credit card fraud detection from a practitioner perspective.* ESWA. `dalpozzolo2014lessons`
- **Summary:** A large industrial study with a card processor. It compares learners, resampling strategies and retraining schedules under realistic constraints (alerts are capped, investigators are limited).
- **Data:** Millions of real European card transactions (proprietary).
- **Metrics:** AUC, average precision, and alert precision (precision of the top-k alerts).
- **Claim:** Random forests with undersampling and frequent retraining work well. Precision on the top-k alerts is the operationally relevant metric, more than AUC.
- **Gap:** Only tree ensembles and classical learners. No study of how the ranking of model families changes under drift or label delay. The data isn't public.

### 1.2 Dal Pozzolo et al. (2015). *Calibrating probability with undersampling for unbalanced classification.* IEEE SSCI. `dalpozzolo2015calibrating`
- **Summary:** Shows analytically that undersampling the majority class distorts posterior probabilities, and gives a closed-form correction.
- **Data:** Several imbalanced benchmarks, plus the card dataset that later became the public ULB/Kaggle dataset.
- **Metrics:** AUC, Brier score, log-loss.
- **Claim:** Undersampling improves ranking but miscalibrates probabilities. A simple correction restores calibration without hurting ranking.
- **Gap:** Calibration is rarely reported in later fraud papers, even though cost-based decision rules depend on it. Calibration (ECE) of modern models under extreme imbalance hasn't been tested.

### 1.3 Correa Bahnsen et al. (2016). *Feature engineering strategies for credit card fraud detection.* ESWA. `bahnsen2016feature`
- **Summary:** Proposes transaction-aggregation features (counts and amounts per customer, merchant group and time window) and periodic time features (a von Mises model of each customer's usual transaction hour).
- **Data:** Proprietary European card transactions.
- **Metrics:** Cost-based "savings" plus F1.
- **Claim:** Aggregation and periodic features substantially raise savings across learners. Feature engineering matters more than the choice of algorithm.
- **Gap:** The hand-crafted features are per-customer only, with no relational or graph features. It's unclear whether learned representations (sequence or graph models) can replace them.

### 1.4 Correa Bahnsen et al. (2015). *Example-dependent cost-sensitive decision trees.* ESWA. `bahnsen2015costtrees`
- **Summary:** Decision trees whose splitting and pruning criteria use example-specific misclassification costs (for example, the transaction amount).
- **Data:** Card fraud, credit scoring and direct marketing (proprietary or public).
- **Metrics:** Savings relative to a cost-insensitive baseline.
- **Claim:** Example-dependent cost-sensitive trees achieve higher savings than cost-insensitive models with post-hoc thresholding.
- **Gap:** Costs are per-example but investigation capacity isn't modelled. There's no comparison with modern GBDTs or neural models under a capacity-limited alert budget.

### 1.5 Dal Pozzolo et al. (2018). *Credit card fraud detection: a realistic modeling and a novel learning strategy.* IEEE TNNLS. `dalpozzolo2018realistic`
- **Summary:** Formalises a realistic fraud-detection system: investigators verify only a daily top-k of alerts (feedback), while all other labels arrive with a delay of days to weeks. It proposes training separate classifiers on feedback and on delayed samples and then aggregating them.
- **Data:** About 75M real European card transactions (proprietary).
- **Metrics:** Alert precision (Pk), AUC.
- **Claim:** Handling feedback and delayed labels separately beats pooling them. AUC is misleading compared with top-k precision.
- **Gap:** Shows that verification latency is central but evaluates only random forests. How label delay hurts *different model families* (GBDT, sequence, graph, foundation models) is untested.

### 1.6 Carcillo et al. (2018). *SCARFF: a scalable framework for streaming credit card fraud detection with Spark.* Information Fusion. `carcillo2018scarff`
- **Summary:** A streaming architecture (Kafka, Spark, Cassandra) that applies the feedback-and-delay learning strategy at scale.
- **Data:** Real card stream (proprietary).
- **Metrics:** Alert precision, throughput.
- **Claim:** The delayed-feedback strategy can run in real time at production volumes.
- **Gap:** Engineering-focused. Model families aren't compared, and the latency of inference per 1k predictions isn't reported for modern models.

### 1.7 Shwartz-Ziv & Armon (2022). *Tabular data: Deep learning is not all you need.* Information Fusion. `shwartzziv2022tabular`
- **Summary:** Re-evaluates recent deep tabular models (TabNet, NODE, DNF-Net, 1D-CNN) against XGBoost on datasets from the original papers and new ones.
- **Data:** 11 public tabular datasets (not fraud-specific).
- **Metrics:** Accuracy, log-loss, tuning time.
- **Claim:** XGBoost generally beats the deep models, especially on datasets outside each model's original paper, and needs far less tuning. An ensemble of both is best.
- **Gap:** No extreme imbalance (<1% positives), no temporal splits, no fraud data.

### 1.8 Grinsztajn, Oyallon & Varoquaux (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS Datasets & Benchmarks. `grinsztajn2022trees`
- **Summary:** A large controlled benchmark with equal random-search budgets. It analyses *why* trees win: robustness to uninformative features, preserving data orientation, and fitting irregular functions.
- **Data:** 45 medium-sized public datasets (about 10k training rows each).
- **Metrics:** Accuracy and R² normalised across datasets, as a function of tuning budget.
- **Claim:** Tree ensembles stay state of the art on medium-sized tabular data even with heavy tuning of neural nets.
- **Gap:** The benchmark deliberately excludes heavily imbalanced and temporal tasks. Fraud is both.

---

## 2. Anomaly detection and unsupervised or semi-supervised methods

### 2.1 Pang, Shen, Cao & van den Hengel (2021). *Deep learning for anomaly detection: a review.* ACM Computing Surveys. `pang2021deepad`
- **Summary:** A taxonomy of deep anomaly detection (feature extraction, normality learning, end-to-end scoring), including autoencoders, GAN-based detectors and one-class objectives. It covers fraud as an application area.
- **Data:** Survey.
- **Metrics:** —
- **Claim:** Deep AD shines with complex data types. Using a few labelled anomalies (weak supervision) is a key open direction.
- **Gap:** Doesn't benchmark against supervised GBDT when labels *do* exist, but late. That's exactly the label-delay regime.

### 2.2 Pang, Shen & van den Hengel (2019). *Deep anomaly detection with deviation networks.* KDD. `pang2019devnet`
- **Summary:** DevNet uses a handful of labelled anomalies plus unlabelled data to learn anomaly scores directly, with a Gaussian-prior deviation loss.
- **Data:** Public AD benchmarks, including the ULB credit card dataset.
- **Metrics:** AUROC, AUPRC.
- **Claim:** A few labelled anomalies substantially improve over unsupervised detectors.
- **Gap:** Static evaluation. The "few labels" come from an i.i.d. sample rather than from the delayed, investigator-biased labelling a real system produces.

### 2.3 Han et al. (2022). *ADBench: Anomaly Detection Benchmark.* NeurIPS Datasets & Benchmarks. `han2022adbench`
- **Summary:** Benchmarks 30 anomaly-detection algorithms on 57 datasets, varying the level of supervision, the anomaly type (local, global, dependency, clustered) and noise.
- **Data:** 57 public datasets, including fraud-like ones.
- **Metrics:** AUROC, AUPRC.
- **Claim:** No unsupervised method is consistently best. Semi-supervised methods with even about 1% labelled anomalies beat the best unsupervised ones. Performance depends strongly on the anomaly type.
- **Gap:** Uses random splits with no time ordering. The anomaly types are generated generically rather than by fraud mechanism (card testing, account takeover, mules).

### 2.4 Carcillo et al. (2019/2021). *Combining unsupervised and supervised learning in credit card fraud detection.* Information Sciences. `carcillo2021combining`
- **Summary:** Adds outlier scores computed at several granularities (global, per-card, per-cluster) as features for a supervised random forest.
- **Data:** Real card transactions (proprietary).
- **Metrics:** Top-k precision, AUPRC.
- **Claim:** Unsupervised outlier-score features give modest but consistent gains over a supervised-only model.
- **Gap:** Labels are assumed available at training time. Whether unsupervised pretraining helps *more when labels are delayed* hasn't been measured.

### 2.5 Karnavou et al. (2025). *I know you're a fraud: Uncovering illicit activity in a Greek bank transactions with unsupervised learning.* ESWA. `karnavou2025greek`
- **Summary:** Compares an autoencoder with per-cardholder Isolation Forest and One-Class SVM on real bank data. Adds genetic-algorithm feature selection and SHAP explanations.
- **Data:** Confidential Greek bank transactions.
- **Metrics:** Precision, recall, F1 against expert-validated cases.
- **Claim:** The autoencoder detected illicit activity best among the unsupervised options.
- **Gap:** No supervised baseline, so the cost of going label-free isn't quantified. The data isn't public.

---

## 3. Sequence models over account histories

### 3.1 Jurgovsky et al. (2018). *Sequence classification for credit-card fraud detection.* ESWA. `jurgovsky2018sequence`
- **Summary:** Frames fraud detection as sequence classification and trains an LSTM on each cardholder's transaction history. Compares it with a random forest given aggregation features.
- **Data:** Real European card transactions (proprietary), split into card-present and card-not-present.
- **Metrics:** AUPRC.
- **Claim:** The LSTM helps on card-present (offline) transactions. With good aggregation features, the random forest is competitive. The two models catch partly different frauds, so combining them helps.
- **Gap:** Uses one temporal test window and no drift analysis. No graph or relational signal.

### 3.2 Branco et al. (2020). *Interleaved Sequence RNNs for Fraud Detection.* KDD. `branco2020interleaved`
- **Summary:** Feature-engineering-free GRU models over interleaved per-card sequences, plus a production pipeline (state stores, millisecond latency).
- **Data:** Real payment data from Feedzai clients (proprietary).
- **Metrics:** Recall at a fixed false-positive rate, money saved, latency.
- **Claim:** Sequence RNNs outperform the production tree-based system that uses engineered features, while using fewer resources.
- **Gap:** The baseline's feature set is proprietary, so "RNN beats trees" can't be reproduced. No controlled study of *which fraud scenarios* benefit from sequence modelling.

### 3.3 Padhi et al. (2021). *Tabular Transformers for Modeling Multivariate Time Series.* ICASSP. `padhi2021tabformer`
- **Summary:** Hierarchical BERT-style (TabBERT) and GPT-style transformers for tabular time series. Releases a *synthetic* credit-card transaction dataset (about 24M transactions) as a fraud benchmark.
- **Data:** Synthetic card transactions (IBM TabFormer) and a pollution dataset.
- **Metrics:** F1 for fraud, fidelity for generation.
- **Claim:** Pretrained tabular-transformer representations support fraud classification and realistic synthetic sequence generation.
- **Gap:** The synthetic generator isn't parameterised by fraud scenario or drift. Baselines are thin, with no tuned GBDT plus engineered sequence features.

### 3.4 Cheng et al. (2020). *Spatio-Temporal Attention-Based Neural Network for Credit Card Fraud Detection.* AAAI. `cheng2020stan`
- **Summary:** STAN uses attention over a cardholder's history along time and location axes, then a 3D convolution, to score a new transaction.
- **Data:** Real card transactions from a Chinese bank (proprietary).
- **Metrics:** AUC, precision, recall.
- **Claim:** Outperforms classical learners and plain LSTMs.
- **Gap:** Proprietary data, no tuned GBDT with velocity features, no drift or label-delay study.

### 3.5 Aminian et al. (2025). *FraudTransformer: Time-Aware GPT for Transaction Fraud Detection.* arXiv. `aminian2025fraudtransformer`
- **Summary:** A GPT-style sequence model with a dedicated time encoder (absolute timestamps or inter-event gaps) plus learned positional encodings.
- **Data:** Large industrial banking stream with tens of millions of events (proprietary).
- **Metrics:** AUROC, AUPRC.
- **Claim:** Beats logistic regression, XGBoost and LightGBM, as well as ablations without the time or position components.
- **Gap:** The data isn't reproducible. There's no compute or latency comparison, and nothing on robustness to drift.

---

## 4. Graph-based fraud detection

### 4.1 Pourhabibi et al. (2020). *Fraud detection: A systematic literature review of graph-based anomaly detection approaches.* Decision Support Systems. `pourhabibi2020graphslr`
- **Summary:** A systematic review of graph-based anomaly detection for fraud (2007–2018), covering structural features, community methods, probabilistic and embedding-based approaches.
- **Data:** Survey.
- **Metrics:** —
- **Claim:** Graph-based approaches are promising but evaluation is fragmented and dominated by proprietary data.
- **Gap:** Calls for benchmark datasets that have known ground-truth fraud structure.

### 4.2 Wang et al. (2019). *A Semi-supervised Graph Attentive Network for Financial Fraud Detection (SemiGNN).* ICDM. `wang2019semignn`
- **Summary:** Hierarchical attention over multi-view user graphs (relations plus attributes). It's semi-supervised to exploit unlabelled users.
- **Data:** Alipay users (proprietary).
- **Metrics:** AUC, KS.
- **Claim:** Outperforms feature-based and earlier graph baselines, and gives interpretable attention.
- **Gap:** Baselines lack hand-crafted graph features in a GBDT. The data isn't reproducible.

### 4.3 Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD workshop / arXiv. `weber2019elliptic`
- **Summary:** Releases the Elliptic Bitcoin transaction graph (203,769 transaction nodes, 234,355 payment-flow edges, 166 features; only about 2% of all nodes are labelled illicit, and most are unlabelled). Compares logistic regression, random forests, MLPs and GCN, including a temporal split.
- **Data:** Elliptic dataset (public).
- **Metrics:** Illicit-class precision, recall, F1.
- **Claim:** A **random forest on node features outperforms GCN**. Adding GCN embeddings as random-forest features helps a little. Every model collapses after a "dark market shutdown" event (drift), and *even a random forest retrained at every time step* doesn't reliably recover.
- **Gap:** Early evidence that tabular plus graph features can match GNNs, but on one dataset with unexplained features. The drift event wasn't designed, so its mechanism is unknown.

### 4.4 Dou et al. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters (CARE-GNN).* CIKM. `dou2020caregnn`
- **Summary:** Fraudsters camouflage their features and relations. CARE-GNN uses similarity-aware neighbour selection trained with reinforcement learning, plus relation-aware aggregation.
- **Data:** YelpChi and Amazon review-fraud graphs (public).
- **Metrics:** AUC, recall.
- **Claim:** Beats GCN, GAT, GraphSAGE and earlier fraud GNNs on both datasets.
- **Gap:** Review spam isn't payment fraud. It wasn't compared with a GBDT on graph-aggregated features, and the data has no time ordering.

### 4.5 Liu et al. (2021). *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection (PC-GNN).* WWW. `liu2021pcgnn`
- **Summary:** Label-balanced sampling of nodes to build subgraphs, plus neighbourhood sampling that over-samples same-class neighbours, to handle class imbalance in GNNs.
- **Data:** YelpChi and Amazon, plus proprietary data.
- **Metrics:** AUC, G-mean.
- **Claim:** Outperforms CARE-GNN and others under imbalance.
- **Gap:** The class imbalance tested (around 5–15%) is far milder than in payments (<0.5%). Static evaluation only.

### 4.6 Rao et al. (2022). *xFraud: Explainable Fraud Transaction Detection.* PVLDB. `rao2022xfraud`
- **Summary:** A heterogeneous GNN on a transaction–user–device–address graph at eBay, with a hybrid explainer (GNNExplainer plus centrality) validated by human experts.
- **Data:** eBay transaction graphs (proprietary, up to about 1B nodes).
- **Metrics:** AUC, explainer agreement with human annotators.
- **Claim:** The heterogeneous GNN beats baselines, and explanations agree with investigators.
- **Gap:** Proprietary data. The baselines don't include a strong GBDT with shared-entity features.

### 4.7 Lu et al. (2022). *BRIGHT – Graph Neural Networks in Real-time Fraud Detection.* CIKM. `lu2022bright`
- **Summary:** A two-stage directed graph and a Lambda neural network that separate batch and real-time inference, so GNNs respect causality (no future edges) and meet latency limits.
- **Data:** eBay (proprietary).
- **Metrics:** Average precision, p99 latency.
- **Claim:** Large speedups in inference latency at comparable or better precision.
- **Gap:** Highlights that many GNN fraud papers leak future edges. Most public GNN benchmarks ignore this, and so do most tabular baselines.

### 4.8 Tang et al. (2022). *Rethinking Graph Neural Networks for Anomaly Detection (BWGNN).* ICML. `tang2022bwgnn`
- **Summary:** Anomalies shift spectral energy to high frequencies ("right shift"). BWGNN uses band-pass Beta-wavelet filters to capture this.
- **Data:** YelpChi, Amazon, T-Finance, T-Social.
- **Metrics:** AUROC, F1-macro.
- **Claim:** State-of-the-art graph anomaly detection with a simple spectral design.
- **Gap:** Static graphs, random splits.

### 4.9 Tang et al. (2023). *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection.* NeurIPS Datasets & Benchmarks. `tang2023gadbench`
- **Summary:** Benchmarks 29 models on 10 graph anomaly detection datasets (up to about 6M nodes) under a uniform protocol.
- **Data:** 10 public datasets (including YelpChi, Amazon, T-Finance, Elliptic, DGraph).
- **Metrics:** AUROC, AUPRC, Rec@K.
- **Claim:** **Tree ensembles with simple neighbourhood aggregation can outperform the latest GNNs** built for graph anomaly detection.
- **Gap:** The most direct prior evidence for our RQ3. But the datasets have unknown fraud mechanisms, so the benchmark can't say *when* (for which fraud structures) GNNs should win.

### 4.10 Altman et al. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS Datasets & Benchmarks. `altman2023amlworld`
- **Summary:** An agent-based generator (AMLworld) calibrated to real transactions. It produces datasets with laundering typologies (fan-in, fan-out, cycles, scatter-gather) and *complete* ground truth. Benchmarks GNNs and GBDTs.
- **Data:** Synthetic AML datasets (public).
- **Metrics:** Minority-class F1.
- **Claim:** Synthetic data with complete labels can be *better* than real data for comparing models, because real laundering labels are incomplete.
- **Gap:** Only laundering. No card-fraud scenarios, no designed drift schedule, no label-delay model. Ships as fixed datasets rather than a library with pluggable scenario injectors.

---

## 5. Foundation-model era: tabular transformers, TabPFN, LLMs

### 5.1 Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data (FT-Transformer).* NeurIPS. `gorishniy2021ftt`
- **Summary:** Proposes strong, simple baselines: a ResNet-like MLP and FT-Transformer (feature tokenizer plus transformer). Evaluated carefully against GBDTs.
- **Data:** 11 public tabular datasets.
- **Metrics:** Accuracy and RMSE with tuned models and multiple seeds.
- **Claim:** FT-Transformer is the strongest deep tabular model across tasks, but **there's still no universal winner between GBDT and deep models**.
- **Gap:** No highly imbalanced or temporal tasks.

### 5.2 Somepalli et al. (2021). *SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training.* arXiv. `somepalli2021saint`
- **Summary:** Attention over both columns and rows (intersample attention), plus contrastive self-supervised pretraining.
- **Data:** 16 public datasets, including the ULB credit card dataset.
- **Metrics:** AUROC, accuracy.
- **Claim:** Outperforms GBDTs on average across the benchmark. Pretraining helps in low-label regimes.
- **Gap:** Random splits. The credit card result is on the PCA-anonymised ULB data with near-saturated AUROC. No temporal or drift evaluation.

### 5.3 Hollmann et al. (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* ICLR. `hollmann2023tabpfn`
- **Summary:** A transformer meta-trained on synthetic datasets drawn from a prior over structural causal models. It performs in-context classification with no gradient updates.
- **Data:** OpenML-CC18 small datasets (≤1,000 rows, ≤100 features).
- **Metrics:** ROC-AUC, runtime.
- **Claim:** Matches tuned GBDTs and AutoML on small datasets in about 1 second.
- **Gap:** Hard limits on rows and features. Imbalanced, temporal fraud wasn't tested.

### 5.4 Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model (TabPFN v2).* Nature. `hollmann2025tabpfnv2`
- **Summary:** Scales TabPFN to datasets of up to 10,000 samples and 500 features, with support for categorical and missing values, regression, and fine-tuning or embeddings.
- **Data:** Public benchmarks (AutoML Benchmark, OpenML-CTR23).
- **Metrics:** Normalised ROC-AUC and accuracy, plus runtime.
- **Claim:** Outperforms tuned GBDT ensembles on datasets up to 10k samples with far less compute.
- **Gap:** Fraud training sets are much larger than 10k and very imbalanced. Subsampling to 10k rows keeps very few positives. **Whether TabPFN's small-data advantage survives in this regime is untested**, which is central to our RQ1.

### 5.5 Krutikov et al. (2024). *Challenging Gradient Boosted Decision Trees with Tabular Transformers for Fraud Detection at Booking.com.* arXiv. `krutikov2024booking`
- **Summary:** Self-supervised pretraining of tabular transformers on all production traffic, then fine-tuning on a small, unbiased control group. This addresses the *selection bias* introduced by the production fraud system.
- **Data:** Booking.com e-commerce fraud (proprietary).
- **Metrics:** Average precision (AUPRC).
- **Claim:** Pretrained tabular transformers outperform heavily tuned GBDTs by a considerable margin in average precision.
- **Gap:** The advantage depends on large-scale pretraining and a selection-biased labelling regime, so it's unclear whether it holds without pretraining or at smaller scale. Not reproducible.

### 5.6 Hegselmann et al. (2023). *TabLLM: Few-shot Classification of Tabular Data with Large Language Models.* AISTATS. `hegselmann2023tabllm`
- **Summary:** Serialises table rows into natural language and fine-tunes an LLM (T0) for few-shot classification.
- **Data:** 9 public tabular datasets.
- **Metrics:** ROC-AUC across numbers of shots.
- **Claim:** Strong in the very-few-shot regime. GBDTs catch up as labelled data grows.
- **Gap:** With fraud volumes (thousands of labelled rows), GBDT is expected to dominate. LLMs' role is more plausibly in feature engineering than in direct classification.

### 5.7 Hollmann, Müller & Hutter (2023). *Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering.* NeurIPS. `hollmann2023caafe`
- **Summary:** An LLM iteratively proposes Python feature-engineering code from dataset descriptions. Each feature is kept only if validation performance improves.
- **Data:** 14 public datasets.
- **Metrics:** ROC-AUC improvement.
- **Claim:** Improves downstream performance on most datasets, with interpretable features.
- **Gap:** Run with large hosted LLMs, not small local models. Not evaluated on fraud or under temporal validation, where features can leak.

### 5.8 Tan, Ma & Zhang (2026). *Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection (FinFRE-RAG).* ACL. `tan2026finfre`
- **Summary:** Direct LLM prompting on tabular fraud data does poorly. FinFRE-RAG reduces features by importance, serialises them, and uses retrieval-augmented in-context examples.
- **Data:** Four public fraud datasets.
- **Metrics:** F1, MCC.
- **Claim:** Big improvement over direct prompting, but **LLMs still lag specialised classifiers**.
- **Gap:** Confirms that LLM-as-classifier isn't competitive. The cost per 1k predictions and the value of LLMs for explanation or triage are open.

---

## 6. Operational issues: drift, label delay, imbalance, costs, feedback

### 6.1 Gama et al. (2014). *A survey on concept drift adaptation.* ACM CSUR. `gama2014drift`
- **Summary:** A taxonomy of drift (sudden, gradual, incremental, recurring) and adaptation strategies (windowing, forgetting, detectors, ensembles).
- **Data:** Survey. **Metrics:** —
- **Claim:** Evaluating drift needs prequential or time-ordered protocols, and the right adaptation depends on the drift type.
- **Gap:** General-purpose. Says nothing on how fraud-specific adversarial drift affects different model families.

### 6.2 Dal Pozzolo et al. (2015). *Credit card fraud detection and concept-drift adaptation with delayed supervised information.* IJCNN. `dalpozzolo2015drift`
- **Summary:** An early version of the feedback-and-delay setting (see 1.5). Compares sliding-window and ensemble approaches under drift.
- **Data:** Real card transactions (proprietary).
- **Metrics:** Alert precision.
- **Claim:** Treating recent feedback separately from delayed labels improves adaptation to drift.
- **Gap:** The drift wasn't designed, so it can't be tied to a specific change in fraudster behaviour.

### 6.3 Lebichot et al. (2019). *Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection.* INNS Big Data. `lebichot2019da`
- **Summary:** Domain adaptation (for example, adversarial feature alignment) to transfer a fraud model between e-commerce and face-to-face transaction domains.
- **Data:** Real Worldline transactions (proprietary).
- **Metrics:** Top-k precision, AUPRC.
- **Claim:** Domain adaptation partly closes the gap when target labels are scarce.
- **Gap:** The shift is between domains rather than over time. Doesn't model a fraudster pattern changing within one stream.

### 6.4 Lebichot et al. (2021). *Transfer Learning Strategies for Credit Card Fraud Detection.* IEEE Access. `lebichot2021transfer`
- **Summary:** A systematic comparison of 15 transfer-learning strategies between card-present and card-not-present domains.
- **Data:** Real card transactions (proprietary).
- **Metrics:** AUPRC, top-k precision.
- **Claim:** Simple strategies are competitive. Gains are context-dependent.
- **Gap:** Same as 6.3.

### 6.5 Gomes et al. (2022). *A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams.* ACM CSUR. `gomes2022delayed`
- **Summary:** A taxonomy of delayed and partially labelled stream learning, covering how verification latency differs from missing labels, and the semi-supervised and active-learning remedies.
- **Data:** Survey. **Metrics:** —
- **Claim:** Delayed labels change both how you train and how you evaluate. Evaluations rarely model realistic delay distributions.
- **Gap:** Fraud-specific empirical comparisons under a parameterised delay distribution are missing.

### 6.6 He & Garcia (2009). *Learning from Imbalanced Data.* IEEE TKDE. `he2009imbalanced`
- **Summary:** A foundational survey of sampling, cost-sensitive and kernel methods for imbalance, and of assessment metrics (it favours PR curves under heavy skew).
- **Data:** Survey. **Metrics:** —
- **Claim:** Accuracy and even ROC can mislead under heavy imbalance. Use precision–recall analysis.
- **Gap:** Predates GBDT dominance and modern deep and foundation models.

### 6.7 Elkan (2001). *The foundations of cost-sensitive learning.* IJCAI. `elkan2001cost`
- **Summary:** Shows that with a cost matrix, the optimal decision is a probability threshold. It derives how to rebalance training data to match costs.
- **Data:** Theory. **Metrics:** —
- **Claim:** Cost-sensitive decisions need *calibrated* probabilities, after which thresholding is optimal.
- **Gap:** Assumes unlimited review capacity. Real fraud operations have a fixed daily alert budget (top-k), which changes the decision rule.

### 6.8 Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. `sculley2015debt`
- **Summary:** Catalogues ML system risks, including *hidden feedback loops*, where a model's decisions influence its future training data.
- **Data:** Position paper. **Metrics:** —
- **Claim:** Feedback loops and entanglement are major sources of debt in production ML.
- **Gap:** In fraud, only alerted transactions get timely labels (investigator feedback). This selection effect is rarely simulated in benchmarks, although Krutikov et al. (5.5) address it with a control group.

---

## 7. Synthetic fraud data generation

### 7.1 Lopez-Rojas, Elmir & Axelsson (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS. `lopezrojas2016paysim`
- **Summary:** An agent-based simulator of mobile-money transactions calibrated on aggregate statistics from a real African mobile-money service, with injected fraud (account takeover followed by cash-out).
- **Data:** Synthetic, calibrated from proprietary logs.
- **Metrics:** Statistical similarity to the source.
- **Claim:** Agent-based simulation can produce privacy-preserving data that's realistic enough for fraud research.
- **Gap:** Essentially one fraud scenario. No graph of devices or mule rings, no drift schedule, no label delay. The released dataset is widely used, but its fraud labels are trivially separable from a few balance fields.

### 7.2 Xu et al. (2019). *Modeling Tabular Data using Conditional GAN (CTGAN / TVAE).* NeurIPS. `xu2019ctgan`
- **Summary:** A conditional GAN with mode-specific normalisation and a training-by-sampling scheme, plus a VAE baseline (TVAE), for mixed-type tabular data.
- **Data:** Simulated and real public tabular datasets.
- **Metrics:** Machine-learning efficacy (train on synthetic, test on real) and likelihood fitness.
- **Claim:** CTGAN beats earlier deep generators on mixed-type tables. TVAE is competitive.
- **Gap:** Learns from real data, so it needs data we're not allowed to use. It models rows i.i.d., which loses sequences, relational structure and controllable fraud mechanisms.

### 7.3 Jesus et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation (BAF).* NeurIPS Datasets & Benchmarks. `jesus2022baf`
- **Summary:** Bank Account Fraud: six 1M-row datasets generated with a CTGAN-family model trained on real account-opening data. Includes temporal dynamics, about 1% fraud, and controlled bias variants for fairness testing.
- **Data:** Synthetic, derived from proprietary Feedzai data.
- **Metrics:** Recall at 5% FPR, plus fairness metrics.
- **Claim:** The first large-scale, privacy-preserving, realistic tabular fraud suite with temporal shift and bias variants. LightGBM is a strong baseline.
- **Gap:** Account *opening* (one row per application) rather than transaction streams. No graph or sequence structure, no label delay, and the drift isn't controllable.

### 7.4 Le Borgne et al. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook.* Université Libre de Bruxelles (open online book). `leborgne2022handbook`
- **Summary:** An open handbook with a simple transaction simulator (customers, terminals, three fraud scenarios), a realistic prequential validation protocol with a delay period, and baselines.
- **Data:** Synthetic, from its own simulator.
- **Metrics:** AUC-ROC, average precision, card precision@k.
- **Claim:** Shows the evaluation pitfalls of random splits and recommends a delay-aware, time-ordered protocol with card precision@k.
- **Gap:** Only three simple scenarios. No graph or ring fraud, no adaptive drift schedule, and no foundation-model or GNN comparisons.

---

## 8. Foundational method references (not reviewed in detail)

| Key | Method | Why it's cited |
|---|---|---|
| `chen2016xgboost` | XGBoost (KDD 2016) | Reference GBDT |
| `ke2017lightgbm` | LightGBM (NeurIPS 2017) | Primary GBDT in this project (lighter on ARM) |
| `prokhorenkova2018catboost` | CatBoost (NeurIPS 2018) | GBDT with ordered boosting for categoricals |
| `liu2008iforest` | Isolation Forest (ICDM 2008) | Unsupervised baseline |
| `scholkopf2001ocsvm` | One-Class SVM (Neural Comp. 2001) | Unsupervised baseline |
| `kingma2014vae` | VAE (ICLR 2014) | Generative anomaly scoring |
| `hamilton2017graphsage` | GraphSAGE (NeurIPS 2017) | Inductive GNN with neighbour sampling, our GNN |
| `velickovic2018gat` | GAT (ICLR 2018) | Attention-based GNN baseline in the literature |

---

## 9. Synthesis

1. **Trees are hard to beat on tabular data, and fraud is mostly tabular.** That's the consistent finding of independent benchmarks (`grinsztajn2022trees`, `shwartzziv2022tabular`, `gorishniy2021ftt`), even if no single fraud paper settles it. Claims that a deep model beats GBDT on fraud (`branco2020interleaved`, `krutikov2024booking`, `aminian2025fraudtransformer`, `cheng2020stan`) almost all use proprietary data, and often depend on pretraining scale or on feature sets the baseline didn't get.
2. **Graph-model advantages shrink against strong tabular baselines.** Elliptic (`weber2019elliptic`) and GADBench (`tang2023gadbench`) find trees plus neighbourhood features competitive with or better than GNNs. No study varies *how much* relational structure the fraud has, so we don't know when GNNs should win.
3. **Operational realism is under-represented.** Label delay and feedback (`dalpozzolo2018realistic`), drift (`gama2014drift`), capacity-limited alerting (`dalpozzolo2014lessons`) and calibration (`dalpozzolo2015calibrating`) are well understood for random forests on proprietary data. Model-family comparisons under these conditions are close to absent.
4. **Foundation models aren't yet tested in the fraud regime.** TabPFN v2 is limited to about 10k rows, and LLMs lag classifiers on tabular fraud (`tan2026finfre`). The literature lacks a controlled test of whether these models help *under extreme imbalance and temporal evaluation*.
5. **Synthetic data is the only route to controlled experiments.** Existing generators each cover one slice: PaySim has one scenario, AMLworld covers laundering only, BAF covers account opening without a graph, and the Handbook has three simple scenarios. None combines pluggable scenarios with a drift schedule, a label-delay model and a graph.

---

## 10. Gap analysis

Coverage legend: **●** well-tested (several independent studies, ideally with public data), **◐** partially tested (one or two studies, or proprietary only), **○** essentially untested.

| # | Model family | Data regime | Evaluation condition | Coverage | Evidence | Gap → RQ |
|---|---|---|---|---|---|---|
| G1 | Tuned GBDT vs. tabular foundation models (TabPFN v2, FT-Transformer) | Extreme imbalance (0.1–1% fraud), >10k rows | Strict temporal split, AUPRC, ≥5 seeds | ○ | TabPFN evaluated on balanced small data (5.3, 5.4); FT-T on non-fraud (5.1); Booking.com proprietary with pretraining (5.5) | **RQ1** |
| G2 | All families | Designed adversarial drift (fraud scenario switch mid-stream) | Metric over time, static vs. periodic retraining | ○ | Drift studied only on real streams with unknown causes (4.3, 6.2) | **RQ2** |
| G3 | GNN vs. GBDT + hand-crafted graph features | Controlled ring / mule / shared-device fraud | Temporal split, no future-edge leakage | ◐ | GADBench / Elliptic favour trees (4.3, 4.9), but fraud structure unknown; BRIGHT shows leakage risk (4.7) | **RQ3** |
| G4 | All families; unsupervised pretraining as mitigation | Realistic label delay (7–30 days) | Train only on labels available at train time | ◐ | Studied for random forests only (1.5, 6.2); unsupervised features studied without delay (2.4) | **RQ4** |
| G5 | All families | Imbalance + per-transaction loss | Cost under fixed daily investigator capacity (top-k/day) | ◐ | Cost-sensitive trees (1.4) and alert precision (1.1) exist separately; not combined across families | **RQ5** |
| G6 | Sequence models (GRU) vs. GBDT + velocity features | Per-account histories, scenario-specific (ATO, card testing) | Per-scenario recall | ◐ | Proprietary comparisons with unequal feature sets (3.1, 3.2, 3.4) | Secondary analysis in RQ1/RQ2 |
| G7 | Calibration of every family | Extreme imbalance | ECE, reliability | ○ | Undersampling calibration only (1.2) | Reported in all RQs |
| G8 | LLM-derived features (small local LLMs) | Fraud tabular | Temporal validation, latency per 1k | ○ | Large hosted LLMs on non-fraud (5.7); direct classification lags (5.8) | Optional; likely future work |
| G9 | Synthetic generator | Multi-scenario + drift schedule + label delay + graph, seeded | Determinism, scenario signature tests | ○ | Generators cover single slices (7.1–7.4, 4.10) | **Contribution of Phase 3** |
| G10 | Investigator feedback loop / selection bias | Only alerted transactions labelled promptly | Labels depend on the previous model | ◐ | Addressed by control group (5.5) and feedback learning (1.5) | Future work (noted in FUTURE_WORK.md) |

These gaps are carried into `docs/RESEARCH_QUESTIONS.md` (Phase 2).
