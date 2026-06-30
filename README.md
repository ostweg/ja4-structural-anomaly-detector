# Learning the Grammar of Network Sessions

A Tabular BERT approach to structural fingerprint auditing using JA4+.

This repository contains the implementation accompanying the paper *"Learning the Grammar of Network Sessions: A Tabular BERT Approach to Structural Fingerprint Auditing using JA4+"* (Gast & Zagidullina, 2026, HSLU).

## Overview

Modern network traffic is overwhelmingly encrypted, which limits the effectiveness of traditional Deep Packet Inspection (DPI). This project explores an alternative: instead of inspecting payloads, we learn the **structural grammar** of TLS and HTTP handshakes using **JA4** and **JA4H** fingerprints.

Each network session is decomposed into a fixed 16-slot sequence of categorical features derived from its JA4/JA4H fingerprint. A BERT-inspired transformer encoder (**TabularBERT**) is trained with a Masked Language Modeling (MLM) objective to learn normal handshake patterns. At inference time, the model's reconstruction loss is used as an anomaly score — sessions that violate the learned "grammar" (e.g. unusual TLS version / cipher count combinations) receive high reconstruction loss and are flagged for further inspection.

The model is benchmarked against a One-Class SVM (OC-SVM) baseline trained on the same unlabeled data.

## How It Works

1. **Fingerprint extraction** — JA4 and JA4H fingerprints are parsed and split into their structural sub-fields (protocol, TLS version, SNI flag, cipher/extension counts, ALPN, HTTP method, header counts, etc.), forming a 16-slot feature sequence per session.
2. **Tokenization** — Each parsed sub-field value is mapped to an integer ID via a global vocabulary built from the training corpus.
3. **Self-supervised training** — During training, 15% of the 16 slots are randomly replaced with a `[MASK]` token. The model predicts the original values using the remaining context, optimized with cross-entropy loss via AdamW.
4. **Anomaly scoring** — At inference, the model scores complete (unmasked) sequences. The reconstruction loss (sum of per-token negative log-likelihoods) reflects how "surprising" a session's structure is relative to learned normal traffic.
5. **Thresholding** — A reconstruction loss threshold separates benign from anomalous sessions, used to drive a two-stage hybrid security framework (TabularBERT as a lightweight gatekeeper, routing only flagged sessions to deeper DPI-based inspection).

## Architecture

| Parameter | Value |
|---|---|
| Sequence Length | 16 protocol slots |
| Vocabulary Size | ~104,000 tokens |
| Embedding Dimension | 64 |
| Attention Heads | 4 |
| Encoder Layers | 3 |
| Feed-Forward Hidden Size | 128 |
| Optimization | AdamW |
| Regularization | Early Stopping |

## Data

- **Training data**: unlabeled JA4/JA4H fingerprints derived from private Microsoft network sessions (sign-in and Graph API activity), assumed to be predominantly benign.
- **Test data**: a curated, labeled dataset combining benign Microsoft sessions with malware-infected traffic extracted from public PCAP sources ([Malware-Traffic-Analysis.net](https://www.malware-traffic-analysis.net/) and [Stratosphere IPS](https://www.stratosphereips.org/)) using TShark with the JA4+ Wireshark plugin.

The dataset can be found here: https://github.com/ostweg/malicious-ja4-fingerprints

> **Note:** The private Microsoft dataset used for training is not included in this repository due to data sharing restrictions. 

## Results

| Model | TPR | FPR | Precision | F1-Score |
|---|---|---|---|---|
| TabularBERT | 0.96 | 0.00 | 1.00 | 0.98 |
| OC-SVM | 0.93 | 0.00 | 1.00 | 0.96 |

TabularBERT outperforms the OC-SVM baseline, particularly in detection rate (TPR), at the cost of higher training time and compute.

## Limitations

- Evaluated on a relatively small test set (215 malicious / 800 benign samples); reported metrics should be interpreted with appropriate caution.
- The decision threshold was calibrated specifically for the Microsoft traffic profile used in training and may not generalize to other organizational network environments without recalibration.
- As with most fingerprinting research, results reflect a closed-world evaluation setting and may be subject to concept drift over time.

## Future Work

- Dynamic, organization-specific threshold calibration
- Model compression (quantization, knowledge distillation) for low-latency deployment
- Release of an open-source, fully labeled JA4/JA4H benchmark dataset

## Citation

If you use this work, please cite:

```
Gast, S., & Zagidullina, A. (2026). Learning the Grammar of Network Sessions:
A Tabular BERT Approach to Structural Fingerprint Auditing using JA4+.
Lucerne University of Applied Sciences and Arts (HSLU).
```

## Acknowledgments

This research made use of JA4+ fingerprinting standards developed by [FoxIO](https://ja4db.com/).
