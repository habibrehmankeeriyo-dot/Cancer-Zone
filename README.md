---
title: Cancer Mutation Detector
emoji: 🧬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Detect cancer driver mutations from genome MAF files
tags:
  - biology
  - genomics
  - cancer
  - bioinformatics
  - DNA
---

# Cancer Mutation Detector

Upload a somatic mutation file (MAF/TSV format) from cBioPortal or TCGA and get:

- **Driver vs passenger** mutation classification
- **Clinical evidence tier** (Tier 1 / 2 / 3)
- **COSMIC Cancer Gene Census** cross-reference
- **Cancer type hints** per gene

## Model

Uses [Nucleotide Transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref) fine-tuned on TCGA somatic mutation data.

## Data sources

- [cBioPortal](https://www.cbioportal.org) — free, no login needed
- [TCGA GDC Portal](https://portal.gdc.cancer.gov)

## Disclaimer

This tool is for **research purposes only**. It is not a clinical diagnostic tool.
