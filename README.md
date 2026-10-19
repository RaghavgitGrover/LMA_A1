# Dual-Stage Legal RAG: Bi-Encoder Baseline vs. Cross-Encoder Re-Ranking

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Sentence--Transformers-6.1.0-orange.svg)](https://www.sbert.net/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.15.1-green.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/LLM-Llama--3.2--3B%20(Ollama)-purple.svg)](https://ollama.ai/)
[![LaTeX Report](https://img.shields.io/badge/Report-PDF%20(6%20Pages)-red.svg)](report.pdf)

> **Course:** Language Models and Applications (LMA) — Assignment 1  
> **Author:** Raghav Grover (Roll No. 2023101102)  
> **Repository:** [https://github.com/RaghavgitGrover/LMA_A1](https://github.com/RaghavgitGrover/LMA_A1)

---

## 📌 Executive Summary

This repository contains the complete implementation, experimental benchmarking, and academic report for a **Dual-Stage Retrieval-Augmented Generation (RAG)** system evaluated on the official text of **The Constitution of India** (Articles & Schedules across pages 32–381, 880 semantic chunks).

We empirically compare and evaluate two retrieval pipelines against a curated ground-truth benchmark of **56 legal queries**:
1. **Pipeline 1 (Baseline Bi-Encoder):** Encodes chunks and queries into 384-dimensional dense vectors using `all-MiniLM-L6-v2`, retrieves Top-10 candidates via inner product search with FAISS, and passes Top-3 candidates directly to `llama3.2:3b`.
2. **Pipeline 2 (Cross-Encoder Re-Ranking):** Retrieves Top-10 candidates via the Bi-Encoder, re-scores them using full token-level cross-attention with `ms-marco-MiniLM-L-6-v2`, and feeds the re-ordered Top-3 candidates to `llama3.2:3b`.

### Key Findings
- **Hit@3 Retrieval Accuracy:** Improves from **60.4%** (Baseline) to **75.0%** (Re-Ranked) — an absolute gain of **+14.6%**.
- **Ranking Quality:** **MRR@10** increases by **+21.4%** ($0.524 \rightarrow 0.636$), and **nDCG@3** increases by **+32.2%** ($0.457 \rightarrow 0.604$).
- **Net Query Improvement:** **18.8%** (9/48) answerable queries fixed, with a low regression rate of **4.2%** (2/48).
- **Latency Profile:** Vector retrieval takes **~10 ms**, while Cross-Encoder re-ranking adds **~325 ms** (a 33× retrieval increase). However, overall latency is dominated by local LLM generation (~13 s), making the re-ranking overhead negligible in practical interactive systems.

---

## 🏗 System Architecture

```
                    ┌─────────────────────────┐
                    │      Incoming Query     │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       [ Pipeline 1: Baseline ]        [ Pipeline 2: Re-Ranked ]
                 │                               │
      Bi-Encoder (all-MiniLM-L6-v2)    Bi-Encoder (all-MiniLM-L6-v2)
                 │                               │
        FAISS IndexFlatIP               FAISS IndexFlatIP
        (Top-10 candidates)             (Top-10 candidates)
                 │                               │
                 │                               ▼
                 │                     Cross-Encoder Re-Ranking
                 │                   (ms-marco-MiniLM-L-6-v2)
                 │                   Full Cross-Attention Logits
                 │                               │
                 ▼                               ▼
        Top-3 Passages                  Top-3 Passages
        (Ranks 1, 2, 3)                 (Sorted by CE Logit)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Local LLM Generation
               (llama3.2:3b via Ollama, temp=0)
                                 │
                                 ▼
                    [ Comparative Evaluation ]
```

### Architectural Comparison

| Dimension | Baseline (Bi-Encoder) | Re-Ranked (+Cross-Encoder) |
|:---|:---|:---|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` (22.7M) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M) |
| **Representation** | Independent 384-d dense embeddings | Joint `[CLS] q [SEP] d [SEP]` token sequence |
| **Scoring Function** | Inner product $s(q, d) = \mathbf{u}_q^\top \mathbf{v}_d$ | Joint cross-attention scalar logit $s(q, d) = f(q, d)$ |
| **Complexity** | $O(N \cdot d)$ via FAISS | $O(k \cdot (L_q + L_d)^2)$ for $k$ candidates |
| **Offline Precomputation** | Fully indexed offline | Cannot be precomputed; per-query dynamic |
| **Median Retrieval Latency** | **10.02 ms** ($N = 880$ chunks) | **333.98 ms** (10.02 ms BE + 324.97 ms CE) |
| **Operational Role** | High-throughput candidate recall | High-precision candidate re-ordering |

---

## 📊 Benchmark Results

### 1. Headline Retrieval Metrics (with 95% Bootstrap Confidence Intervals)

| Pipeline | Candidate Recall@10 | Hit@3 | MRR@10 | nDCG@3 | Mean Gold Rank | Fix Rate | Regression Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (BE)** | 0.833 | 0.604 [0.458, 0.750] | 0.524 [0.409, 0.640] | 0.457 [0.343, 0.578] | 4.19 | — | — |
| **Re-Ranked (+CE)** | 0.833 | **0.750** [0.625, 0.855] | **0.636** [0.524, 0.752] | **0.604** [0.489, 0.711] | **3.42** | **18.8%** (9/48) | **4.2%** (2/48) |

### 2. Breakdown Across 8 Query Categories

| Category | Queries | Candidate Recall@10 | Baseline Hit@3 | Re-Ranked Hit@3 | Fixed | Degraded | Both OK |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exception** | 7 | 85.7% | 57.1% | **85.7%** | +2 | 0 | 4 |
| **Negation** | 7 | 85.7% | 57.1% | **85.7%** | +2 | 0 | 4 |
| **Procedure** | 7 | 100.0% | 57.1% | **85.7%** | +2 | 0 | 4 |
| **Keyword Trap** | 7 | 42.9% | 28.6% | **42.9%** | +1 | 0 | 2 |
| **Role Limit** | 7 | 100.0% | 85.7% | **100.0%** | +1 | 0 | 6 |
| **Cross-Article** | 7 | 100.0% | 71.4% | 71.4% | +1 | 1 | 4 |
| **Control** | 6 | 66.7% | 66.7% | 50.0% | 0 | 1 | 3 |
| **Unanswerable** | 7 | N/A | N/A | N/A | 0 | 0 | 0 |

### 3. Stage-Wise Latency Profiling

| Pipeline Stage | Median Latency | Mean Latency | Std Dev |
|:---|---:|---:|---:|
| **Query Encoding (Bi-Encoder)** | 9.88 ms | 9.72 ms | 0.54 ms |
| **Vector Search (FAISS IndexFlatIP)** | 0.14 ms | 0.14 ms | 0.02 ms |
| **Cross-Encoder Scoring ($k=10$)** | 324.97 ms | 322.03 ms | 12.18 ms |
| **Total Baseline Retrieval** | **10.02 ms** | **9.85 ms** | 0.55 ms |
| **Total Re-Ranked Retrieval** | **333.98 ms** | **331.88 ms** | 12.21 ms |
| **LLM Generation (Answer 1)** | 13,160.00 ms | 14,654.90 ms | 3,412.10 ms |
| **LLM Generation (Answer 2)** | 12,740.00 ms | 13,454.30 ms | 2,891.40 ms |

---

## 📁 Repository Structure

```
.
├── rag_pipeline.py            # Standalone, end-to-end Python pipeline & benchmark CLI
├── rag_pipeline.ipynb         # Interactive Jupyter Notebook mirroring the benchmark
├── constitution_of_india.pdf  # Primary corpus source (Official Constitution of India)
├── report.tex                 # LaTeX source code for the 6-page academic report
├── report.pdf                 # Compiled 6-page comprehensive academic report
├── etoolbox.sty               # TeX macro package fallback
├── results/                   # Experimental benchmark outputs and artifacts
│   ├── figures/               # High-resolution plots (rank_shift, latency, scaling)
│   ├── run_config.json        # Execution environment and hyperparameter manifest
│   ├── queries.json           # 56 ground-truth queries, annotations, and gold regexes
│   ├── chunks.jsonl           # 880 preprocessed semantic chunks with Article metadata
│   ├── embeddings.npy         # Precomputed 384-d normalized float32 embeddings
│   ├── cache_manifest.json    # SHA-256 cryptographic cache verification
│   ├── metrics_summary.csv    # Overall retrieval metrics with 95% bootstrap CIs
│   ├── metrics_by_category.csv# Category-level accuracy breakdown
│   ├── rank_shift.csv         # Per-candidate rank transition log
│   ├── status_counts.csv      # Status classification (Fixed, Degraded, Both OK, etc.)
│   ├── answers_comparison.csv # Side-by-side textual and factual comparison of LLM answers
│   ├── latency.csv            # Microsecond-level per-query latency logs
│   ├── latency_summary.csv    # Aggregate latency statistics per stage
│   ├── latency_scaling.csv    # Empirical CPU scaling measurements across corpus subsets
│   ├── latency_projection.csv # Theoretical scaling extrapolation up to 10M chunks
│   ├── candidate_pool_sweep.csv # Sensitivity analysis for k in {3, 5, 10, 20}
│   └── report_tables.md       # Pre-rendered Markdown tables for quick inspection
├── .gitignore                 # Ignores venv, LaTeX artifacts, caches, and logs
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python:** 3.10, 3.11, or 3.12
- **System Utilities:** `pdftotext` (via `poppler-utils` on Linux / macOS)
- **Ollama:** Installed and running locally with `llama3.2:3b` pulled
- **LaTeX:** `pdflatex` (optional, only needed if recompiling the PDF report)

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y poppler-utils texlive-latex-base texlive-latex-extra

# macOS
brew install poppler
```

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/RaghavgitGrover/LMA_A1.git
cd LMA_A1

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install torch sentence-transformers faiss-cpu numpy pandas matplotlib requests pymupdf ollama
```

### 3. Pull the Ollama Model

```bash
# Ensure ollama server is running in background
ollama serve &

# Pull Llama-3.2-3B
ollama pull llama3.2:3b
```

---

## 💻 Running the Benchmark

### Option A: Complete Pipeline Execution (Retrieval + LLM Generation)

Run the full benchmark on the 56 queries, measuring latency with 5 repeats per stage:
```bash
python rag_pipeline.py --pdf constitution_of_india.pdf --llm llama3.2:3b --top-k 10 --top-n 3 --repeats 5 --out results
```

### Option B: Fast Retrieval-Only Benchmark (Skips LLM Generation)

To evaluate retrieval metrics, candidate pools, and ranking shifts without invoking Ollama:
```bash
python rag_pipeline.py --pdf constitution_of_india.pdf --skip-llm --out results
```

### Option C: Interactive Jupyter Notebook

Launch Jupyter and open `rag_pipeline.ipynb`:
```bash
jupyter notebook rag_pipeline.ipynb
```

### Command-Line Arguments Reference

| Flag | Default | Description |
|:---|:---:|:---|
| `--pdf` | `constitution_of_india.pdf` | Path to the source Constitution PDF |
| `--llm` | `llama3.2:3b` | Local Ollama model name |
| `--top-k` | `10` | Number of initial candidates retrieved by Bi-Encoder |
| `--top-n` | `3` | Number of top passages supplied to LLM context |
| `--repeats` | `5` | Latency measurement repetitions per query |
| `--skip-llm` | `False` | Run retrieval evaluation only (bypasses LLM inference) |
| `--show-chunks` | `0` | Print $N$ random chunks from the corpus and exit |
| `--out` | `results` | Output directory for all CSVs, JSONs, and figures |

---

## 📄 Recompiling the Academic Report

The academic report is written in LaTeX (`report.tex`) and formatted to fit a 6-page layout. To recompile:

```bash
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex  # Second pass for cross-references
```

The output will be saved to `report.pdf`.

---

## 🔬 Methodology Highlights

### 1. Robust Chunking & Sanitization
- Operative text extracted from pages 32 to 381 (Articles 1 through 395 and Schedules).
- Footnotes removed via position-aware heuristic ($y \ge 0.45 \cdot \text{page\_height}$) to prevent truncation of legal clauses.
- Fixed-size sliding window (900 characters with 150-character overlap) preserves clause coherence, yielding exactly 880 chunks.

### 2. Ground-Truth Query Benchmark
- 56 queries across 8 distinct categories designed to test complex linguistic and legal phenomena:
  - **Negation & Exceptions:** Articles with "notwithstanding" clauses or disqualification overrides.
  - **Keyword Traps:** Queries where naive lexical or bi-encoder overlap matches irrelevant clauses.
  - **Cross-Article Relationships:** Queries requiring joint reasoning across distinct constitutional provisions.
  - **Unanswerable / Out-of-Scope:** Queries probing legal topics absent from the text to test hallucination resistance.

### 3. Strict Legal Prompting
- Zero-temperature generation with exact negative fallback: `"The provided context does not contain the answer."`
- Eliminates extrinsic hallucination and forces direct reliance on Top-3 retrieved passages.

---

## 📜 Citation & Academic Use

If you use this benchmark, code, or experimental setup, please cite:

```bibtex
@misc{grover2026dualrag,
  author = {Grover, Raghav},
  title = {Dual-Stage Legal RAG: Bi-Encoder Baseline vs. Cross-Encoder Re-Ranking on The Constitution of India},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RaghavgitGrover/LMA_A1}},
  note = {Language Models and Applications (LMA), Assignment 1}
}
```
