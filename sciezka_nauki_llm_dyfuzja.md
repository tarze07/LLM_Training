# Ścieżka nauki: LLM-y i Modele Dyfuzyjne

## 1. Fundamenty architektury (Transformers Deep Dive)

### `07-transformers-deep-dive/`

| Lekcja | Temat |
|--------|-------|
| 01 | Why Transformers |
| 02 | Self-Attention from scratch |
| 03 | Multi-Head Attention |
| 04 | Positional Encoding |
| 05 | Full Transformer |
| 06 | BERT – Masked Language Modeling |
| 07 | GPT – Causal Language Modeling |
| 08 | T5/BART – Encoder-Decoder |

Opcjonalnie doczytaj:
- 11 – Mixture of Experts
- 12 – KV Cache & Flash Attention
- 13 – Scaling Laws

---

## 2. LLM-y od podstaw

### `10-llms-from-scratch/`

| Lekcja | Temat |
|--------|-------|
| 01 | Tokenizers |
| 02 | Building a Tokenizer |
| 03 | Data Pipelines |
| 04 | Pre-training mini-GPT |
| 06 | Instruction Tuning (SFT) |
| 07 | RLHF |
| 08 | DPO |
| 11 | Quantization |
| 12 | Inference Optimization |
| 14 | Open Models Architecture Walkthroughs |

Opcjonalnie doczytaj:
- 05 – Scaling & Distributed Training
- 09 – Constitutional AI & Self-Improvement
- 10 – Evaluation

---

## 3. Modele dyfuzyjne

### `08-generative-ai/`

| Lekcja | Temat |
|--------|-------|
| 06 | Diffusion – DDPM from scratch |
| 07 | Latent Diffusion / Stable Diffusion |
| 08 | ControlNet & LoRA Conditioning |

Opcjonalnie doczytaj:
- 01 – Generative Models Taxonomy & History
- 02 – Autoencoders / VAE (przydatne przed dyfuzją)
- 13 – Flow Matching & Rectified Flows

---

## 4. Inżynieria LLM (zastosowania praktyczne)

### `11-llm-engineering/`

| Lekcja | Temat |
|--------|-------|
| 01 | Prompt Engineering |
| 02 | Few-Shot & Chain of Thought |
| 04 | Embeddings |
| 06 | RAG |
| 07 | Advanced RAG |
| 08 | Fine-tuning with LoRA |
| 10 | Evaluation |

---

## Rekomendowana kolejność

```
07-transformers (01–08)
    ↓
10-llms-from-scratch (01–12)
    ↓
08-generative-ai (06–08)
    ↓
11-llm-engineering (01–10)
```

## Prerequisites (opcjonalnie, jeśli brakuje podstaw)

- `01-math-foundations/` — algebra liniowa, rachunek prawdopodobieństwa
- `02-ml-fundamentals/` — podstawy ML
- `03-deep-learning-core/` — sieci neuronowe, backpropagation, optymalizacja