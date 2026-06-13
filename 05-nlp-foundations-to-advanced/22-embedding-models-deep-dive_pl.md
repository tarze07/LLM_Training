# Modele Embeddingów — Dogłębna Analiza 2026

> Word2Vec dawał ci wektor na słowo. Nowoczesne modele embeddingów dają ci wektor na fragment, wielojęzyczny, z widokami rzadkimi, gęstymi i wielowektorowymi, dopasowanymi rozmiarem do twojego indeksu. Wybierz źle, a twój RAG odzyskuje niewłaściwe rzeczy.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## Problem

Twój system RAG odzyskuje niewłaściwy fragment w 40% przypadków. Winowajcą rzadko jest baza danych wektorów czy podpowiedź. To model embeddingów.

Wybór embeddingu w 2026 oznacza decyzję wzdłuż pięciu osi:

1. **Gęste vs rzadkie vs wielowektorowe.** Jeden wektor na fragment, czy jeden na token, czy rzadki ważony worek słów.
2. **Pokrycie językowe.** Jednojęzyczne modele angielskie wciąż wygrywają na zadaniach tylko po angielsku. Modele wielojęzyczne wygrywają, gdy korpusy są mieszane.
3. **Długość kontekstu.** 512 tokenów vs 8,192 vs 32,768 — a rzeczywista efektywna pojemność to często 60-70% reklamowanego maksimum.
4. **Budżet wymiarów.** 3,072 liczby zmiennoprzecinkowe w pełnej precyzji = 12 KB na wektor. Przy 100M wektorów, przechowywanie kosztuje $1,300/miesiąc. Przycinanie Matryoshka redukuje to 4×.
5. **Otwarte vs hostowane.** Otwarte wagi oznaczają, że kontrolujesz stos i dane. Hostowane oznacza, że wymieniasz kontrolę na zawsze-najnowsze.

Ta lekcja nazywa kompromisy, abyś mógł wybrać na podstawie dowodów, a nie tego, co było popularne w zeszłym kwartale.

## Koncepcja

![Gęste, rzadkie i wielowektorowe embeddingi](../assets/embedding-modes.svg)

**Gęste embeddingi.** Jeden wektor na fragment (zwykle 384-3,072 wymiarów). Podobieństwo cosinusowe rankuje fragmenty według bliskości semantycznej. OpenAI `text-embedding-3-large`, tryb gęsty BGE-M3, Voyage-3. Domyślny wybór.

**Rzadkie embeddingi.** Styl SPLADE. Transformer przewiduje wagę dla każdego tokena słownika, a następnie zeruje większość z nich. Wynikiem jest rzadki wektor rozmiaru |vocab|. Wychwytuje dopasowanie leksykalne (jak BM25), ale z nauczonymi wagami termów. Silny na zapytaniach z dużą liczbą słów kluczowych.

**Wielowektorowe (późna interakcja).** ColBERTv2, Jina-ColBERT. Jeden wektor na token. Punktowanie za pomocą MaxSim: dla każdego tokena zapytania znajdź najbardziej podobny token dokumentu, zsumuj wyniki. Bardziej kosztowne w przechowywaniu i punktowaniu, ale wygrywa na długich zapytaniach i korpusach domenowych.

**BGE-M3: wszystkie trzy naraz.** Pojedynczy model wyprowadza jednocześnie reprezentacje gęste, rzadkie i wielowektorowe. Każda może być odpytywana niezależnie; wyniki są łączone przez ważoną sumę. Domyślny wybór w 2026, gdy chcesz elastyczności z jednego punktu kontrolnego.

**Matryoshka Representation Learning.** Trenowane tak, że pierwsze N wymiarów wektora tworzy użyteczny samodzielny embedding. Przytnij wektor 1,536-wymiarowy do 256 wymiarów i zapłać ~1% dokładności za 6× oszczędności w przechowywaniu. Obsługiwane przez OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Tablica liderów MTEB opowiada tylko część historii

Massive Text Embedding Benchmark — 56 zadań w 8 typach zadań na starcie (2022), rozszerzone do 100+ zadań w MTEB v2. Na początku 2026, Gemini Embedding 2 jest na szczycie wyszukiwania (67.71 MTEB-R). Cohere embed-v4 prowadzi w ogólnym (65.2 MTEB). BGE-M3 prowadzi w otwartych wagach wielojęzycznych (63.0). Tablica liderów jest konieczna, ale niewystarczająca — zawsze benchmarkuj na swojej domenie.

### Wzór trzech poziomów

| Przypadek użycia | Wzór |
|----------|---------|
| Szybki pierwszy przebieg | Gęsty bi-enkoder (BGE-M3, text-3-small) |
| Zwiększenie odzysku | Rzadki (SPLADE, BGE-M3 sparse) + RRF fuzja |
| Precyzja w top-50 | Wielowektorowy (ColBERTv2) lub cross-encoder reranker |

Większość produkcyjnych stosów używa wszystkich trzech.

## Zbuduj To

### Krok 1: linia bazowa — gęste embeddingi z Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` sprawia, że iloczyn skalarny jest równy podobieństwu cosinusowemu. Zawsze to ustawiaj.

### Krok 2: Przycinanie Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Ponownie normalizuj po przycięciu. Nomic v1.5, OpenAI text-3 i Voyage-4 są trenowane tak, że jest to bezstratne dla pierwszych kilku poziomów. Modele nie-Matryoshka (oryginalny Sentence-BERT) degradują się gwałtownie po przycięciu.

### Krok 3: Wielofunkcyjność BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Trzy indeksy, jedno wywołanie inferencji. Fuzja wyników:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Dostosuj wagi do swojej domeny.

### Krok 4: Ewaluacja MTEB na niestandardowym zadaniu

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Uruchom swoje modele kandydujące na *reprezentatywnym* podzbiorze. Nie ufaj samej pozycji w tablicy liderów — twoja domena ma znaczenie.

### Krok 5: ręcznie napisane podobieństwo cosinusowe od zera

Zobacz `code/main.py`. Embeddingi uśrednionego triku haszującego (tylko biblioteka standardowa). Nie konkurują z embeddingami transformerowymi, ale pokazują kształt: tokenizuj → wektor → normalizuj → iloczyn skalarny.

## Pułapki

- **Ten sam model dla zapytania i dokumentu.** Niektóre modele (Voyage, Jina-ColBERT) używają kodowania asymetrycznego — zapytanie i dokument przechodzą przez różne ścieżki. Zawsze sprawdzaj kartę modelu.
- **Brakujący prefiks.** Modele `bge-*` potrzebują `"Represent this sentence for searching relevant passages: "` poprzedzającego zapytania. 3-5 punktów luki w odzysku, jeśli zapomnisz.
- **Zbyt agresywne przycinanie Matryoshka.** 1,536 → 256 jest zwykle bezpieczne. 1,536 → 64 nie jest. Waliduj na swoim zbiorze ewaluacyjnym.
- **Obcięcie kontekstu.** Większość modeli po cichu obcina dane wejściowe powyżej swojej maksymalnej długości. Długie dokumenty wymagają dzielenia na fragmenty (patrz lekcja 23).
- **Ignorowanie ogona opóźnienia.** Wyniki MTEB ukrywają p99 opóźnienia. Model 600M może pokonać model 335M o 2 punkty, ale kosztować 3× więcej na zapytanie.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Tylko angielski, szybki, API | `text-embedding-3-large` lub `voyage-3-large` |
| Otwarte wagi, angielski | `BAAI/bge-large-en-v1.5` |
| Otwarte wagi, wielojęzyczne | `BAAI/bge-m3` lub `Qwen3-Embedding-8B` |
| Długi kontekst (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Wdrożenie tylko na CPU | Nomic Embed v2 (137M parametrów, MoE) |
| Ograniczone przechowywanie | Przycięte Matryoshka + kwantyzacja int8 |
| Zapytania z dużą liczbą słów kluczowych | Dodaj SPLADE rzadki, RRF-fuzja z gęstym |

Wzór w 2026: zacznij od BGE-M3 lub text-3-large, oceń na swojej domenie za pomocą MTEB, wymień, jeśli model specyficzny dla domeny wygrywa o więcej niż 3 punkty.

## Wdróż To

Zapisz jako `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Ćwiczenia

1. **Łatwe.** Zakoduj 100 zdań z `bge-small-en-v1.5` w pełnym wymiarze (384), a następnie w Matryoshka 128. Zmierz spadek MRR na 10 zapytaniach.
2. **Średnie.** Porównaj BGE-M3 gęsty, rzadki i colbert na 500 fragmentach z twojej domeny. Który wygrywa na recall@10? Czy fuzja RRF pokonuje najlepszy pojedynczy tryb?
3. **Trudne.** Uruchom MTEB na trzech modelach kandydujących na twoich 2 najlepszych zadaniach domenowych. Raportuj wynik MTEB, p99 opóźnienia na partii 100 zapytań i $/1M zapytań. Wybierz Pareto-optymalny.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Dense embedding | Wektor | Jeden wektor o stałym rozmiarze na tekst. Podobieństwo cosinusowe do rankingu. |
| Sparse embedding | Nauczone BM25 | Jedna waga na token słownika; głównie zera; trenowane end-to-end. |
| Multi-vector | Styl ColBERT | Jeden wektor na token; punktowanie MaxSim; większy indeks, lepszy odzysk. |
| Matryoshka | Sztuczka rosyjskiej lalki | Pierwsze N wymiarów to samodzielnie poprawny mniejszy embedding. |
| MTEB | Benchmark | Massive Text Embedding Benchmark — 56 zadań na starcie, 100+ w v2. |
| BEIR | Benchmark wyszukiwania | 18 zadań wyszukiwania zero-shot; często cytowany za solidność międzydomenową. |
| Asymmetric encoding | Zapytanie ≠ ścieżka dokumentu | Model używa różnych projekcji dla zapytań i dokumentów. |

## Dalsza Lektura

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — artykuł o bi-enkoderze.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — artykuł o tablicy liderów.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — ujednolicony model trzech trybów.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — cel treningowy drabiny wymiarów.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — późna interakcja w produkcji.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — rankingi na żywo.