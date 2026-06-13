# Wyszukiwanie Informacji i Wyszukiwanie

> BM25 jest precyzyjny, ale kruchy. Gęste zarzuca szeroką sieć, ale gubi słowa kluczowe. Hybryda to domyślny wybór w 2026 roku. Cała reszta to strojenie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## Problem

Użytkownik wpisuje "co się stanie, jeśli ktoś kłamie, żeby zdobyć pieniądze" i oczekuje znalezienia przepisu, który to faktycznie reguluje: "Sekcja 420 Kodeksu Karnego." Wyszukiwanie słów kluczowych całkowicie to pomija (brak wspólnego słownictwa). Wyszukiwanie semantyczne pomija to, jeśli osadzenia nie były trenowane na tekstach prawnych. Prawdziwe wyszukiwanie musi radzić sobie z oboma.

IR to potok pod każdym systemem RAG, każdym paskiem wyszukiwania, każdym rozmytym wyszukiwaniem na stronie dokumentacji. Architektura 2026, która działa w produkcji, to nie pojedyncza metoda. To łańcuch uzupełniających się metod, z których każda łapie błędy poprzedniej.

Ta lekcja buduje każdy element i nazywa, które błędy każdy z nich łapie.

## Koncepcja

![Hybrydowe wyszukiwanie: BM25 + gęste + RRF + reranking cross-encoder](../assets/retrieval.svg)

Cztery warstwy. Wybierz te, których potrzebujesz.

1. **Rzadkie wyszukiwanie (BM25).** Szybkie, precyzyjne dla dokładnych dopasowań, okropne dla semantyki. Działa na odwróconym indeksie. Poniżej 10 ms na zapytanie dla milionów dokumentów. Poprawnie znajduje odniesienia do przepisów, kody produktów, komunikaty błędów, nazwy własne.
2. **Gęste wyszukiwanie.** Zakoduj zapytanie i dokumenty w wektory. Wyszukiwanie najbliższych sąsiadów. Wychwytuje parafrazy i podobieństwo semantyczne. Gubi dokładne dopasowania słów kluczowych różniące się jednym znakiem. 50-200 ms na zapytanie z FAISS lub bazą wektorową.
3. **Fuzja.** Połącz rankingi z rzadkiego i gęstego wyszukiwania. Reciprocal Rank Fusion (RRF) to łatwy domyślny wybór, ponieważ ignoruje surowe wyniki (które są w różnych skalach) i używa tylko pozycji w rankingu. Ważona fuzja jest opcją, gdy wiesz, że jeden sygnał dominuje w twojej domenie.
4. **Reranking cross-encoder.** Weź górne 30 z fuzji. Uruchom cross-encoder (zapytanie + dokument razem, oceniając każdą parę). Zachowaj górne 5. Cross-encodery są wolniejsze na parę niż bi-encodery, ale znacznie dokładniejsze. Amortyzujesz koszt, uruchamiając je tylko na górnych 30.

Trójstronne wyszukiwanie (BM25 + gęste + learned-sparse jak SPLADE) osiąga lepsze wyniki niż dwustronne w benchmarkach 2026, ale wymaga infrastruktury dla indeksów learned-sparse. Dla większości zespołów dwustronne plus reranking cross-encoder to optymalny punkt.

## Zbuduj To

### Krok 1: BM25 od podstaw

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Dwa parametry, które warto znać. `k1=1.5` kontroluje nasycenie częstotliwości terminów; wyższa wartość oznacza większą wagę powtórzeń terminów. `b=0.75` kontroluje normalizację długości; 0 ignoruje długość dokumentu, 1 w pełni normalizuje. Wartości domyślne to zalecenia Robertsona z oryginalnego artykułu i rzadko wymagają strojenia.

### Krok 2: gęste wyszukiwanie z bi-encoderem

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

Normalizuj L2 osadzenia, aby iloczyn skalarny był równy cosinusowi. `all-MiniLM-L6-v2` ma 384 wymiary, jest szybki i wystarczająco silny dla większości angielskiego wyszukiwania. Do pracy wielojęzycznej użyj `paraphrase-multilingual-MiniLM-L12-v2`. Dla najwyższej dokładności `bge-large-en-v1.5` lub `e5-large-v2`.

### Krok 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Stała `k=60` pochodzi z oryginalnego artykułu RRF. Wyższe `k` spłaszcza wpływ różnic w rankingu; niższe `k` sprawia, że górne pozycje dominują. 60 to opublikowana wartość domyślna i rzadko wymaga strojenia.

### Krok 4: hybrydowe wyszukiwanie + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Trzy złożone etapy. BM25 znajduje dopasowania leksykalne. Gęste znajduje dopasowania semantyczne. RRF łączy dwa rankingi bez potrzeby kalibracji wyników. Cross-encoder ponownie ocenia górne 30, używając par zapytanie-dokument razem, co wychwytuje drobnoziarnistą trafność, którą bi-encoder pominął. Zachowaj górne 5.

### Krok 5: ewaluacja

| Metryka | Znaczenie |
|--------|---------|
| Recall@k | Na ile zapytań, dla których istnieje poprawny dokument, znajduje się on w górnym k? |
| MRR (Mean Reciprocal Rank) | Średnia z 1/pozycji pierwszego trafnego dokumentu. |
| nDCG@k | Uwzględnia gradacje trafności, nie tylko binarnie trafny/nie. |

Dla RAG szczególnie **Recall@k** wyszukiwarki jest najważniejszą liczbą. Twój czytnik nie może odpowiedzieć, jeśli właściwy fragment nie znajduje się w pobranym zestawie.

Wskazówka do debugowania: dla zawodzących zapytań, porównaj rankingi rzadkie i gęste. Jeśli jeden znajduje właściwy dokument, a drugi nie, masz niedopasowanie słownictwa (naprawa: dodaj brakującą połowę) lub niejednoznaczność semantyczną (naprawa: lepsze osadzenia lub reranker).

## Użyj Tego

Stos w 2026 roku:

| Skala | Stos |
|-------|-------|
| 1k-100k dokumentów | BM25 w pamięci + osadzenia `all-MiniLM-L6-v2` + RRF. Bez osobnej bazy danych. |
| 100k-10M dokumentów | FAISS lub pgvector dla gęstego + Elasticsearch / OpenSearch dla BM25. Uruchom równolegle. |
| 10M+ dokumentów | Qdrant / Weaviate / Vespa / Milvus z obsługą hybrydową. Reranking cross-encoder na górnych 30. |
| Najwyższa jakość | Trójstronne (BM25 + gęste + SPLADE) + reranking ColBERT late-interaction |

Cokolwiek wybierzesz, zaplanuj budżet na ewaluację. Mierz recall wyszukiwarki przed benchmarkowaniem dokładności RAG end-to-end. Czytnik nie może naprawić tego, czego wyszukiwarka nie znalazła.

### Trudno wyciągnięte lekcje z produkcyjnego RAG w 2026

- **80% awarii RAG wynika z ingestii i chunkowania, a nie z modelu.** Zespoły spędzają tygodnie na wymianie LLM i strojeniu promptów, podczas gdy wyszukiwarka cicho zwraca zły kontekst co trzecie zapytanie. Najpierw napraw chunkowanie.
- **Strategia chunkowania ma większe znaczenie niż rozmiar chunka.** Podziały o stałej wielkości niszczą tabele, kod i zagnieżdżone nagłówki. Świadomość zdań jest domyślna; chunkowanie semantyczne lub oparte na LLM opłaca się w przypadku dokumentacji technicznej i instrukcji produktów.
- **Wzorzec parent-doc.** Pobieraj małe "dziecięce" chunki dla precyzji. Gdy wiele dzieci z tej samej sekcji nadrzędnej pojawia się razem, zamień na blok nadrzędny, aby zachować kontekst. To konsekwentnie podnosi jakość odpowiedzi bez konieczności ponownego trenowania.
- **k_rerank=3 jest zazwyczaj optymalne.** Każdy dodatkowy chunk powyżej tego zwiększa koszt tokenów i opóźnienie generacji bez podnoszenia jakości odpowiedzi. Jeśli k=8 jest wciąż lepsze niż k=3, reranker działa słabo.
- **HyDE / rozszerzanie zapytania.** Wygeneruj hipotetyczną odpowiedź z zapytania, osadź ją, wyszukaj. Mostkuje lukę w sformułowaniach między krótkimi pytaniami a długimi dokumentami. Darmowy wzrost precyzji bez trenowania.
- **Budżet kontekstu poniżej 8K tokenów.** Konsekwentne trafienia na tym limicie oznaczają, że próg rerankera jest zbyt luźny.
- **Wersjonuj wszystko.** Prompty, reguły chunkowania, model osadzania, reranker. Każdy dryf cicho psuje jakość odpowiedzi. Bramki CI na wierność, precyzję kontekstu i wskaźnik nieodpowiedzianych pytań blokują regresje, zanim zobaczą je użytkownicy.
- **Trójstronne wyszukiwanie (BM25 + gęste + learned-sparse jak SPLADE) osiąga lepsze wyniki niż dwustronne** w benchmarkach 2026, szczególnie dla zapytań mieszających nazwy własne z semantyką. Wdróż, gdy infrastruktura obsługuje indeksy SPLADE.

Odpowiedni projekt wyszukiwania zmniejsza halucynacje o 70-90% według pomiarów branżowych z 2026 roku. Większość zysków wydajności RAG pochodzi z lepszego wyszukiwania, a nie z fine-tuningu modelu.

## Wdróż To

Zapisz jako `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `hybrid_search` powyżej na korpusie 500 dokumentów. Przetestuj 20 zapytań. Porównaj recall na 5 między BM25-only, dense-only i hybrydą.
2. **Średnie.** Dodaj obliczanie MRR. Dla każdego testowego zapytania ze znanym poprawnym dokumentem, znajdź pozycję poprawnego dokumentu w rankingach BM25, gęstym i hybrydowym. Podaj MRR dla każdego.
3. **Trudne.** Dostrój gęsty encoder na swojej domenie używając MultipleNegativesRankingLoss (Sentence Transformers). Zbuduj zestaw treningowy z 500 par zapytanie-dokument. Porównaj recall przed i po fine-tuningu.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| BM25 | Wyszukiwanie słów kluczowych | Okapi BM25. Ocenia dokumenty według częstotliwości terminów, IDF i długości. |
| Gęste wyszukiwanie | Wyszukiwanie wektorowe | Zakoduj zapytanie + dokument w wektory, znajdź najbliższych sąsiadów. |
| Bi-encoder | Model osadzania | Koduje zapytanie i dokument niezależnie. Szybki w czasie zapytania. |
| Cross-encoder | Model rerankera | Koduje zapytanie + dokument razem. Wolny, ale dokładny. |
| RRF | Fuzja rankingów | Łączy dwa rankingi przez sumowanie `1/(k + rank)`. |
| Recall@k | Metryka wyszukiwania | Frakcja zapytań, dla których trafny dokument jest w górnym k. |

## Dalsza Lektura

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — definitywne opracowanie BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, kanoniczny bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — learned-sparse wyszukiwarka, która zamyka lukę z gęstą.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — artykuł RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — wyszukiwanie late-interaction.