# Strategie Dzielenia na Fragmenty dla RAG

> Konfiguracja dzielenia na fragmenty wpływa na jakość wyszukiwania tak samo jak wybór modelu embeddingów (Vectara NAACL 2025). Źle ustawione dzielenie i żadne rerankingowanie cię nie uratuje.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## Problem

Umieszczasz 50-stronicową umowę w systemie RAG. Użytkownik pyta: "Jaka jest klauzula wypowiedzenia?" Wyszukiwarka zwraca stronę tytułową. Dlaczego? Ponieważ model był trenowany na fragmentach 512-tokenowych, a klauzula wypowiedzenia znajduje się 20 stron dalej, podzielona podziałem strony, bez lokalnych słów kluczowych łączących ją z zapytaniem.

Rozwiązaniem nie jest "kup lepszy model embeddingów." Rozwiązaniem jest dzielenie na fragmenty. Jak duże? Nakładanie? Gdzie dzielić? Z otaczającym kontekstem?

Benchmarki z lutego 2026 pokazują zaskakujące wyniki:

- Badanie Vectara z 2026: rekurencyjne dzielenie 512-tokenowe pokonało semantyczne 69% → 54% dokładności.
- SPLADE + Mistral-8B na Natural Questions: nakładanie nie przyniosło mierzalnej korzyści.
- Urwisko kontekstu: jakość odpowiedzi gwałtownie spada w okolicach 2,500 tokenów kontekstu.

"Oczywista" odpowiedź (dzielenie semantyczne, 20% nakładania, 1000 tokenów) jest często błędna. Ta lekcja buduje intuicję dla sześciu strategii i mówi, kiedy po którą sięgnąć.

## Koncepcja

![Sześć strategii dzielenia wizualizowanych na jednym fragmencie](../assets/chunking.svg)

**Dzielenie stałe.** Dziel co N znaków lub tokenów. Najprostsza linia bazowa. Łamie w środku zdania. Dobra kompresja, zła spójność.

**Rekurencyjne.** `RecursiveCharacterTextSplitter` z LangChain. Najpierw próbuj dzielić na `\n\n`, potem `\n`, potem `.`, potem spację. Bezpieczne opadanie. Domyślny wybór w 2026.

**Semantyczne.** Osadź każde zdanie. Oblicz podobieństwo cosinusowe między sąsiednimi zdaniami. Dziel, gdy podobieństwo spada poniżej progu. Zachowuje spójność tematyczną. Wolniejsze; czasami produkuje maleńkie 40-tokenowe fragmenty, które szkodzą wyszukiwaniu.

**Zdaniowe.** Dziel na granicach zdań. Jedno zdanie na fragment lub okno N zdań. Dorównuje dzieleniu semantycznemu do ~5k tokenów przy ułamku kosztu.

**Rodzic-dziecko.** Przechowuj małe fragmenty dziecięce do wyszukiwania *oraz* większy fragment rodzicielski dla kontekstu. Wyszukuj po dziecku; zwracaj rodzica. Degraduje się łagodnie: złe fragmenty dziecięce wciąż zwracają rozsądnych rodziców.

**Późne dzielenie (2024).** Osadź cały dokument najpierw na poziomie tokenów, a następnie agreguj embeddingi tokenów w embeddingi fragmentów. Zachowuje kontekst międzydzielniczy. Działa z długokontekstowymi embedderami (BGE-M3, Jina v3). Wyższe zapotrzebowanie obliczeniowe.

**Wyszukiwanie kontekstowe (Anthropic, 2024).** Poprzedź każdy fragment podsumowaniem wygenerowanym przez LLM jego pozycji w dokumencie ("Ten fragment to sekcja 3.2 klauzul wypowiedzenia..."). 35-50% poprawy wyszukiwania w benchmarku Anthropica. Drogie w indeksowaniu.

### Zasada, która bije każdą domyślną wartość

Dopasuj rozmiar fragmentu do typu zapytania:

| Typ zapytania | Rozmiar fragmentu |
|------------|-----------|
| Faktograficzne ("jak się nazywa CEO?") | 256-512 tokenów |
| Analityczne / wieloetapowe | 512-1024 tokenów |
| Zrozumienie całej sekcji | 1024-2048 tokenów |

Benchmark NVIDII z 2026. Fragment powinien być wystarczająco duży, aby zawierać odpowiedź plus lokalny kontekst, i wystarczająco mały, aby top-K wyszukiwarki skupiał się na odpowiedzi, a nie na szumie kontekstowym.

## Zbuduj To

### Krok 1: dzielenie stałe i rekurencyjne

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Krok 2: dzielenie semantyczne

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Dostosuj `threshold` do swojej domeny. Za wysoki → fragmenty. Za niski → jeden gigantyczny fragment.

### Krok 3: rodzic-dziecko

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Kluczowa obserwacja: deduplikuj rodziców. Wiele dzieci może mapować do tego samego rodzica; zwracanie wszystkich marnowałoby kontekst.

### Krok 4: wyszukiwanie kontekstowe (wzór Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Indeksuj skontekstualizowane fragmenty. W czasie zapytania wyszukiwarka korzysta z dodatkowego otaczającego sygnału.

### Krok 5: ewaluacja

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Zawsze benchmarkuj. "Najlepsza" strategia dla twojego korpusu może nie pasować do żadnego wpisu na blogu.

## Pułapki

- **Dzielenie oceniane tylko na zapytaniach faktograficznych.** Zapytania wieloetapowe ujawniają bardzo różnych zwycięzców. Używaj zbioru ewaluacyjnego stratyfikowanego według typu zapytania.
- **Semantyczne dzielenie bez minimalnego rozmiaru.** Produkuje 40-tokenowe fragmenty, które szkodzą wyszukiwaniu. Zawsze egzekwuj `min_tokens`.
- **Nakładanie jako kargo-kult.** Badania z 2026 wykazują, że nakładanie często nie przynosi korzyści i podwaja koszt indeksu. Mierz, nie zakładaj.
- **Brak egzekwowania min/max.** Fragmenty 5 tokenów lub 5000 tokenów oba psują wyszukiwanie. Zastosuj ograniczenia.
- **Międzydokumentowe dzielenie.** Nigdy nie pozwól, aby fragment obejmował dwa dokumenty. Zawsze dziel per-dokument, potem scalaj.

## Użyj Tego

Stos w 2026:

| Sytuacja | Strategia |
|-----------|----------|
| Pierwsza budowa, nieznany korpus | Rekurencyjne, 512 tokenów, bez nakładania |
| QA faktograficzne | Rekurencyjne, 256-512 tokenów |
| Analityczne / wieloetapowe | Rekurencyjne, 512-1024 tokenów + rodzic-dziecko |
| Ciężkie odsyłacze (umowy, artykuły) | Późne dzielenie lub wyszukiwanie kontekstowe |
| Korpus konwersacyjny / dialogowy | Fragmenty na poziomie wypowiedzi + metadane mówcy |
| Krótkie wypowiedzi (tweety, recenzje) | Jeden dokument = jeden fragment |

Zacznij od rekurencyjnego 512. Zmierz recall@5 na zbiorze ewaluacyjnym 50 zapytań. Dostrajaj stamtąd.

## Wdróż To

Zapisz jako `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Ćwiczenia

1. **Łatwe.** Podziel jeden 20-stronicowy dokument na fragmenty używając fixed(512, 0), recursive(512, 0) i recursive(512, 100). Porównaj liczbę fragmentów i jakość granic.
2. **Średnie.** Zbuduj zestaw ewaluacyjny 30 zapytań na 5 dokumentach. Zmierz recall@5 dla rekurencyjnego, semantycznego i rodzic-dziecko. Który wygrywa? Czy pasuje do wpisów na blogu?
3. **Trudne.** Zaimplementuj wyszukiwanie kontekstowe. Zmierz poprawę MRR względem bazowego rekurencyjnego. Raportuj koszt indeksu (wywołania LLM) vs zysk dokładności.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Chunk | Kawałek dokumentu | Jednostka poddokumentowa, która jest osadzana, indeksowana i wyszukiwana. |
| Overlap | Margines bezpieczeństwa | N tokenów współdzielonych między sąsiednimi fragmentami; często bezużyteczne w benchmarkach 2026. |
| Semantic chunking | Inteligentne dzielenie | Dziel, gdzie podobieństwo embeddingów sąsiednich zdań spada. |
| Parent-document | Dwupoziomowe wyszukiwanie | Wyszukuj małe dzieci, zwracaj większych rodziców. |
| Late chunking | Dzielenie po osadzeniu | Osadź pełny dokument na poziomie tokenów, agreguj w wektory fragmentów. |
| Contextual retrieval | Sztuczka Anthropica | Podsumowanie generowane przez LLM poprzedzające każdy fragment przed indeksowaniem. |
| Context cliff | Ściana 2500 tokenów | Spadek jakości obserwowany w okolicach 2.5k tokenów kontekstu w RAG (styczeń 2026). |

## Dalsza Lektura

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — domyślny wybór w produkcji.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — dzielenie ma znaczenie równie duże jak wybór embeddingu.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — artykuł o późnym dzieleniu.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 35-50% poprawy wyszukiwania z prefiksami kontekstu generowanymi przez LLM.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — rozmiar fragmentu według typu zapytania.