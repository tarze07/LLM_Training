# Streszczanie Tekstu

> Systemy ekstrakcyjne mówią ci, co dokument powiedział. Systemy abstrakcyjne mówią ci, co autor miał na myśli. Różne zadania, różne pułapki.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## Problem

Artykuł liczący 2,000 słów ląduje w twoim feedzie. Potrzebujesz 120 słów, które go oddają. Możesz albo wybrać trzy najważniejsze zdania z artykułu (ekstrakcyjnie) albo przepisać treść własnymi słowami (abstrakcyjnie). Obie nazywają się streszczaniem. To zupełnie różne problemy.

Streszczanie ekstrakcyjne to problem rankingowy. Oceń każde zdanie, zwróć top-`k`. Output jest zawsze poprawny gramatycznie, ponieważ jest dosłownie zaczerpnięty. Ryzyko polega na pominięciu treści rozproszonej po artykule.

Streszczanie abstrakcyjne to problem generacji. Transformer produkuje nowy tekst warunkowany wejściem. Output jest płynny i zwięzły, ale może halucynować fakty, których nie było w źródle. Ryzyko to pewna siebie fabrykacja.

Ta lekcja buduje oba, wraz z trybem awarii, za który każdy odpowiada.

## Koncepcja

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Ekstrakcyjne.** Traktuj artykuł jako graf, gdzie węzły to zdania, a krawędzie to podobieństwa. Uruchom PageRank (lub coś podobnego) na grafie, aby ocenić zdania według tego, jak bardzo są połączone z resztą. Zdania z najwyższą oceną to streszczenie. Kanoniczną implementacją jest **TextRank** (Mihalcea i Tarau, 2004).

**Abstrakcyjne.** Dostrój transformer enkoder-dekoder (BART, T5, Pegasus) na parach dokument-streszczenie. Przy inferencji model czyta dokument i generuje streszczenie token po tokenie przez cross-attention. Pegasus w szczególności używa obiektywu pretreningowego gap-sentence, co czyni go doskonałym do streszczania bez dużej ilości dostrajania.

Ewaluacja z **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 i ROUGE-2 punktują nakładanie się unigramów i bigramów. ROUGE-L punktuje najdłuższy wspólny podciąg. Wyższe jest lepsze, ale 40 ROUGE-L to "dobrze", a 50 to "wyjątkowo". Każdy artykuł raportuje wszystkie trzy. Używaj pakietu `rouge-score`.

## Zbuduj to

### Krok 1: TextRank (ekstrakcyjny)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Dwie rzeczy warte wymienienia. Funkcja podobieństwa używa logarytmicznie znormalizowanego nakładania się słów, co jest oryginalnym wariantem TextRank. Cosinus wektorów TF-IDF też działa. Współczynnik tłumienia 0.85 i liczba iteracji to domyślne PageRank.

### Krok 2: abstrakcyjny z BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN jest dostrojony na korpusie CNN/DailyMail. Produkuje streszczenia w stylu newsowym od razu. Dla innych domen (artykuły naukowe, dialogi, prawne), użyj odpowiedniego checkpointu Pegasus lub dostrój na swoich danych docelowych.

### Krok 3: ewaluacja ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Zawsze używaj stemmingu. Bez niego "running" i "run" liczą się jako różne słowa, a ROUGE niedolicza.

### Poza ROUGE (ewaluacja streszczania 2026)

ROUGE było dominującą metryką streszczania przez dwadzieścia lat i jest niewystarczające samo w sobie w 2026. Meta-analiza dużej skali artykułów NLG pokazała:

- **BERTScore** (podobieństwo osadzeń kontekstowych) zyskało popularność do 2023 i jest teraz raportowane obok ROUGE w większości artykułów o streszczaniu.
- **BARTScore** traktuje ewaluację jako generację: oceń streszczenie przez to, jak prawdopodobne jest, że wstępnie wytrenowany BART przypisze je do źródła.
- **MoverScore** (Earth Mover's Distance na osadzeniach kontekstowych) osiągnął najwyższą pozycję w benchmarkach streszczania w 2025, ponieważ lepiej uchwytuje nakładanie semantyczne niż ROUGE.
- **FactCC** i **wierność oparta na QA** były powszechne w latach 2021-2023, teraz często zastąpione przez **G-Eval** (łańcuch promptów GPT-4, który punktuje spójność, konsystencję, płynność, trafność z rozumowaniem łańcucha myśli).
- **G-Eval** i podobne podejścia LLM-jako-sędzia osiągają zgodność z ludzką oceną ~80% czasu, gdy rubryki są dobrze zaprojektowane.

Zalecenie produkcyjne: raportuj ROUGE-L dla porównań z legacy, BERTScore dla nakładania semantycznego, G-Eval dla spójności i faktyczności. Kalibruj na 50-100 ręcznie oznakowanych streszczeniach.

### Krok 4: problem faktyczności

Streszczenia abstrakcyjne są podatne na halucynacje. Streszczenia ekstrakcyjne niosą znacznie niższe ryzyko halucynacji, ponieważ output jest dosłownie zaczerpnięty ze źródła, choć nadal mogą wprowadzać w błąd, jeśli zdania źródłowe są wyjęte z kontekstu, nieaktualne lub zacytowane w złej kolejności. To największy powód, dla którego systemy produkcyjne wciąż preferują metody ekstrakcyjne dla treści związanych ze zgodnością.

Typy halucynacji do nazwania:

- **Zamiana encji.** Źródło mówi "John Smith." Streszczenie mówi "John Brown."
- **Dryf liczbowy.** Źródło mówi "25,000." Streszczenie mówi "25 million."
- **Odwrócenie polaryzacji.** Źródło mówi "rejected the offer." Streszczenie mówi "accepted the offer."
- **Wymyślenie faktu.** Źródło nie wspomina CEO. Streszczenie mówi, że CEO zatwierdził.

Podejścia ewaluacyjne, które działają:

- **FactCC.** Klasyfikator binarny trenowany na wynikaniu między zdaniem źródłowym a zdaniem streszczenia. Przewiduje faktyczne/niefaktyczne.
- **Wierność oparta na QA.** Zadaj modelowi QA pytania, na które odpowiedzi są w źródle. Jeśli streszczenie obsługuje inne odpowiedzi, zgłoś.
- **F1 na poziomie encji.** Porównaj nazwane encje w źródle vs. streszczeniu. Encje obecne tylko w streszczeniu są podejrzane.

Do czegokolwiek skierowanego do użytkownika, gdzie faktyczność ma znaczenie (newsy, medycyna, prawo, finanse), ekstrakcyjne jest bezpieczniejszym domyślnym. Abstrakcyjne potrzebuje kontroli faktyczności w pętli.

## Użyj tego

Stos na 2026:

| Zastosowanie | Zalecane |
|---------|-------------|
| Newsy, 3-5-zdaniowe streszczenie, angielski | `facebook/bart-large-cnn` |
| Artykuły naukowe | `google/pegasus-pubmed` lub dostrojony T5 |
| Wielodokumentowe, długie formy | Dowolny LLM z kontekstem 32k+, promptowany |
| Streszczanie dialogów | `philschmid/bart-large-cnn-samsum` |
| Ekstrakcyjne, niskie ryzyko halucynacji z założenia | TextRank lub LSA / LexRank z `sumy` |

LLM z długim kontekstem często biją wyspecjalizowane modele w 2026, gdy moc obliczeniowa nie jest ograniczeniem. Kompromisem jest koszt i powtarzalność; wyspecjalizowane modele dają bardziej spójne outputy.

## Dostarcz to

Zapisz jako `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Ćwiczenia

1. **Łatwe.** Uruchom TextRank na 5 artykułach newsowych. Porównaj top-3 zdania z referencyjnym streszczeniem. Zmierz ROUGE-L. Powinieneś zobaczyć 30-45 ROUGE-L na artykułach w stylu CNN/DailyMail.
2. **Średnie.** Zaimplementuj faktyczność na poziomie encji: wydobądź nazwane encje ze źródła i streszczenia (spaCy), oblicz czułość encji źródłowych w streszczeniu i precyzję encji streszczenia względem źródła. Wysoka precyzja i niska czułość oznaczają bezpieczne, ale zwięzłe; niska precyzja oznacza halucynacje encji.
3. **Trudne.** Porównaj BART-large-CNN z LLM (Claude lub GPT-4) na 50 artykułach CNN/DailyMail. Raportuj ROUGE-L, faktyczność (przez F1 encji) i koszt na streszczenie. Udokumentuj, gdzie każdy wygrywa.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Ekstrakcyjne | Wybierz zdania | Zwróć zdania dosłownie ze źródła. Nigdy nie halucynuje. |
| Abstrakcyjne | Przepisz | Wygeneruj nowy tekst warunkowany źródłem. Może halucynować. |
| ROUGE | Metryka streszczeń | Nakładanie n-gramów / NJP między wyjściem systemu a referencją. |
| TextRank | Grafowy ekstrakcyjny | PageRank na grafie podobieństwa zdań. |
| Faktyczność | Czy jest poprawne | Czy twierdzenia streszczenia są poparte przez źródło. |
| Halucynacja | Wymyślone treści | Treści w streszczeniu, których źródło nie obsługuje. |

## Dalsze czytanie

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — kanoniczny artykuł ekstrakcyjny.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — artykuł BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus i obiektyw gap-sentence.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — artykuł ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — artykuł o krajobrazie faktyczności.