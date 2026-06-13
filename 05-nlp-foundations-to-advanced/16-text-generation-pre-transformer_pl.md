# Generowanie Tekstu Przed Transformerami — N-gramowe Modele Językowe

> Jeśli słowo jest zaskakujące, model jest zły. Perplexity zamienia zaskoczenie w liczbę. Wygładzanie utrzymuje ją skończoną.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Problem

Przed transformerami, przed RNN, przed osadzeniami słów, model językowy przewidywał następne słowo, zliczając, jak często występowało ono po poprzednich `n-1` słowach. Zlicz "the cat" → "sat" 47 razy, "the cat" → "jumped" 12 razy, "the cat" → "refrigerator" 0 razy. Znormalizuj, aby otrzymać rozkład prawdopodobieństwa.

To jest n-gramowy model językowy. Napędzał każdy rozpoznawacz mowy, każdy sprawdzacz pisowni i każdy system tłumaczenia maszynowego opartego na frazach od 1980 do 2015. Wciąż działa, gdy potrzebujesz taniego modelowania języka na urządzeniu.

Interesującym problemem jest to, co zrobić z niewidzianymi n-gramami. Surowy model oparty na zliczeniach przypisuje zerowe prawdopodobieństwo wszystkiemu, czego nie widział, co jest katastrofalne, ponieważ zdania są długie i prawie każde długie zdanie zawiera co najmniej jedną niewidzianą sekwencję. Pięćdziesiąt lat badań nad wygładzaniem naprawiło to. Wygładzanie Knesera-Neya jest wynikiem, a współczesne głębokie uczenie odziedziczyło jego empiryczną tradycję.

## Koncepcja

![Model n-gramowy: zlicz, wygładź, generuj](../assets/ngram.svg)

**Prawdopodobieństwo n-gramowe:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Ustal `n` (zazwyczaj 3 dla trigramów, 4 dla 4-gramów). Oblicz ze zliczeń:

```text
P(w | context) = count(context, w) / count(context)
```

**Problem zerowych zliczeń.** Każdy n-gram nie widziany w treningu dostaje zerowe prawdopodobieństwo. Badanie z 2007 roku na korpusie Browna wykazało, że nawet model 4-gramowy miał 30% wstrzymanych 4-gramów niewidzianych w treningu. Nie można oceniać na żadnym prawdziwym tekście bez wygładzania.

**Podejścia do wygładzania, w kolejności zaawansowania:**

1. **Laplace (add-one).** Dodaj 1 do każdego zliczenia. Proste, okropne dla rzadkich zdarzeń.
2. **Good-Turing.** Realokuj masę prawdopodobieństwa ze zdarzeń o wyższej częstotliwości do niewidzianych na podstawie częstotliwości-częstotliwości.
3. **Interpolacja.** Połącz estymaty n-gramowe, (n-1)-gramowe itd. z regulowanymi wagami.
4. **Backoff.** Jeśli n-gram ma zerowe zliczenie, spadnij do (n-1)-gramu. Katz backoff normalizuje to.
5. **Absolute discounting.** Odejmij stały dyskont `D` od wszystkich zliczeń, rozdziel do niewidzianych.
6. **Kneser-Ney.** Absolute discounting plus sprytny wybór dla modelu niższego rzędu: użyj *prawdopodobieństwa kontynuacji* (w ilu kontekstach pojawia się słowo) zamiast surowej częstotliwości.

Wgląd Knesera-Neya jest głęboki. "San Francisco" to powszechny bigram. Unigram "Francisco" pojawia się głównie po "San." Naiwne absolute discounting daje "Francisco" wysokie prawdopodobieństwo unigramowe (ponieważ zliczenie jest wysokie). Kneser-Ney zauważa, że "Francisco" pojawia się tylko w jednym kontekście i obniża jego prawdopodobieństwo kontynuacji. Wynik: nowy bigram kończący się na "Francisco" dostaje odpowiednio niskie prawdopodobieństwo.

**Ewaluacja: perplexity.** Wykładnik średniego ujemnego logarytmu wiarygodności na słowo na wstrzymanym zestawie testowym. Niższe jest lepsze. Perplexity równe 100 oznacza, że model jest tak samo zdezorientowany, jak przy równomiernym wyborze spośród 100 słów.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Zbuduj To

### Krok 1: zliczenia trigramów

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Wejście to lista tokenizowanych zdań. Wyjście to zliczenia n-gramów i zliczenia kontekstów. `<s>` i `</s>` to granice zdań.

### Krok 2: wygładzanie Laplace

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Dodaj 1 do każdego zliczenia. Wygładza, ale nadmiernie alokuje masę do niewidzianych zdarzeń, szkodząc również rzadkim znanym zdarzeniom.

### Krok 3: Kneser-Ney (bigram, interpolowany)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Trzy ruchome części. `continuation_prob` wychwytuje "w ilu różnych kontekstach pojawia się to słowo?" (innowacja Knesera-Neya). `lambda_prev` to masa uwolniona przez dyskont, używana do ważenia backoffu. Ostateczne prawdopodobieństwo to zdyskontowany człon główny plus ważony człon kontynuacji.

### Krok 4: generowanie tekstu z próbkowaniem

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Próbkowanie proporcjonalne do prawdopodobieństwa. Zawsze daje inny wynik na ziarno. Dla wyniku podobnego do beam-search, wybierz argmax na każdym kroku (zachłannie) i dodaj mały pokrętło losowości (temperatura).

### Krok 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Niższe jest lepsze. Dla korpusu Browna, dobrze dostrojony model 4-gramowy KN osiąga perplexity około 140. Transformer LM osiąga 15-30 na tym samym zestawie testowym. Różnica wynosi około 10x. Ta różnica jest powodem, dla którego dziedzina poszła dalej.

## Użyj Tego

- **Nauczanie klasycznego NLP.** Najczystsza ekspozycja na wygładzanie, MLE i perplexity, jaką możesz dostać.
- **KenLM.** Produkcyjna biblioteka n-gramowa. Używana jako rescorer w systemach mowy i MT, gdzie niskie opóźnienie ma znaczenie.
- **Autouzupełnianie na urządzeniu.** Modele trigramowe w klawiaturach. Wciąż.
- **Baseline.** Zawsze oblicz perplexity n-gramowego LM, zanim uznasz swój neuronowy LM za dobry. Jeśli twój transformer nie bije KN z dużym marginesem, coś jest nie tak.

## Wdróż To

Zapisz jako `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Ćwiczenia

1. **Łatwe.** Wytrenuj trigramowy LM na 1000-zdaniowym korpusie Szekspira. Wygeneruj 20 zdań. Będą lokalnie prawdopodobne, ale globalnie niespójne. To jest kanoniczne demo.
2. **Średnie.** Zaimplementuj perplexity dla swojego modelu KN na wstrzymanym podziale Szekspira. Porównaj z Laplace. Powinieneś zobaczyć KN z perplexity niższym o 30-50%.
3. **Trudne.** Zbuduj trigramowy korektor pisowni: mając błędnie napisane słowo i jego kontekst, generuj poprawki i ranking według prawdopodobieństwa kontekstu pod LM. Oceń na korpusie pisowni Birkbeck (publiczny).

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| N-gram | Sekwencja słów | Sekwencja `n` kolejnych tokenów. |
| Wygładzanie | Unikanie zer | Realokacja masy prawdopodobieństwa, aby niewidziane zdarzenia dostały niezerowe prawdopodobieństwo. |
| Perplexity | Metryka jakości LM | `exp(-średni log-prob)` na wstrzymanych danych. Niższe jest lepsze. |
| Backoff | Powrót do krótszego kontekstu | Jeśli zliczenie trigramu jest zerowe, użyj bigramu. Katz backoff formalizuje to. |
| Kneser-Ney | Najlepsze wygładzanie dla n-gramów | Absolute discounting + prawdopodobieństwo kontynuacji dla modelu niższego rzędu. |
| Prawdopodobieństwo kontynuacji | Specyficzne dla KN | `P(w)` ważone liczbą kontekstów, w których pojawia się `w`, a nie surowym zliczeniem. |

## Dalsza Lektura

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — kanoniczne opracowanie n-gramowych LM i wygładzania.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — artykuł, który ustalił Kneser-Ney jako najlepsze wygładzanie n-gramowe.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — oryginalny artykuł KN.
- [KenLM](https://kheafield.com/code/kenlm/) — szybki produkcyjny n-gramowy LM, wciąż używany w 2026 roku do aplikacji wrażliwych na opóźnienia.