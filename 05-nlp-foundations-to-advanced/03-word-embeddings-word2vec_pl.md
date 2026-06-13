# Embeddingi Słów — Word2Vec od Zera

> Słowo towarzystwem swe. Wytrenuj płytką sieć na tym pomyśle, a geometria sama wypłynie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## Problem

TF-IDF wie, że `dog` i `puppy` to różne słowa. Nie wie, że znaczą prawie to samo. Klasyfikator wytrenowany na `dog` nie może uogólnić do recenzji o `puppy`. Można to zamaskować, wymieniając synonimy, ale to zawodzi na rzadkich terminach, żargonie dziedzinowym i każdym języku, którego nie przewidziałeś.

Chcesz reprezentacji, w której `dog` i `puppy` lądują blisko siebie w przestrzeni. Gdzie `king - man + woman` ląduje blisko `queen`. Gdzie model wytrenowany na `dog` przenosi część sygnału na `puppy` za darmo.

Word2Vec dał nam tę przestrzeń. Dwuwarstwowa sieć neuronowa, trenowana na bilionach tokenów, opublikowana w 2013. Architektura jest wręcz zawstydzająco prosta. Wyniki przekształciły NLP na dekadę.

## Koncepcja

**Hipoteza dystrybucyjna** (Firth, 1957): "Poznasz słowo po towarzystwie, w którym przebywa." Jeśli dwa słowa pojawiają się w podobnych kontekstach, prawdopodobnie znaczą podobne rzeczy.

Word2Vec występuje w dwóch wariantach, oba wykorzystujące ten pomysł.

- **Skip-gram.** Mając słowo środkowe, przewidź otaczające słowa. `cat -> (the, sat, on)` z rozmiarem okna 2.
- **CBOW (ciągły worek słów).** Mając otaczające słowa, przewidź środkowe. `(the, sat, on) -> cat`.

Skip-gram jest wolniejszy w treningu, ale lepiej radzi sobie z rzadkimi słowami. Stał się domyślnym.

Sieć ma jedną warstwę ukrytą bez nieliniowości. Wejście to wektor one-hot nad słownikiem. Wyjście to softmax nad słownikiem. Po treningu wyrzucasz warstwę wyjściową. Wagi warstwy ukrytej to embeddingi.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Sztuczka: softmax na 100k słowach jest niedostępnie kosztowny. Word2Vec używa **negatywnego próbkowania**, aby zamienić to w zadanie klasyfikacji binarnej. Przewiduj "czy to słowo kontekstowe pojawiło się obok tego słowa środkowego, tak czy nie". Próbkuj garść negatywnych (niewspółwystępujących) słów na parę treningową zamiast obliczać softmax na całym słowniku.

```figure
word-vector-arithmetic
```

## Zbuduj To

### Krok 1: pary treningowe z korpusu

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Każda para (środek, kontekst) w oknie jest pozytywnym przykładem treningowym.

### Krok 2: tablice embeddingów

Dwie macierze. `W` to tablica embeddingów słów środkowych (ta, którą zachowujesz). `W'` to tablica słów kontekstowych (często odrzucana, czasem uśredniana z `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Mała losowa inicjalizacja. Rozmiar słownika 10k i wymiar 100 jest realistyczny; do nauczania 50 słów x 16 wymiarów wystarczy, aby zobaczyć geometrię.

### Krok 3: cel negatywnego próbkowania

Dla każdej pozytywnej pary `(center, context)`, próbkuj `k` losowych słów ze słownika jako negatywne. Trenuj model tak, aby iloczyn skalarny `W[center] · W'[context]` był wysoki dla pozytywnych i niski dla negatywnych.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Magiczna formuła: strata logistyczna na parze pozytywnej (chcemy sigmoid blisko 1) plus strata logistyczna na parach negatywnych (chcemy sigmoid blisko 0). Gradienty płyną do obu tablic. Pełne wyprowadzenie jest w oryginalnej pracy; przejdź przez nie raz z papierem i długopisem, jeśli chcesz, aby utkwiło.

### Krok 4: trenuj na zabawkowym korpusie

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Po wystarczającej liczbie epok na dużym korpusie słowa dzielące konteksty mają podobne embeddingi środkowe. Na zabawkowym korpusie widzisz efekt słabo. Na miliardach tokenów widzisz go dramatycznie.

### Krok 5: sztuczka z analogią

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Na wstępnie wytrenowanych 300-wymiarowych wektorach Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Nie dlatego, że model wie, czym jest monarchia. Ponieważ wektor `(king - man)` uchwyca coś w rodzaju "królewskości", a dodanie go do `woman` ląduje w pobliżu regionu żeńsko-królewskiego.

## Użyj Tego

Pisanie Word2Vec od zera to nauczanie. Produkcyjne NLP używa `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Do prawdziwej pracy prawie nigdy nie trenujesz Word2Vec samodzielnie. Pobierasz wstępnie wytrenowane wektory.

- **GloVe** — podejście Stanforda oparte na faktoryzacji macierzy współwystępowania. Punkty kontrolne 50d, 100d, 200d, 300d. Dobry ogólny zasięg. Lekcja 04 obejmuje GloVe szczegółowo.
- **fastText** — rozszerzenie Word2Vec od Facebooka, które osadza n-gramy znaków. Radzi sobie ze słowami spoza słownika, komponując podsłowa. Lekcja 04.
- **Wstępnie wytrenowany Word2Vec na Google News** — 300d, słownik 3M słów, opublikowany 2013. Wciąż pobierany codziennie.

### Kiedy Word2Vec wciąż wygrywa w 2026

- Lekkie wyszukiwanie specyficzne dla domeny. Trenuj na abstraktach medycznych w godzinę na laptopie, uzyskaj wyspecjalizowane wektory, których żaden ogólny model nie uchwyci.
- Inżynieria cech w stylu analogii. `gender_vector = mean(man - woman pairs)`. Odejmij go od innych słów, aby uzyskać neutralną płciowo oś. Wciąż używane w badaniach nad sprawiedliwością.
- Interpretowalność. 100d jest wystarczająco małe, aby wykreślić przez PCA lub t-SNE i faktycznie zobaczyć tworzące się klastry.
- Wszędzie, gdzie inferencja musi działać na urządzeniu bez GPU. Odczyt Word2Vec to pojedyncze pobranie wiersza.

### Gdzie Word2Vec zawodzi

Ściana polisemii. `bank` ma jeden wektor. `river bank` i `financial bank` dzielą go. `table` (arkusz kalkulacyjny kontra mebel) dzieli go. Klasyfikator downstream nie może rozróżnić znaczeń z wektora.

Embeddingi kontekstowe (ELMo, BERT, każdy transformer od tamtego czasu) rozwiązały to, produkując inny wektor dla każdego wystąpienia słowa w oparciu o otaczający kontekst. To skok od Word2Vec do BERTa: od statycznego do kontekstowego. Faza 7 obejmuje połowę transformerową.

Problem słów spoza słownika to druga porażka. Word2Vec nigdy nie widział `Zoomer-approved`, jeśli nie było w danych treningowych. Brak mechanizmu awaryjnego. fastText naprawia to za pomocą kompozycji podsłów (lekcja 04).

## Dostarcz To

Zapisz jako `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Ćwiczenia

1. **Łatwe.** Uruchom pętlę treningową na małym korpusie (20 zdań o kotach i psach). Po 200 epokach sprawdź, czy `nearest(vocab, W, W[vocab["cat"]])` zwraca `dog` w swoich top 3. Jeśli nie, zwiększ epoki lub słownik.
2. **Średnie.** Dodaj podpróbkowanie częstych słów. Słowa z częstotliwością powyżej `10^-5` są odrzucane z par treningowych z prawdopodobieństwem proporcjonalnym do ich częstotliwości. Zmierz wpływ na podobieństwo rzadkich słów.
3. **Trudne.** Wytrenuj model na korpusie 20 Newsgroups. Oblicz dwie osie błędu: `he - she` i `doctor - nurse`. Rzutuj słowa zawodów na obie osie. Raportuj, które zawody mają największą lukę błędu. To rodzaj sondy, której używają badacze uczciwości.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| Embedding słowa | Słowo jako wektor | Gęsta, niskowymiarowa (zwykle 100-300) reprezentacja wyuczona z kontekstu. |
| Skip-gram | Sztuczka Word2Vec | Przewiduj słowa kontekstowe ze słowa środkowego. Wolniejszy niż CBOW, lepszy dla rzadkich słów. |
| Negatywne próbkowanie | Skrót treningowy | Zastąp softmax na całym słowniku klasyfikacją binarną względem `k` losowych słów. |
| Embedding statyczny | Jeden wektor na słowo | Ten sam wektor niezależnie od kontekstu. Zawodzi na polisemii. |
| Embedding kontekstowy | Wektor wrażliwy na kontekst | Inny wektor dla każdego wystąpienia w oparciu o otaczające słowa. To produkują transformery. |
| OOV | Poza słownikiem | Słowo niewidziane w treningu. Word2Vec nie może wyprodukować dla niego wektora. |

## Dalsza Literatura

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — praca o negatywnym próbkowaniu. Krótka i czytelna.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — najjaśniejsze wyprowadzenie gradientów, jeśli matematyka oryginalnej pracy wydaje się gęsta.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — ustawienia treningu produkcyjnego, które faktycznie działają.