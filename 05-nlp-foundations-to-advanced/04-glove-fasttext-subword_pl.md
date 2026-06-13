# GloVe, FastText i Embeddingi Podsłów

> Word2Vec trenował jeden embedding na słowo. GloVe faktoryzowało macierz współwystępowania. FastText osadzało kawałki. BPE zbudowało most do transformerów.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## Problem

Word2Vec pozostawił dwa otwarte pytania.

Po pierwsze, istniała równoległa linia badań, która faktoryzowała macierz współwystępowania bezpośrednio (LSA, HAL), zamiast wykonywać iteracyjne aktualizacje skip-gram. Czy iteracyjne podejście Word2Vec było zasadniczo lepsze, czy różnica była artefaktem sposobu, w jaki obie metody radziły sobie zliczeniami? **GloVe** odpowiedziało: faktoryzacja macierzy z przemyślaną funkcją straty dorównuje lub bije Word2Vec i kosztuje mniej w treningu.

Po drugie, żadna metoda nie miała historii dla słów, których nigdy nie widziała. `Zoomer-approved`, `dogecoin`, każda nazwa własna ukuta w zeszłym tygodniu, każda odmieniona forma rzadkiego rdzenia. **FastText** naprawił to, osadzając n-gramy znaków: słowo jest sumą swoich części, włączając morfemy, więc nawet słowa spoza słownika otrzymują sensowny wektor.

Po trzecie, gdy pojawiły się transformery, pytanie znów się przesunęło. Słowniki na poziomie słów osiągają limit około miliona wpisów; prawdziwy język jest bardziej otwarty. **Kodowanie par bajtów (BPE)** i jego krewni rozwiązali to, ucząc się słownika częstych jednostek podsłów, który obejmuje wszystko. Każdy nowoczesny tokenizer każdego nowoczesnego LLM to tokenizer podsłów.

Ta lekcja omawia wszystkie trzy, a następnie wyjaśnia, po które sięgać w jakiej sytuacji.

## Koncepcja

**GloVe (Globalne Wektory).** Zbuduj macierz współwystępowania słowo-słowo `X`, gdzie `X[i][j]` mówi, jak często słowo `j` pojawia się w kontekście słowa `i`. Trenuj wektory tak, aby `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Zważ stratę, aby częste pary nie dominowały. Gotowe.

**FastText.** Słowo jest sumą swoich n-gramów znaków plus samego słowa. `where` staje się `<wh, whe, her, ere, re>, <where>`. Wektor słowa jest sumą tych wektorów składowych. Trenuj jak Word2Vec. Korzyść: niewidziane słowa (`whereupon`) komponują się ze znanych n-gramów.

**BPE (Kodowanie Par Bajtów).** Zacznij od słownika pojedynczych bajtów (lub znaków). Policz każdą sąsiednią parę w korpusie. Połącz najczęstszą parę w nowy token. Powtórz dla `k` iteracji. Wynik: słownik `k + 256` tokenów, w którym częste sekwencje (`ing`, `tion`, `the`) są pojedynczymi tokenami, a rzadkie słowa są dzielone na znajome kawałki. Każde zdanie tokenizuje się na coś.

## Zbuduj To

### GloVe: faktoryzacja macierzy współwystępowania

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Dwa ruchome elementy warte nazwania. Funkcja ważenia `f(x) = (x/x_max)^alpha` osłabia bardzo częste pary (jak `(the, and)`), aby nie dominowały straty. Ostateczny embedding jest sumą tablic `W` (środek) i `W_tilde` (kontekst). Sumowanie obu to opublikowana sztuczka, która zwykle działa lepiej niż użycie tylko jednej.

### FastText: embeddingi świadome podsłów

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Każde słowo jest reprezentowane przez swój zbiór n-gramów (zwykle 3 do 6 znaków). Embedding słowa jest sumą embeddingów jego n-gramów. Do treningu skip-gram, podłącz to tam, gdzie Word2Vec używał pojedynczego wektora.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Dla niewidzianego słowa wciąż otrzymujesz wektor, o ile niektóre z jego n-gramów są znane. `whereupon` dzieli `<wh`, `her`, `ere` i `<where` z `where`, więc dwa lądują blisko siebie.

### BPE: nauczony słownik podsłów

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Pierwsza iteracja łączy najczęstszą sąsiednią parę. Po wystarczającej liczbie iteracji częste podciągi (`low`, `est`, `tion`) stają się pojedynczymi tokenami, a rzadkie słowa dzielą się czysto.

Prawdziwe tokenizery GPT / BERT / T5 uczą się 30k-100k połączeń. Wynik: każdy tekst tokenizuje się do ograniczonej długości sekwencji znanych ID, bez żadnych OOV.

## Użyj Tego

W praktyce rzadko trenujesz którekolwiek z nich samodzielnie. Ładujesz wstępnie wytrenowane punkty kontrolne.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Do tokenizacji podsłów w stylu BPE w erze transformerów:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Prefiks `Ġ` oznacza granice słów (konwencja GPT-2). Każdy nowoczesny tokenizer jest wariantem BPE, WordPiece (BERT) lub SentencePiece (T5, LLaMA).

### Kiedy który wybrać

| Sytuacja | Wybierz |
|-----------|--------|
| Wstępnie wytrenowane ogólne wektory słów, brak potrzeby tolerancji OOV | GloVe 300d |
| Wstępnie wytrenowane ogólne wektory słów, musi obsługiwać literówki / neologizmy / języki bogate morfologicznie | FastText |
| Cokolwiek idącego do transformera (trening lub inferencja) | Tokenizer, z którym model został dostarczony. Nigdy nie zamieniaj. |
| Trenowanie własnego modelu językowego od zera | Wytrenuj najpierw tokenizer BPE lub SentencePiece na swoim korpusie |
| Produkcyjna klasyfikacja tekstu z modelem liniowym | Wciąż TF-IDF. Lekcja 02. |

## Dostarcz To

Zapisz jako `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Ćwiczenia

1. **Łatwe.** Uruchom `char_ngrams("playing")` i `char_ngrams("played")`. Oblicz przecięcie Jaccarda dwóch zbiorów n-gramów. Powinieneś zobaczyć znaczące wspólne części (`pla`, `lay`, `play`), dlatego FastText dobrze przenosi się między wariantami morfologicznymi.
2. **Średnie.** Rozszerz `learn_bpe`, aby śledzić wzrost słownika. Wykreśl tokeny-na-znak-korpusu jako funkcję liczby połączeń. Powinieneś zobaczyć szybką kompresję na początku, asymptotycznie zbliżającą się do ~2-3 znaków na token.
3. **Trudne.** Wytrenuj BPE z 1k połączeń na kompletnym dziele Szekspira. Porównaj tokenizację popularnych słów vs. rzadkich nazw własnych. Zmierz średnią liczbę tokenów na słowo przed i po. Opisz, co cię zaskoczyło.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| Macierz współwystępowania | Tabela częstości słowo-słowo | `X[i][j]` = jak często słowo `j` pojawia się w oknie wokół słowa `i`. |
| Podsłowo | Kawałek słowa | n-gram znaków (FastText) lub nauczony token (BPE/WordPiece/SentencePiece). |
| BPE | Kodowanie par bajtów | Iteracyjne łączenie najczęstszych sąsiednich par, aż słownik osiągnie docelowy rozmiar. |
| OOV | Poza słownikiem | Słowo, którego model nigdy nie widział. Word2Vec/GloVe zawodzą. FastText i BPE sobie radzą. |
| BPE na poziomie bajtów | BPE na surowych bajtach | Schemat GPT-2. Słownik zaczyna się od 256 bajtów, więc nic nigdy nie jest OOV. |

## Dalsza Literatura

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — praca GloVe, siedem stron, wciąż najlepsze wyprowadzenie straty.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — praca, która wprowadziła BPE do nowoczesnego NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — jak BPE, WordPiece i SentencePiece faktycznie różnią się w praktyce.