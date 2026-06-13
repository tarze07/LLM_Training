# Worek Słów, TF-IDF i Reprezentacja Tekstu

> Najpierw policz, potem myśl. TF-IDF wciąż bije embeddingi w dobrze zdefiniowanych zadaniach w 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## Problem

Model potrzebuje liczb. Masz ciągi znaków.

Każdy potok NLP musi odpowiedzieć na to samo pytanie. Jak zamienić strumień tokenów o zmiennej długości na wektor o stałym rozmiarze, który klasyfikator może konsumować. Pierwszą odpowiedzią, na którą wpadło pole, była najgłupsza, która działa. Policz słowa. Zrób wektor.

Ten wektor niósł więcej produkcyjnego NLP niż jakikolwiek model embeddingu. Filtry spamu, klasyfikatory tematów, wykrywanie anomalii w logach, ranking wyszukiwania (przed BM25), pierwsza fala analizy sentymentu, pierwsza dekada akademickich benchmarków NLP. Praktycy w 2026 wciąż sięgają po niego najpierw w wąskich zadaniach klasyfikacyjnych. Jest szybki, interpretowalny i często nie do odróżnienia od modelu embeddingu z 400M parametrami w zadaniach, gdzie obecność słowa jest tym, co się liczy.

Ta lekcja buduje worek słów, a następnie TF-IDF, od zera. Potem pokazuje scikit-learn robiące to samo w trzech liniach. Na koniec wymienia sposób na porażkę, który sprawia, że sięgasz po embeddingi.

## Koncepcja

**Worek Słów (BoW)** wyrzuca kolejność. Dla każdego dokumentu policz, ile razy każde słowo ze słownika występuje. Długość wektora to rozmiar słownika. Pozycja `i` to liczba wystąpień słowa `i`.

**TF-IDF** przeważa BoW. Słowo, które występuje w każdym dokumencie, jest nieinformacyjne, więc je skaluj w dół. Słowo rzadkie w całym korpusie, ale częste w pojedynczym dokumencie, jest sygnałem, więc skaluj je w górę.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Gdzie `TF` to częstotliwość terminu w dokumencie, `df` to częstotliwość dokumentowa (ile dokumentów zawiera to słowo), `N` to całkowita liczba dokumentów. `log` utrzymuje wagę w ryzach dla wszechobecnych słów.

Kluczowa właściwość: oba produkują rzadkie wektory z interpretowalnymi osiami. Możesz spojrzeć na wagi wytrenowanego klasyfikatora i odczytać, które słowa popychają dokument w kierunku każdej klasy. Nie możesz tego zrobić z 768-wymiarowym embeddingiem BERTa.

```figure
bow-tfidf
```

## Zbuduj To

### Krok 1: zbuduj słownik

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Wejście: lista tokenizowanych dokumentów (dowolny tokenizer słów działa; `code/main.py` w tej lekcji używa uproszczonego wariantu z małymi literami). Wyjście: słownik `{word: index}`. Stabilna kolejność wstawiania oznacza, że indeks słowa 0 to pierwsze słowo widziane w pierwszym dokumencie. Konwencje są różne; scikit-learn sortuje alfabetycznie.

### Krok 2: worek słów

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Wiersze to dokumenty. Kolumny to indeksy słownika. Wpis `[i][j]` to "ile razy słowo `j` pojawia się w dokumencie `i`." Dokument 1 ma `cat` dwa razy, bo tak było. Dokument 0 ma `ran` zero razy, bo go nie było.

### Krok 3: częstotliwość terminu i częstotliwość dokumentowa

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Dwa triki wygładzania warte nazwania. `(n+1)/(d+1)` unika `log(x/0)`. Końcowe `+1` zapewnia, że słowo w każdym dokumencie wciąż ma IDF równe 1 (nie 0), zgodne z domyślnym scikit-learn. Inne implementacje używają surowego `log(N/df).` Oba działają; wygładzona wersja jest przyjaźniejsza.

### Krok 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Trzy dokumenty, pięć słów w słowniku (`the`, `cat`, `sat`, `dog`, `ran`). `the` pojawia się we wszystkich trzech, więc jego IDF jest niskie. `dog` pojawia się w jednym, więc jego IDF jest wysokie. Wektory są rzadkie (większość wpisów jest mała), a dyskryminujące słowa się wyróżniają.

### Krok 5: Normalizacja L2 wierszy

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Bez normalizacji dłuższy dokument ma większy wektor i dominuje wyniki podobieństwa. Normalizacja L2 umieszcza każdy dokumen na jednostkowej hipersferze. Podobieństwo cosinusowe między wierszami jest teraz tylko iloczynem skalarnym.

## Użyj Tego

scikit-learn dostarcza wersję produkcyjną.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` wykonuje tokenizację, słownik i BoW w jednym wywołaniu. `TfidfVectorizer` dodaje ważenie IDF i normalizację L2. Oba zwracają rzadkie macierze. Dla 100k dokumentów gęsta wersja nie mieści się w pamięci; pozostań przy rzadkiej, dopóki klasyfikator nie zażąda gęstej.

Pokrętła, które zmieniają wszystko:

| Arg | Efekt |
|-----|--------|
| `ngram_range=(1, 2)` | Dołącz bigramy. Zwykle poprawia klasyfikację. |
| `min_df=2` | Odrzuć słowa w mniej niż 2 dokumentach. Przycina słownik na zaszumionych danych. |
| `max_df=0.95` | Odrzuć słowa w więcej niż 95% dokumentów. Przybliża usuwanie stopwords bez zakodowanej listy. |
| `stop_words="english"` | Wbudowana lista stopwords scikit-learn. Zależne od zadania — analiza sentymentu nie powinna usuwać negacji. |
| `sublinear_tf=True` | Użyj `1 + log(tf)` zamiast surowego `tf`. Pomaga, gdy termin powtarza się wiele razy w jednym dokumencie. |

### Kiedy TF-IDF wciąż wygrywa (stan na 2026)

- Wykrywanie spamu, etykietowanie tematów, flagowanie anomalii w logach. Obecność słowa jest tym, co się liczy; niuanse semantyczne nie.
- Reżimy niskiej ilości danych (setki oznakowanych przykładów). TF-IDF plus regresja logistyczna nie ma kosztu pretreningu.
- Wszędzie, gdzie liczy się opóźnienie. TF-IDF plus model liniowy odpowiada w mikrosekundach. Osadzenie dokumentu przez transformer zajmuje 10-100ms.
- Systemy, które muszą wyjaśniać swoje przewidywania. Sprawdź współczynniki klasyfikatora. Najwyższe pozytywne słowa są powodem.

### Kiedy TF-IDF zawodzi

Porażka ślepoty semantycznej. Rozważ te dwa dokumenty:

- "The movie was not good at all."
- "The movie was excellent."

Jeden to negatywna recenzja. Drugi jest pozytywna. Ich nakładanie się TF-IDF to dokładnie `{the, movie, was}`. Klasyfikator worka słów musi zapamiętać, że słowo `not` obok `good` odwraca etykietę. Może się tego nauczyć na wystarczającej ilości danych, ale nigdy tak elegancko, jak model, który rozumie składnię.

Inna porażka: słowa spoza słownika podczas inferencji. Model BoW wytrenowany na recenzjach IMDb nie ma pojęcia, co zrobić z `Zoomer-approved`, jeśli ten token nigdy nie pojawił się w treningu. Embeddingi podsłów (lekcja 04) sobie z tym radzą. TF-IDF nie może.

### Hybryda: embeddingi ważone TF-IDF

Praktyczna domyślna opcja 2026 dla średniej ilości danych w klasyfikacji: użyj wag TF-IDF jako uwagi nad embeddingami słów.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Zyskujesz pojemność semantyczną z embeddingów i nacisk na rzadkie słowa z TF-IDF. Klasyfikator trenuje na połączonym wektorze. To przewyższa każdą z tych metod osobno dla klasyfikacji sentymentu, tematu i intencji poniżej około 50k oznakowanych przykładów.

## Dostarcz To

Zapisz jako `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `cosine_similarity(doc_vec_a, doc_vec_b)` na wyjściu TF-IDF znormalizowanym L2. Sprawdź, że identyczne dokumenty uzyskują 1.0, a dokumenty z rozłącznym słownikiem uzyskują 0.0.
2. **Średnie.** Dodaj obsługę `n-gram` do `bag_of_words`. Parametr `n` produkuje zliczenia dla `n`-gramów. Sprawdź, że `n=2` na `["the", "cat", "sat"]` produkuje zliczenia bigramów dla `["the cat", "cat sat"]`.
3. **Trudne.** Zbuduj hybrydę embeddingów ważonych TF-IDF powyżej, używając wektorów GloVe 100d (pobierz raz, cache'uj). Porównaj dokładność klasyfikacji z czystym TF-IDF i czystymi średnimi embeddingami na zbiorze 20 Newsgroups. Raportuj, który wygrywa gdzie.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| BoW | Wektor częstotliwości słów | Zliczenia słów ze słownika w jednym dokumencie. Wyrzuca kolejność. |
| TF | Częstotliwość terminu | Liczba wystąpień słowa w dokumencie, opcjonalnie znormalizowana przez długość dokumentu. |
| DF | Częstotliwość dokumentowa | Liczba dokumentów zawierających dane słowo co najmniej raz. |
| IDF | Odwrotna częstotliwość dokumentowa | `log(N / df)` wygładzone. Osłabia słowa występujące wszędzie. |
| Rzadki wektor | Głównie zera | Słownik ma zazwyczaj 10k-100k słów; większość jest nieobecna w danym dokumencie. |
| Podobieństwo cosinusowe | Kąt wektorów | Iloczyn skalarny wektorów znormalizowanych L2. 1 oznacza identyczne, 0 oznacza ortogonalne. |

## Dalsza Literatura

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — kanoniczna referencja API plus notatki o każdym pokrętle.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — praca, która uczyniła TF-IDF domyślnym na dekadę.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — spojrzenie z 2026 na to, kiedy stara metoda wygrywa i dlaczego.