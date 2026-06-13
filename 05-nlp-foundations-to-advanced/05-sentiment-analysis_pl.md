# Analiza Sentymentu

> Kanoniczne zadanie NLP. Większość tego, co musisz wiedzieć o klasycznej klasyfikacji tekstu, pojawia się tutaj.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## Problem

"The food was not great." Pozytywny czy negatywny?

Sentyment brzmi prosto. Recenzent powiedział, że coś mu się podobało lub nie. Oznakuj zdanie. Powodem, dla którego stał się kanonicznym zadaniem NLP, jest to, że każdy łatwo wyglądający przypadek kryje trudny. Negacja odwraca znaczenie. Sarkazm je odwraca. "Not bad at all" jest pozytywne mimo dwóch negatywnie zakodowanych słów. Emoji niosą więcej sygnału niż otaczający tekst. Słownictwo dziedzinowe ma znaczenie (`tight` w recenzji muzycznej kontra `tight` w recenzji mody).

Sentyment jest laboratorium roboczym klasycznego NLP. Jeśli rozumiesz, dlaczego każdy naiwny baseline ma określony sposób na porażkę, rozumiesz, dlaczego wynaleziono każdy bogatszy model. Ta lekcja buduje baseline Naive Bayes od zera, dodaje regresję logistyczną i wymienia pułapki, które sprawiają, że produkcyjny sentyment jest problemem na poziomie zgodności.

## Koncepcja

Klasyczny sentyment to przepis dwuetapowy.

1. **Reprezentuj.** Zamień tekst na wektor cech. BoW, TF-IDF lub n-gramy.
2. **Klasyfikuj.** Dopasuj model liniowy (Naive Bayes, regresja logistyczna, SVM) na oznakowanych przykładach.

Naive Bayes to najgłupszy model, który działa. Załóż, że każda cecha jest niezależna przy danej etykiecie. Oszacuj `P(word | positive)` i `P(word | negative)` ze zliczeń. Przy inferencji pomnóż prawdopodobieństwa. Założenie "naiwnej" niezależności jest śmiesznie błędne, a jednak wyniki są zaskakująco silne. Powód: przy rzadkich cechach tekstowych i umiarkowanej ilości danych klasyfikatorowi zależy bardziej na to, na którą stronę każde słowo przechyla niż jak bardzo.

Regresja logistyczna naprawia założenie niezależności. Uczy się wagi na cechę, w tym wag negatywnych. `not good` jako cecha bigramu otrzymuje wagę ujemną. Naive Bayes nie może tego zrobić dla bigramów, których nigdy nie oznaczył.

```figure
sentiment-logits
```

## Zbuduj To

### Krok 1: prawdziwy mini-zbiór danych

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Celowo mały. Prawdziwa praca używa dziesiątek tysięcy przykładów (IMDb, SST-2, Yelp polarity). Matematyka jest identyczna.

### Krok 2: wielomianowy Naive Bayes od zera

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Wygładzanie addytywne (alpha=1.0) to wygładzanie Laplace'a. Bez niego słowo niewidziane w klasie ma prawdopodobieństwo zero, a log wybucha. `alpha=0.01` jest powszechne w praktyce. `alpha=1.0` to domyślna wartość dydaktyczna.

### Krok 3: regresja logistyczna od zera

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Regularyzacja L2 ma tutaj znaczenie. Cechy tekstowe są rzadkie; bez L2 model zapamiętuje przykłady treningowe. Zacznij od `0.01` i dostrajaj.

### Krok 4: obsługa negacji (sposób na porażkę)

Rozważ "not good" i "not bad". Klasyfikator BoW widzi `{not, good}` i `{not, bad}` i uczy się z tego, co pojawiło się częściej w treningu. Klasyfikator bigramowy widzi `not_good` i `not_bad` i uczy się ich jako odrębnych cech. To zwykle wystarcza.

Bardziej prymitywna poprawka, która działa, gdy nie masz bigramów: **zakres negacji**. Prefiksuj tokeny następujące po słowie negacji za pomocą `NOT_` aż do następnej interpunkcji.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Teraz `good` i `NOT_good` to różne cechy. Klasyfikator może je ważyć przeciwnie. Trzy linie preprocessingu, mierzalny wzrost dokładności na benchmarkach sentymentu.

### Krok 5: metryki ewaluacji, które mają znaczenie

Sama dokładność jest myląca, jeśli klasy są niezrównoważone. Prawdziwe korpusy sentymentu są zwykle 70-80% pozytywne lub 70-80% negatywne; stały klasyfikator większościowy osiąga 80% dokładności i jest bezużyteczny. Raportuj każde z poniższych:

- **Precyzja i recall na klasę.** Jedna para na klasę. Uśrednij je makro, aby uzyskać pojedynczą liczbę respektującą balans klas.
- **Macro-F1 (podstawowa metryka dla niezrównoważonych danych).** Średnia wyników F1 na klasę, równo ważona. Użyj tego zamiast dokładności, gdy klasy są niezrównoważone.
- **Weighted-F1 (alternatywa).** To samo co macro, ale ważone częstotliwością klas. Raportuj obok macro-F1, gdy sam brak równowagi ma znaczenie biznesowe.
- **Macierz pomyłek.** Surowe liczby. Zawsze sprawdzaj przed zaufaniem jakiejkolwiek skalarnej metryce; ujawnia, którą parę klas model myli.
- **Próbki błędów na klasę.** Wyciągnij 5 błędnych przewidywań na klasę. Przeczytaj je. Nic nie zastępuje czytania rzeczywistych błędów.

Dla silnie niezrównoważonych danych (> stosunek 95-5) raportuj **AUROC** i **AUPRC** zamiast dokładności. AUPRC jest bardziej czułe na klasę mniejszościową, która zwykle jest tym, co cię interesuje (spam, oszustwa, rzadki sentyment).

**Częsty błąd do uniknięcia.** Raportowanie micro-F1 zamiast macro-F1 na niezrównoważonych danych daje liczbę, która wygląda wysoko, ponieważ jest zdominowana przez klasę większościową. Macro-F1 zmusza cię do zobaczenia wydajności klasy mniejszościowej.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Użyj Tego

scikit-learn robi to w sześciu liniach, poprawnie.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Trzy rzeczy do zauważenia. `stop_words=None` zachowuje negacje. `ngram_range=(1, 2)` dodaje bigramy, więc `not_good` staje się cechą. `sublinear_tf=True` tłumi powtarzające się słowa. Te trzy flagi to różnica między baselineem z 75% dokładnością a baselineem z 85% dokładnością na SST-2.

### Kiedy sięgnąć po transformer

- Wykrywanie sarkazmu. Klasyczne modele zawodzą tutaj. Kropka.
- Długie recenzje, w których sentyment zmienia się w trakcie dokumentu.
- Sentyment aspektowy. "Camera was great but battery was terrible." Musisz przypisać sentyment do aspektów. Tylko transformery lub modele ze strukturalnym wyjściem.
- Języki nieangielskie, o niskich zasobach. Wielojęzyczny BERT daje ci baseline zero-shot za darmo.

Jeśli potrzebujesz któregokolwiek z powyższych, przejdź do fazy 7 (głębokie zanurzenie w transformery). W przeciwnym razie Naive Bayes lub regresja logistyczna na TF-IDF plus bigramy plus obsługa negacji to twój produkcyjny baseline na 2026.

### Pułapka odtwarzalności (znowu)

Ponowne trenowanie modeli sentymentu jest rutynowe. Ponowna ewaluacja nie. Wyniki dokładności raportowane w pracach używają konkretnych podziałów, konkretnego preprocessingu, konkretnych tokenizerów. Jeśli porównujesz swój nowy model z baselineem bez używania identycznego potoku, otrzymasz mylące różnice. Zawsze regeneruj baseline na swoim potoku, a nie na liczbie z pracy.

## Dostarcz To

Zapisz jako `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Ćwiczenia

1. **Łatwe.** Dodaj `apply_negation` jako krok preprocessingu w potoku scikit-learn i zmierz różnicę F1 na małym zbiorze danych sentymentu.
2. **Średnie.** Zaimplementuj regresję logistyczną ważoną klasami (przekaż `class_weight="balanced"` do scikit-learn lub wyprowadź gradient samodzielnie). Zmierz wpływ na syntetycznym braku równowagi klas 90-10.
3. **Trudne.** Zbuduj detektor sarkazmu, trenując drugi klasyfikator na resztach modelu sentymentu. Udokumentuj swoją konfigurację eksperymentalną. Ostrzeż czytelnika, gdy twoja dokładność jest poniżej szansy (poziom szansy na 2-klasowym sarkazmie wynosi ~50%, a większość pierwszych prób tam ląduje).

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| Polaryzacja | Pozytywny lub negatywny | Etykieta binarna; czasem rozszerzona o neutralny lub drobnoziarnisty (5-gwiazdkowy). |
| Sentyment aspektowy | Polaryzacja na aspekt | Przypisz sentyment do konkretnych encji lub atrybutów wymienionych w tekście. |
| Zakres negacji | Odwracanie pobliskich tokenów | Prefiksuj tokeny po "not" za pomocą `NOT_` aż do interpunkcji. |
| Wygładzanie Laplace'a | Dodawanie 1 do zliczeń | Zapobiega cechom o zerowym prawdopodobieństwie w Naive Bayes. |
| Regularyzacja L2 | Kurczenie wag | Dodaje `lambda * sum(w^2)` do straty. Niezbędna dla rzadkich cech tekstowych. |

## Dalsza Literatura

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — fundamentalny przegląd. Długi, ale pierwsze cztery sekcje obejmują wszystko, co klasyczne.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — praca, która pokazała, że bigramy + Naive Bayes są trudne do pobicia na krótkim tekście.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — referencja dla `CountVectorizer`, `TfidfVectorizer` i każdego pokrętła, które będziesz dostrajać.