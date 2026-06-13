# Naiwny Bayes

> Założenie "naiwne" jest błędne, a mimo to działa. W tym tkwi jego piękno.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Learning Objectives

- Zaimplementować wielomianowy naiwny Bayes od podstaw z wygładzaniem Laplace'a do klasyfikacji tekstu
- Wyjaśnić, dlaczego naiwne założenie o niezależności jest matematycznie błędne, ale w praktyce daje prawidłowe rankingi klas
- Porównać warianty wielomianowego, Bernoulliego i gaussowskiego naiwnego Bayesa oraz wybrać odpowiedni dla danego typu cech
- Ocenić naiwny Bayes w porównaniu z regresją logistyczną na wysokowymiarowych danych rzadkich i wyjaśnić działający kompromis błędu systematycznego i wariancji

## The Problem

Musisz klasyfikować tekst. E-maile na spam i nie-spam. Recenzje klientów na pozytywne i negatywne. Zgłoszenia pomocy technicznej na kategorie. Masz tysiące cech (jedna na słowo) i ograniczone dane treningowe.

Większość klasyfikatorów sobie z tym nie radzi. Regresja logistyczna potrzebuje wystarczająco dużo próbek, aby wiarygodnie oszacować tysiące wag. Drzewa decyzyjne dzielą na jedno słowo naraz i bardzo przeuczają się. KNN w 10 000 wymiarów jest bez znaczenia, ponieważ każdy punkt jest równie odległy od każdego innego punktu.

Naiwny Bayes radzi sobie z tym. Przyjmuje matematycznie błędne założenie (że każda cecha jest niezależna od każdej innej cechy przy danej klasie), a mimo to przewyższa "inteligentniejsze" modele w klasyfikacji tekstu, szczególnie przy małych zbiorach treningowych. Trenuje w jednym przebiegu przez dane. Skaluje się do milionów cech. Generuje oszacowania prawdopodobieństwa (choć często słabo skalibrowane z powodu założenia o niezależności).

Zrozumienie, dlaczego błędne założenie prowadzi do dobrych predykcji, uczy czegoś fundamentalnego o uczeniu maszynowym: najlepszy model to nie ten najbardziej poprawny, ale ten z najlepszym kompromisem błędu systematycznego i wariancji dla twoich danych.

## The Concept

### Twierdzenie Bayesa (szybkie przypomnienie)

Twierdzenie Bayesa odwraca prawdopodobieństwa warunkowe:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

Chcemy `P(class | features)` -- prawdopodobieństwo, że dokument należy do klasy, biorąc pod uwagę słowa w nim zawarte. Możemy to obliczyć z:
- `P(features | class)` -- wiarygodność zobaczenia tych słów w dokumentach tej klasy
- `P(class)` -- prawdopodobieństwo a priori klasy (jak powszechny jest spam ogólnie?)
- `P(features)` -- dowód, taki sam dla wszystkich klas, więc możemy go zignorować przy porównywaniu

Klasa z najwyższym `P(class | features)` wygrywa.

### Naiwne założenie o niezależności

Dokładne obliczenie `P(features | class)` wymaga oszacowania łącznego prawdopodobieństwa wszystkich cech razem. Przy słowniku 10 000 słów musiałbyś oszacować rozkład na 2^10 000 możliwych kombinacji. Niemożliwe.

Naiwne założenie: każda cecha jest warunkowo niezależna przy danej klasie.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Zamiast jednego niemożliwego rozkładu łącznego, szacujesz n prostych rozkładów per-cecha. Każdy potrzebuje tylko zliczeń.

To założenie jest oczywiście błędne. Słowa "maszyna" i "uczenie" nie są niezależne w żadnym dokumencie. Ale klasyfikator nie potrzebuje poprawnych oszacowań prawdopodobieństwa. Potrzebuje poprawnych rankingów -- która klasa ma najwyższe prawdopodobieństwo. Założenie o niezależności wprowadza systematyczne błędy, ale te błędy wpływają podobnie na wszystkie klasy, więc ranking pozostaje poprawny.

### Dlaczego wciąż działa

Trzy powody:

1. **Ranking nad kalibracją.** Klasyfikacja potrzebuje tylko, aby najwyżej sklasyfikowana klasa była poprawna. Nawet jeśli P(spam) = 0.99999, gdy prawdziwe prawdopodobieństwo wynosi 0.7, klasyfikator nadal poprawnie wybiera spam. Nie potrzebujemy poprawnych prawdopodobieństw. Potrzebujemy poprawnego zwycięzcy.

2. **Wysoki błąd systematyczny, niska wariancja.** Założenie o niezależności to silny prior. Silnie ogranicza model, co zapobiega przeuczeniu. Przy ograniczonych danych treningowych model, który jest lekko błędny, ale stabilny, pokonuje model, który jest teoretycznie poprawny, ale bardzo niestabilny. To kompromis błędu systematycznego i wariancji w działaniu.

3. **Redundancja cech znosi się.** Skorelowane cechy dostarczają zbędnych dowodów. Klasyfikator podwójnie liczy ten dowód, ale podwójnie liczy go dla poprawnej klasy. Jeśli "maszyna" i "uczenie" zawsze występują razem, oba dostarczają dowodów dla klasy "tech". Naiwny Bayes liczy je dwa razy, ale liczy je dwa razy dla właściwej klasy.

Czwarty, praktyczny powód: naiwny Bayes jest niezwykle szybki. Trenowanie to jeden przebieg przez dane zliczający częstości. Predykcja to mnożenie macierzy. Możesz trenować na milionie dokumentów w sekundach. Ta szybkość oznacza, że możesz iterować szybciej, wypróbować więcej zestawów cech i przeprowadzić więcej eksperymentów niż z wolniejszymi modelami.

### Matematyka krok po kroku

Prześledźmy konkretny przykład. Załóżmy, że mamy dwie klasy: spam i nie-spam. Nasze słownictwo ma trzy słowa: "free", "money", "meeting".

Dane treningowe:
- E-maile spam zawierają "free" 80 razy, "money" 60 razy, "meeting" 10 razy (150 słów łącznie)
- E-maile nie-spam zawierają "free" 5 razy, "money" 10 razy, "meeting" 100 razy (115 słów łącznie)
- 40% e-maili to spam, 60% to nie-spam

Z wygładzaniem Laplace'a (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Nowy e-mail zawiera: "free" (2 razy), "money" (1 raz), "meeting" (0 razy).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam wygrywa z dużą przewagą. Słowo "free" pojawiające się dwa razy jest silnym dowodem na spam. Zauważ, że "meeting" nie pojawiające się wnosi zero do obu sum logarytmów (0 * log(P)) -- w wielomianowym NB nieobecne słowa nie mają wpływu. To Bernoulliego NB wyraźnie modeluje nieobecność słów.

### Trzy warianty

Naiwny Bayes występuje w trzech wersjach. Każda modeluje `P(feature | class)` inaczej.

#### Wielomianowy naiwny Bayes (Multinomial)

Modeluje każdą cechę jako zliczenie. Najlepszy do danych tekstowych, gdzie cechy to częstości słów lub wartości TF-IDF.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` to wygładzanie Laplace'a (wyjaśnione poniżej). Ten wariant jest podstawą klasyfikacji tekstu.

#### Gaussowski naiwny Bayes (Gaussian)

Modeluje każdą cechę jako rozkład normalny. Najlepszy do cech ciągłych.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Każda klasa otrzymuje własną średnią i wariancję na cechę. Działa dobrze, gdy cechy rzeczywiście mają rozkład dzwonowy w każdej klasie.

#### Bernoulliego naiwny Bayes (Bernoulli)

Modeluje każdą cechę jako binarną (obecna lub nieobecna). Najlepszy do krótkiego tekstu lub binarnych wektorów cech.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Inaczej niż wielomianowy, Bernoulli wyraźnie karze za nieobecność słowa. Jeśli "free" typowo pojawia się w spamie, ale jest nieobecne w tym e-mailu, Bernoulli liczy to jako dowód przeciwko spamowi.

### Kiedy używać którego wariantu

| Wariant | Typ cechy | Najlepszy do | Przykład |
|---------|-------------|----------|---------|
| Wielomianowy | Zliczenia lub częstości | Klasyfikacja tekstu, worek słów | Spam e-mail, klasyfikacja tematów |
| Gaussowski | Wartości ciągłe | Dane tabelaryczne z cechami o rozkładzie zbliżonym do normalnego | Klasyfikacja Iris, dane z czujników |
| Bernoulli | Binarne (0/1) | Krótki tekst, binarne wektory cech | Spam SMS, cechy obecności/nieobecności |

### Wygładzanie Laplace'a

Co się dzieje, gdy słowo pojawia się w danych testowych, ale nigdy nie pojawiło się w danych treningowych dla danej klasy?

Bez wygładzania: `P(word | class) = 0/N = 0`. Jedno zero pomnożone przez cały iloczyn sprawia, że `P(class | features) = 0`, niezależnie od wszystkich innych dowodów. Pojedyncze niewidziane słowo niszczy całą predykcję, bez względu na to, ile innych dowodów ją wspiera.

Wygładzanie Laplace'a dodaje małe zliczenie `alpha` (zwykle 1) do każdego zliczenia cechy:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

Z alpha=1, każde słowo otrzymuje przynajmniej maleńkie prawdopodobieństwo. Słowo "discombobulate" pojawiające się w testowym e-mailu nie zabija już prawdopodobieństwa spamu. Wygładzanie ma interpretację bayesowską: jest równoważne nałożeniu jednorodnego priora Dirichleta na rozkłady słów.

Wyższe alpha oznacza silniejsze wygładzanie (bardziej jednorodne rozkłady). Niższe alpha oznacza, że model bardziej ufa danym. Alpha to hiperparametr, który dostrajasz.

Efekt alpha:

| Alpha | Efekt | Kiedy używać |
|-------|--------|-------------|
| 0.001 | Prawie brak wygładzania, ufaj danym | Bardzo duży zbiór treningowy, brak spodziewanych niewidzianych cech |
| 0.1 | Lekkie wygładzanie | Duży zbiór treningowy |
| 1.0 | Standardowe wygładzanie Laplace'a | Domyślny punkt startowy |
| 10.0 | Silne wygładzanie, spłaszcza rozkłady | Bardzo mały zbiór treningowy, wiele spodziewanych niewidzianych cech |

### Obliczenia w przestrzeni logarytmicznej

Mnożenie setek prawdopodobieństw (każde mniejsze niż 1) powoduje niedomiar zmiennoprzecinkowy. Iloczyn staje się zerem w arytmetyce zmiennoprzecinkowej, mimo że prawdziwa wartość jest bardzo małą liczbą dodatnią.

Rozwiązanie: pracuj w przestrzeni logarytmicznej. Zamiast mnożyć prawdopodobieństwa, dodawaj ich logarytmy:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

To zamienia predykcję w iloczyn skalarny:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Mnożenie macierzy. Dlatego predykcja naiwnego Bayesa jest tak szybka -- to ta sama operacja co jednowarstwowy model liniowy.

### Naiwny Bayes vs regresja logistyczna

Oba są liniowymi klasyfikatorami dla tekstu. Różnica polega na tym, co modelują.

| Aspekt | Naiwny Bayes | Regresja logistyczna |
|--------|------------|-------------------|
| Typ | Generatywny (modeluje P(X\|Y)) | Dyskryminacyjny (modeluje P(Y\|X)) |
| Trenowanie | Zliczanie częstości | Optymalizacja funkcji straty |
| Małe dane | Lepiej (silny prior pomaga) | Gorzej (za mało danych do oszacowania wag) |
| Duże dane | Gorzej (błędne założenie szkodzi) | Lepiej (elastyczna granica) |
| Cechy | Zakłada niezależność | Radzi sobie z korelacjami |
| Szybkość | Jeden przebieg, bardzo szybko | Iteracyjna optymalizacja |
| Kalibracja | Słabe prawdopodobieństwa | Lepsze prawdopodobieństwa |

Zasada kciuka: zacznij od naiwnego Bayesa. Jeśli masz wystarczająco dużo danych i naiwny Bayes osiąga plateau, przełącz się na regresję logistyczną.

### Potok klasyfikacji

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

W praktyce pracujemy w przestrzeni logarytmicznej, aby uniknąć niedomiaru zmiennoprzecinkowego. Zamiast mnożyć wiele małych prawdopodobieństw, dodajemy ich logarytmy:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## Build It

Kod w `code/naive_bayes.py` implementuje zarówno MultinomialNB, jak i GaussianNB od podstaw.

### MultinomialNB

Implementacja od podstaw:

1. **fit(X, y)**: Dla każdej klasy policz częstość każdej cechy. Dodaj wygładzanie Laplace'a. Oblicz logarytmy prawdopodobieństw. Zapisz priory klas (logarytm częstości klas).

2. **predict_log_proba(X)**: Dla każdej próbki oblicz log P(class) + suma log P(feature_i | class) dla wszystkich klas. To mnożenie macierzy: X @ log_probs.T + log_priors.

3. **predict(X)**: Zwróć klasę z najwyższym logarytmem prawdopodobieństwa.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Kluczowy wgląd: po dopasowaniu, predykcja to po prostu mnożenie macierzy plus bias. Dlatego naiwny Bayes jest tak szybki.

### GaussianNB

Dla cech ciągłych szacujemy średnią i wariancję na klasę na cechę:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Predykcja używa gaussowskiego PDF na cechę, pomnożonego przez cechy (dodane w przestrzeni logarytmicznej).

### Demo: klasyfikacja tekstu

Kod generuje syntetyczne dane w formie worka słów symulujące dwie klasy (artykuły techniczne vs artykuły sportowe). Każda klasa ma inny rozkład częstości słów. MultinomialNB klasyfikuje je za pomocą zliczeń słów.

Syntetyczne dane działają tak: tworzymy 200 "słów" (kolumn cech). Słowa 0-39 mają wysoką częstość w artykułach technicznych i niską w sportowych. Słowa 80-119 mają wysoką częstość w sportowych i niską w technicznych. Słowa 40-79 mają średnią częstość w obu. Tworzy to realistyczny scenariusz, w którym niektóre słowa są silnymi wskaźnikami klas, a inne są szumem.

### Demo: cechy ciągłe

Kod generuje dane podobne do Iris (3 klasy, 4 cechy, klastry gaussowskie). GaussianNB klasyfikuje za pomocą średniej i wariancji na klasę. Każda klasa ma inny środek (wektor średnich) i inny rozrzut (wariancję), naśladując dane ze świata rzeczywistego, gdzie pomiary różnią się systematycznie między kategoriami.

Kod pokazuje również:
- **Porównanie wygładzania:** Trenowanie MultinomialNB z różnymi wartościami alpha, aby pokazać wpływ siły wygładzania na dokładność.
- **Eksperyment z rozmiarem treningu:** Jak dokładność NB poprawia się wraz ze wzrostem danych treningowych od 20 do 1600 próbek. NB osiąga przyzwoitą dokładność nawet przy bardzo małej liczbie próbek -- to jego główna zaleta.
- **Macierz pomyłek:** Precyzja, recall i F1 na klasę, aby pokazać, gdzie NB popełnia błędy.

### Szybkość predykcji

Predykcja naiwnego Bayesa to mnożenie macierzy. Dla n próbek z d cechami i k klasami:
- MultinomialNB: jedno mnożenie macierzy (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k ewaluacji gaussowskiego PDF, każda po d cechach = O(n * d * k)

Oba są liniowe w każdym wymiarze. Porównaj to z KNN (który wymaga obliczania odległości do wszystkich punktów treningowych) lub SVM z jądrem RBF (który wymaga ewaluacji jądra względem wszystkich wektorów nośnych). NB jest szybszy o rzędy wielkości w czasie predykcji.

## Use It

Z sklearn oba warianty to jednolinijkowce:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Do klasyfikacji tekstu z sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Kod w `naive_bayes.py` porównuje implementacje od podstaw z sklearn na tych samych danych, aby zweryfikować poprawność.

### TF-IDF z naiwnym Bayesem

Surowe zliczenia słów nadają każdej wystąpieniu słowa równą wagę. Ale powszechne słowa, takie jak "the" i "is", pojawiają się często w każdej klasie -- nie niosą informacji. TF-IDF (Term Frequency - Inverse Document Frequency) zmniejsza wagę powszechnych słów i zwiększa wagę rzadkich, dyskryminujących słów.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Wartości TF-IDF są nieujemne, więc działają z MultinomialNB. Połączenie TF-IDF + MultinomialNB jest jednym z najsilniejszych punktów odniesienia dla klasyfikacji tekstu. Często pokonuje bardziej złożone modele na zbiorach danych z mniej niż 10 000 próbek treningowych.

### BernoulliNB dla krótkiego tekstu

Dla krótkiego tekstu (tweety, SMS, wiadomości chat), BernoulliNB może przewyższyć MultinomialNB. Krótkie teksty mają niskie zliczenia słów, więc informacja o częstości, na której polega MultinomialNB, jest zaszumiona. BernoulliNB dba tylko o obecność lub nieobecność, co jest bardziej wiarygodne w przypadku krótkiego tekstu.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Flaga `binary=True` w CountVectorizer konwertuje wszystkie zliczenia na 0/1. Bez niej BernoulliNB nadal działa, ale widzi zliczenia, do których nie został zaprojektowany.

### Kalibracja prawdopodobieństw NB

Prawdopodobieństwa NB są słabo skalibrowane. Gdy NB mówi P(spam) = 0.95, prawdziwe prawdopodobieństwo może wynosić 0.7. Jeśli potrzebujesz wiarygodnych oszacowań prawdopodobieństwa (na przykład do ustawienia progu lub połączenia z innymi modelami), użyj CalibratedClassifierCV z sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

To dopasowuje regresję logistyczną na surowych wynikach NB przy użyciu walidacji krzyżowej. Otrzymane prawdopodobieństwa są znacznie bliższe prawdziwym częstościom klas.

### Typowe pułapki

1. **Ujemne wartości cech.** MultinomialNB wymaga nieujemnych cech. Jeśli masz ujemne wartości (jak TF-IDF z pewnymi ustawieniami lub zestandaryzowane cechy), użyj zamiast tego GaussianNB lub przesuń cechy, aby były dodatnie.

2. **Cechy o zerowej wariancji.** GaussianNB dzieli przez wariancję. Jeśli cecha ma zerową wariancję dla klasy (wszystkie wartości identyczne), obliczanie prawdopodobieństwa się psuje. Kod dodaje mały człon wygładzający (1e-9) do wszystkich wariancji, aby temu zapobiec.

3. **Nierównowaga klas.** Jeśli 99% e-maili to nie-spam, prior P(nie-spam) = 0.99 jest tak silny, że przytłacza dowody wiarygodności. Możesz ustawić priory klas ręcznie lub użyć parametru class_prior w sklearn.

4. **Skalowanie cech.** MultinomialNB nie wymaga skalowania (działa na zliczeniach). GaussianNB również nie wymaga skalowania (szacuje statystyki na cechę). To przewaga nad regresją logistyczną i SVM, które są wrażliwe na skale cech.

## Ship It

Ta lekcja produkuje:
- `outputs/skill-naive-bayes-chooser.md` -- umiejętność decyzyjna do wyboru odpowiedniego wariantu NB
- `code/naive_bayes.py` -- MultinomialNB i GaussianNB od podstaw, z porównaniem z sklearn

### Kiedy naiwny Bayes zawodzi

NB zawodzi, gdy założenie o niezależności powoduje niepoprawne rankingi (nie tylko niepoprawne prawdopodobieństwa). Dzieje się tak, gdy:

1. **Silne interakcje cech.** Jeśli klasa zależy od kombinacji dwóch cech, ale nie od żadnej z osobna (wzorce XOR), NB całkowicie to przegapi. Każda cecha osobno nie dostarcza dowodów, a NB nie może ich połączyć nieliniowo.

2. **Wysoce skorelowane cechy z przeciwstawnymi dowodami.** Jeśli cecha A mówi "spam", a cecha B mówi "nie-spam", ale A i B są idealnie skorelowane (w rzeczywistości zawsze się zgadzają), NB zobaczy sprzeczne dowody, których nie ma.

3. **Bardzo duże zbiory treningowe.** Z wystarczającą ilością danych, modele dyskryminacyjne, takie jak regresja logistyczna, uczą się prawdziwej granicy decyzyjnej i przewyższają NB. Założenie o niezależności, które pomagało przy małych danych, teraz ogranicza model.

W praktyce te tryby awarii są rzadkie w klasyfikacji tekstu. Cechy tekstowe są liczne, indywidualnie słabe, a błędy założenia o niezależności mają tendencję do znoszenia się. Dla danych tabelarycznych z kilkoma silnie skorelowanymi cechami rozważ najpierw regresję logistyczną lub modele oparte na drzewach.

## Exercises

1. **Eksperyment z wygładzaniem.** Trenuj MultinomialNB na danych tekstowych z wartościami alpha 0.01, 0.1, 1.0, 10.0 i 100.0. Narysuj wykres dokładności vs alpha. Gdzie osiągany jest szczyt wydajności? Dlaczego bardzo wysokie alpha szkodzi?

2. **Test niezależności cech.** Weź rzeczywisty zbiór danych tekstowych. Wybierz dwa słowa, które są oczywiście skorelowane ("machine" i "learning"). Oblicz P(word1 | class) * P(word2 | class) i porównaj z P(word1 AND word2 | class). Jak bardzo błędne jest założenie o niezależności? Czy wpływa na dokładność klasyfikacji?

3. **Implementacja Bernoulliego.** Rozszerz kod o klasę BernoulliNB. Konwertuj worek słów na binarny (obecne/nieobecne) i porównaj dokładność z MultinomialNB na danych tekstowych. Kiedy Bernoulli wygrywa?

4. **NB vs regresja logistyczna.** Trenuj oba na danych tekstowych. Zacznij od 100 próbek treningowych i zwiększaj do 10 000. Narysuj wykres dokładności vs rozmiar zbioru treningowego dla obu. W którym momencie regresja logistyczna przewyższa naiwny Bayes?

5. **Filtr spamu.** Zbuduj kompletny klasyfikator spamu: tokenizuj surowy tekst e-maila, zbuduj słownik, utwórz cechy worka słów, trenuj MultinomialNB, oceń za pomocą precyzji i recall (nie tylko dokładności -- dlaczego?).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naiwny Bayes | "Prosty probabilistyczny klasyfikator" | Klasyfikator stosujący twierdzenie Bayesa z założeniem, że cechy są warunkowo niezależne przy danej klasie |
| Niezależność warunkowa | "Cechy nie wpływają na siebie" | P(A, B \| C) = P(A \| C) * P(B \| C) -- znajomość B nie mówi nic nowego o A, gdy znasz C |
| Wygładzanie Laplace'a | "Wygładzanie dodaj-jeden" | Dodawanie małego zliczenia do każdej cechy, aby zapobiec dominacji zerowych prawdopodobieństw w predykcji |
| Prior | "Co wierzyłeś przed zobaczeniem danych" | P(class) -- prawdopodobieństwo każdej klasy przed obserwacją jakichkolwiek cech |
| Wiarygodność | "Jak dobrze dane pasują" | P(features \| class) -- prawdopodobieństwo zaobserwowania tych cech, jeśli klasa jest znana |
| Posterior | "Co wierzysz po zobaczeniu danych" | P(class \| features) -- zaktualizowane prawdopodobieństwo klasy po obserwacji cech |
| Model generatywny | "Modeluje jak dane są generowane" | Model, który uczy się P(X \| Y) i P(Y), a następnie używa twierdzenia Bayesa do uzyskania P(Y \| X) |
| Model dyskryminacyjny | "Modeluje granicę decyzyjną" | Model, który bezpośrednio uczy się P(Y \| X) bez modelowania, jak X jest generowane |
| Logarytm prawdopodobieństwa | "Unikanie niedomiaru" | Praca z log P zamiast P, aby zapobiec zerowaniu się iloczynu wielu małych liczb w arytmetyce zmiennoprzecinkowej |

## Further Reading

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- all three variants with mathematical details
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- the classic comparison of Multinomial vs Bernoulli for text
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- improvements to NB for text
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- proves NB converges faster than LR with less data