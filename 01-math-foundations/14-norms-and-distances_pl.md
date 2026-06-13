# Normy i odległości

> Twoja funkcja odległości definiuje, co znaczy "podobne". Wybierz źle, a wszystko dalsze się psuje.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minut

## Learning Objectives

- Zaimplementuj funkcje odległości L1, L2, cosinusową, Mahalanobisa, Jaccarda i edycyjną od podstaw
- Wybierz odpowiednią metrykę odległości dla danego zadania ML i wyjaśnij, dlaczego alternatywy zawodzą
- Połącz normy L1 i L2 z regularyzacją LASSO i Ridge oraz ich geometrycznymi regionami ograniczeń
- Zademonstruj, jak te same dane produkują różnych najbliższych sąsiadów pod różnymi metrykami

## Problem

Masz dwa wektory. Może to embeddingi słów. Może profile użytkowników. Może tablice pikseli. Potrzebujesz wiedzieć: jak blisko są?

Odpowiedź zależy całkowicie od tego, którą funkcję odległości wybierzesz. Dwa punkty danych mogą być najbliższymi sąsiadami pod jedną metryką i daleko od siebie pod inną. Twój klasyfikator KNN, silnik rekomendacji, baza wektorów, algorytm klastrowania, funkcja straty -- wszystkie zależą od tego wyboru. Wybierz źle, a twój model optymalizuje niewłaściwą rzecz.

Nie ma uniwersalnej najlepszej odległości. L2 działa dla danych przestrzennych. Podobieństwo cosinusowe dominuje w NLP. Jaccard obsługuje zbiory. Odległość edycyjna obsługuje stringi. Mahalanobis uwzględnia korelacje. Wasserstein przesuwa masę prawdopodobieństwa. Każda koduje inne założenie o tym, co znaczy "podobne".

Ta lekcja buduje każdą główną funkcję odległości od podstaw, pokazuje, kiedy każda jest właściwym narzędziem, i demonstruje, jak te same dane produkują zupełnie różnych najbliższych sąsiadów w zależności od używanej metryki.

## Koncepcja

### Normy: mierzenie wielkości wektora

Norma mierzy "rozmiar" wektora. Każdą funkcję odległości między dwoma wektorami można zapisać jako normę ich różnicy: d(a, b) = ||a - b||. Zatem zrozumienie norm to zrozumienie odległości.

### Norma L1 (odległość Manhattan)

Norma L1 sumuje wartości bezwzględne wszystkich składowych.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nazywa się odległością Manhattan, bo mierzy, jak daleko idziesz na siatce miejskiej, gdzie możesz poruszać się tylko wzdłuż osi. Bez przekątnych.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L1 = |4-1| + |5-1| = 3 + 4 = 7

Na siatce idziesz 3 przecznice wschód i 4 przecznice północ.
```

Kiedy używać L1:
- Wysokowymiarowe rzadkie dane (cechy tekstowe, kodowanie one-hot)
- Gdy chcesz odporności na wartości odstające (pojedyncza ogromna różnica nie dominuje)
- Problemy selekcji cech (regularyzacja L1 promuje rzadkość)

Związek z regularyzacją L1 (Lasso): dodanie ||w||_1 do funkcji straty karze sumę bezwzględnych wartości wag. To popycha małe wagi dokładnie do zera, wykonując automatyczną selekcję cech. Kara L1 tworzy diamentowe regiony ograniczeń w przestrzeni wag, a wierzchołki diamentów leżą na osiach, gdzie niektóre wagi są zerowe.

Związek z funkcjami straty: Mean Absolute Error (MAE) to średnia odległość L1 między przewidywaniami a celami. Karze wszystkie błędy liniowo, czyniąc ją odporną na wartości odstające w porównaniu do MSE.

### Norma L2 (odległość euklidesowa)

Norma L2 to odległość w linii prostej. Pierwiastek kwadratowy z sumy kwadratów składowych.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

To odległość, której nauczyłeś się na geometrii. Pitagoras w n wymiarach.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L2 = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

Linia prosta, tnąca po przekątnej przez siatkę.
```

Kiedy używać L2:
- Dane ciągłe o niskiej do średniej wymiarowości
- Gdy skale cech są porównywalne
- Odległości fizyczne (dane przestrzenne, odczyty czujników)
- Podobieństwo obrazów na poziomie pikseli

Związek z regularyzacją L2 (Ridge): dodanie ||w||_2^2 do funkcji straty karze duże wagi. W przeciwieństwie do L1, nie popycha wag do zera. Ściska wszystkie wagi w kierunku zera proporcjonalnie. Kara L2 tworzy okrągłe regiony ograniczeń, więc nie ma wierzchołków na osiach. Wagi stają się małe, ale rzadko dokładnie zerowe.

Związek z funkcjami straty: Mean Squared Error (MSE) to średnia kwadratów odległości L2. Kwadrat karze duże błędy bardziej niż małe.

```
MAE (strata L1):  |y - y_hat|         Kara liniowa. Odporna na wartości odstające.
MSE (strata L2):  (y - y_hat)^2       Kara kwadratowa. Wrażliwa na wartości odstające.
```

### Normy Lp: ogólna rodzina

L1 i L2 to szczególne przypadki normy Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Różne wartości p produkują różne kształty "kul jednostkowych" (zbior wszystkich punktów w odległości 1 od początku):

```
p=1:    Kształt diamentu  (wierzchołki na osiach)
p=2:    Okrąg/kula       (typowa okrągła kula)
p=3:    Superelipsa      (zaokrąglony kwadrat)
p=inf:  Kwadrat/hipersześcian (płaskie boki wzdłuż osi)
```

### Norma L-nieskończoność (odległość Czebyszewa)

Gdy p dąży do nieskończoności, norma Lp zbiega do maksymalnej bezwzględnej składowej.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Odległość między dwoma punktami jest określona przez pojedynczy wymiar, w którym różnią się najbardziej. Wszystkie inne wymiary są ignorowane.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L-inf = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Kiedy używać L-nieskończoności:
- Gdy najgorsze odchylenie w dowolnym pojedynczym wymiarze ma znaczenie
- Gry planszowe (król w szachach porusza się w L-nieskończoności: jeden krok w dowolnym kierunku kosztuje 1)
- Tolerancje produkcyjne (każdy wymiar musi być w specyfikacji)

### Podobieństwo cosinusowe i odległość cosinusowa

Podobieństwo cosinusowe mierzy kąt między dwoma wektorami, ignorując ich długości.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Zakres od -1 (przeciwne kierunki) do +1 (ten sam kierunek). Prostopadłe wektory mają podobieństwo cosinusowe 0.

Odległość cosinusowa przekształca to na odległość: odległość_cos = 1 - podobieństwo_cos. Zakres od 0 (identyczny kierunek) do 2 (przeciwny kierunek).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Dlaczego cosinus dominuje w NLP i embeddingach: w tekście długość dokumentu nie powinna wpływać na podobieństwo. Dokument o kotach, dwa razy dłuższy niż inny dokument o kotach, powinien być nadal "podobny". Podobieństwo cosinusowe ignoruje długość i przejmuje się tylko kierunkiem. Dwa dokumenty z tym samym rozkładem słów, ale różnymi długościami, wskazują ten sam kierunek i dostają podobieństwo cosinusowe 1.0.

Kiedy używać podobieństwa cosinusowego:
- Podobieństwo tekstu (wektory TF-IDF, embeddingi słów, embeddingi zdań)
- Dowolna dziedzina, gdzie długość to szum, a kierunek to sygnał
- Systemy rekomendacji (wektory preferencji użytkownika)
- Wyszukiwanie embeddingów (bazy wektorów prawie zawsze używają cosinusa lub iloczynu skalarnego)

### Iloczyn skalarny vs podobieństwo cosinusowe

Iloczyn skalarny dwóch wektorów to:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(kąt)
```

Podobieństwo cosinusowe to iloczyn skalarny znormalizowany przez obie długości. Gdy oba wektory są już znormalizowane jednostkowo (długość = 1), iloczyn skalarny i podobieństwo cosinusowe są identyczne.

```
Jeśli ||a|| = 1 i ||b|| = 1:
    a . b = cos(kąt między a i b)
```

Gdy się różnią: iloczyn skalarny zawiera informację o długości. Wektor z większą długością dostaje wyższy wynik iloczynu skalarnego. To ma znaczenie w niektórych systemach wyszukiwania, gdzie chcesz, by "popularne" elementy były wyżej. Długość działa jako ukryty sygnał jakości lub ważności.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Oba zgadzają się co do kierunku, ale iloczyn skalarny odzwierciedla też długość.
```

W praktyce:
- Użyj podobieństwa cosinusowego, gdy chcesz czystego kierunkowego podobieństwa
- Użyj iloczynu skalarnego, gdy długości niosą znaczącą informację
- Wiele baz wektorów (Pinecone, Weaviate, Qdrant) pozwala wybierać między nimi
- Jeśli twoje embeddingi są L2-znormalizowane, wybór nie ma znaczenia

### Odległość Mahalanobisa

Odległość euklidesowa traktuje wszystkie wymiary równo. Ale jeśli twoje cechy są skorelowane lub mają różne skale, L2 daje mylące wyniki.

Odległość Mahalanobisa uwzględnia strukturę kowariancji danych.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

gdzie S to macierz kowariancji danych.

Intuicyjnie: odległość Mahalanobisa najpierw dekorreluje i normalizuje dane (wybielanie), a następnie oblicza odległość L2 w tej przekształconej przestrzeni. Jeśli S jest macierzą jednostkową (cechy nieskorelowane, jednostkowa wariancja), odległość Mahalanobisa redukuje się do odległości euklidesowej.

```
Przykład: wzrost i waga są skorelowane.
Ktoś 188 cm i 82 kg nie jest niezwykły.
Ktoś 152 cm i 82 kg jest niezwykły.

Odległość euklidesowa może powiedzieć, że oboje są równie daleko od średniej.
Odległość Mahalanobisa poprawnie identyfikuje drugiego jako wartość odstającą,
bo uwzględnia korelację wzrost-waga.
```

Kiedy używać odległości Mahalanobisa:
- Wykrywanie wartości odstających (punkty z dużą odległością Mahalanobisa od średniej to wartości odstające)
- Klasyfikacja, gdy cechy mają różne skale i korelacje
- Gdy masz wystarczająco danych, by oszacować wiarygodną macierz kowariancji
- Kontrola jakości w produkcji (wielowymiarowe monitorowanie procesu)

### Podobieństwo Jaccarda (dla zbiorów)

Podobieństwo Jaccarda mierzy przecięcie dwóch zbiorów.

```
J(A, B) = |A przecięcie B| / |A suma B|
```

Zakres od 0 (brak przecięcia) do 1 (identyczne zbiory). Odległość Jaccarda = 1 - podobieństwo Jaccarda.

```
A = {kot, pies, ryba}
B = {kot, ptak, ryba, wąż}

Przecięcie = {kot, ryba}           rozmiar = 2
Suma = {kot, pies, ryba, ptak, wąż}  rozmiar = 5

Podobieństwo Jaccarda = 2/5 = 0.4
Odległość Jaccarda = 0.6
```

Kiedy używać Jaccarda:
- Porównywanie zbiorów tagów, kategorii lub cech
- Podobieństwo dokumentów na podstawie obecności słów (nie częstości)
- Wykrywanie prawie duplikatów (MinHash przybliżenie Jaccarda)
- Porównywanie binarnych wektorów cech (dane o obecności/braku)
- Ocena modeli segmentacji (Intersection over Union = Jaccard)

### Odległość edycyjna (odległość Levenshteina)

Odległość edycyjna liczy minimalną liczbę operacji na pojedynczych znakach potrzebną do przekształcenia jednego stringa w drugi. Operacje to: wstaw, usuń lub zastąp.

```
"kitten" -> "sitting"

kitten -> sitten  (zastąp k -> s)
sitten -> sittin  (zastąp e -> i)
sittin -> sitting (wstaw g)

Odległość edycyjna = 3
```

Obliczana przy użyciu programowania dynamicznego. Wypełnij macierz, gdzie element (i, j) to odległość edycyjna między pierwszymi i znakami stringa A a pierwszymi j znakami stringa B.

```
         ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Kiedy używać odległości edycyjnej:
- Sprawdzanie pisowni i korekta
- Dopasowanie sekwencji DNA (z ważonymi operacjami)
- Rozmyte dopasowanie stringów
- Deduplikacja nieczystych danych tekstowych

### Dywergencja KL (nie odległość, ale używana jak jedna)

Dywergencja KL mierzy, jak bardzo jeden rozkład prawdopodobieństwa różni się od innego. Omówiona w Lekcji 09, ale należy do tej dyskusji, bo ludzie używają jej jako "odległości" mimo że nią nie jest.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Krytyczna własność: dywergencja KL NIE jest symetryczna.

```
D_KL(P || Q) != D_KL(Q || P)
```

To znaczy, że nie spełnia podstawowego wymogu metryki odległości. Nie spełnia też nierówności trójkąta. Jest dywergencją, a nie odległością.

Forward KL (D_KL(P || Q)) jest "średnio-szukający": Q próbuje pokryć wszystkie tryby P. Reverse KL (D_KL(Q || P)) jest "trybo-szukający": Q skupia się na pojedynczym trybie P.

Gdy widzisz dywergencję KL:
- VAE (człon KL w ELBO popycha rozkład latentny w kierunku a priori)
- Dystylacja wiedzy (uczeń próbuje dopasować rozkład nauczyciela)
- RLHF (kara KL utrzymuje dostrojony model blisko modelu bazowego)
- Metody gradientu polityki (ograniczanie aktualizacji polityki)

### Odległość Wassersteina (odległość kramarza)

Odległość Wassersteina mierzy minimalną "pracę" potrzebną do przekształcenia jednego rozkładu prawdopodobieństwa w drugi. Myśl o tym: jeśli jeden rozkład to kupa ziemi, a drugi to dół, ile ziemi musisz przenieść i jak daleko?

```
W(P, Q) = inf po wszystkich planach transportu gamma z E[d(x, y)]
```

Dla rozkładów 1D upraszcza się do całki z bezwzględnej różnicy dystrybuant:

```
W_1(P, Q) = całka |CDF_P(x) - CDF_Q(x)| dx
```

Dlaczego Wasserstein ma znaczenie:
- Jest prawdziwą metryką (symetryczna, spełnia nierówność trójkąta)
- Daje gradienty nawet, gdy rozkłady się nie pokrywają (dywergencja KL idzie do nieskończoności)
- Ta własność uczyniła go centralnym dla WGANów, które rozwiązały niestabilność trenowania oryginalnych GAN

```
Rozkłady bez przecięcia:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

Dywergencja KL: nieskończoność (log z zera)
Wasserstein: 4 (przenieś całą masę o 4 pojemniki)

Wasserstein daje znaczący gradient. KL nie.
```

Kiedy używać Wassersteina:
- Trenowanie GAN (WGAN, WGAN-GP)
- Porównywanie rozkładów, które mogą się nie pokrywać
- Problemy optymalnego transportu
- Wyszukiwanie obrazów (porównywanie histogramów kolorów)

### Dlaczego różne zadania potrzebują różnych odległości

| Zadanie | Najlepsza odległość | Dlaczego |
|------|--------------|-----|
| Podobieństwo tekstu | Cosinus | Długość to szum, kierunek to znaczenie |
| Porównanie pikseli obrazu | L2 | Relacje przestrzenne mają znaczenie, cechy są porównywalnej skali |
| Rzadkie cechy wysokowymiarowe | L1 | Odporna, nie amplifikuje rzadkich dużych różnic |
| Przecięcie zbiorów (tagi, kategorie) | Jaccard | Dane są naturalnie zbiorowe, nie wektorowe |
| Dopasowanie stringów | Edycyjna | Operacje mapują na ludzką intuicję edycyjną |
| Wykrywanie wartości odstających | Mahalanobis | Uwzględnia korelacje i skale cech |
| Porównywanie rozkładów | Dywergencja KL | Mierzy informację straconą przez używanie Q zamiast P |
| Trenowanie GAN | Wasserstein | Daje gradienty nawet, gdy rozkłady się nie pokrywają |
| Embeddingi (baza wektorów) | Cosinus lub iloczyn skalarny | Embeddingi są trenowane do kodowania znaczenia w kierunku |
| Rekomendacje | Iloczyn skalarny | Długość może kodować popularność lub pewność |
| Sekwencje DNA | Ważona odległość edycyjna | Koszty zastąpienia różnią się w zależności od pary nukleotydów |
| Kontrola jakości w produkcji | L-nieskończoność | Najgorsze odchylenie w dowolnym wymiarze ma znaczenie |

### Związek z funkcjami straty

Funkcje straty to funkcje odległości zastosowane do przewidywań vs celów.

```
Funkcja straty       Używana odległość       Zachowanie
MSE                 L2 do kwadratu          Karze duże błędy mocno
MAE                 L1                      Karze wszystkie błędy równo
Strata Hubera       L1 dla dużych błędów,   Najlepsze z obu: odporna na wartości
                     L2 dla małych błędów    odstające, gładki gradient przy zerze
Cross-entropia      Dywergencja KL          Mierzy niedopasowanie rozkładów
Strata zawiasu      max(0, margines - d)    Karze tylko poniżej marginesu
Strata tripletowa   L2 (zazwyczaj)           Przyciąga pozytywne, odpycha negatywne
Strata kontrastowa  L2                      Podobne pary blisko, różne pary poza marginesem
```

### Związek z regularyzacją

Regularyzacja dodaje karę normy na wagach do funkcji straty.

```
Regularyzacja L1 (Lasso):   strata + lambda * ||w||_1
  -> Rzadkie wagi. Niektóre wagi stają się dokładnie zerowe.
  -> Automatyczna selekcja cech.
  -> Rozwiązanie ma wierzchołki (nieróżniczkowalne przy zerze).

Regularyzacja L2 (Ridge):   strata + lambda * ||w||_2^2
  -> Małe wagi. Wszystkie wagi kurczą się w kierunku zera.
  -> Bez selekcji cech (nic nie idzie dokładnie do zera).
  -> Gładkie rozwiązanie wszędzie.

Elastic Net:                  strata + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Łączy rzadkość L1 ze stabilnością L2.
  -> Grupy skorelowanych cech są zachowane lub odrzucone razem.
```

Dlaczego L1 produkuje rzadkość, a L2 nie: wyobraź sobie region ograniczeń w 2D przestrzeni wag. L1 to diament, L2 to okrąg. Kontury funkcji straty (elipsy) najprawdopodobniej dotkną diamentu w wierzchołku, gdzie jedna waga jest zerowa. Dotkną okręgu w gładkim punkcie, gdzie obie wagi są niezerowe.

### Wyszukiwanie najbliższego sąsiada

Każda funkcja odległości implikuje problem wyszukiwania najbliższego sąsiada: dla danego punktu zapytania znajdź najbliższe punkty w zbiorze danych.

Dokładne wyszukiwanie najbliższego sąsiada to O(n * d) na zapytanie w zbiorze n punktów z d wymiarami. Dla dużych zbiorów danych to zbyt wolne.

Przybliżone wyszukiwanie najbliższego sąsiada (ANN) wymienia małą dokładność na ogromne zyski szybkości:

```
Algorytm         Podejście                      Używany przez
KD-drzewa        Podział przestrzeni osiowo      scikit-learn (niskie wym)
Kule             Zagnieżdżone hipersfery         scikit-learn (średnie wym)
LSH              Losowe projekcje hashujące      Wykrywanie prawie duplikatów
HNSW             Hierarchiczny graf              FAISS, Qdrant, Weaviate
                 małego świata
IVF              Odwrócony indeks plików z       FAISS (skala miliardów)
                 wyszukiwaniem klastrowym
Kwant. produktu  Kompresuj wektory, szukaj       FAISS (ograniczona pamięć)
                 w skompresowanej przestrzeni
```

HNSW (Hierarchical Navigable Small World) jest dominującym algorytmem w nowoczesnych bazach wektorów. Buduje wielowarstwowy graf, gdzie każdy węzeł łączy się ze swoimi przybliżonymi najbliższymi sąsiadami. Wyszukiwanie zaczyna się w górnej warstwie (rzadkiej, dalekie skoki) i schodzi do dolnej warstwy (gęstej, krótkie skoki).

```figure
norm-unit-balls
```

## Build It

### Krok 1: Wszystkie funkcje normy i odległości

Zobacz `code/distances.py` dla kompletnej implementacji. Każda funkcja jest zbudowana od podstaw używając tylko podstawowej matematyki Pythona.

### Krok 2: Te same dane, różne odległości, różni sąsiedzi

Demo w `distances.py` tworzy zbiór danych, wybiera punkt zapytania i pokazuje, jak najbliższy sąsiad zmienia się w zależności od metryki odległości. Punkt, który jest "najbliższy" pod L1, może nie być najbliższy pod L2 lub cosinusem.

### Krok 3: Wyszukiwanie podobieństwa embeddingów

Kod zawiera przykładowe wyszukiwanie podobieństwa embeddingów, które znajduje najbardziej podobne "dokumenty" do zapytania używając podobieństwa cosinusowego vs odległości L2, pokazując, że rankingi mogą się różnić.

## Use It

Najczęstsze praktyczne zastosowanie: znajdowanie podobnych elementów w bazie wektorów.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 najbardziej podobnych do elementu 0: {top_k}")
print(f"Podobieństwa: {similarities[top_k]}")
```

Gdy wywołujesz `model.encode(text)` i następnie szukasz w bazie wektorów, to dzieje się pod maską. Model embeddingów mapuje tekst na wektory. Baza wektorów oblicza podobieństwo cosinusowe (lub iloczyn skalarny) między twoim wektorem zapytania a każdym przechowywanym wektorem, używając algorytmów ANN, by uniknąć sprawdzania wszystkich.

## Ćwiczenia

1. Oblicz odległości L1, L2 i L-nieskończoność między (1, 2, 3) a (4, 0, 6). Zweryfikuj, że L-inf <= L2 <= L1 zawsze zachodzi dla dowolnej pary punktów. Udowodnij, dlaczego to uporządkowanie jest gwarantowane.

2. Stwórz dwa wektory, gdzie podobieństwo cosinusowe jest wysokie (> 0.9), ale odległość L2 jest duża (> 10). Wyjaśnij geometrycznie, co się dzieje. Następnie stwórz dwa wektory, gdzie podobieństwo cosinusowe jest niskie (< 0.3), ale odległość L2 jest mała (< 0.5).

3. Zaimplementuj funkcję, która przyjmuje zbiór danych i punkt zapytania i zwraca najbliższego sąsiada pod L1, L2, cosinusem i odległością Mahalanobisa. Znajdź zbiór danych, gdzie wszystkie cztery nie zgadzają się co do tego, który punkt jest najbliższy.

4. Oblicz odległość Wassersteina między [0.5, 0.5, 0, 0] a [0, 0, 0.5, 0.5] ręcznie używając metody CDF. Następnie oblicz ją między [0.25, 0.25, 0.25, 0.25] a [0, 0, 0.5, 0.5]. Która jest większa i dlaczego?

5. Zaimplementuj MinHash dla przybliżonego podobieństwa Jaccarda. Wygeneruj 100 losowych zbiorów, oblicz dokładny Jaccard dla wszystkich par i porównaj z przybliżeniem MinHash używając 50, 100 i 200 funkcji hashujących. Wykreśl błąd przybliżenia.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| Norma | "Rozmiar wektora" | Funkcja mapująca wektor na nieujemny skalar, spełniająca nierówność trójkąta, bezwzględną jednorodność i zero tylko dla wektora zerowego |
| Norma L1 | "Odległość Manhattan" | Suma bezwzględnych wartości składowych. Produkuje rzadkość w optymalizacji. Odporna na wartości odstające |
| Norma L2 | "Odległość euklidesowa" | Pierwiastek kwadratowy z sumy kwadratów składowych. Odległość w linii prostej w przestrzeni euklidesowej |
| Norma Lp | "Uogólniona norma" | p-ty pierwiastek z sumy p-tych potęg bezwzględnych składowych. L1 i L2 to przypadki szczególne |
| Norma L-nieskończoność | "Norma max" lub "odległość Czebyszewa" | Maksymalna bezwzględna wartość składowej. Granica Lp gdy p dąży do nieskończoności |
| Podobieństwo cosinusowe | "Kąt między wektorami" | Iloczyn skalarny znormalizowany przez obie długości. Zakres od -1 do +1. Ignoruje długość wektora |
| Odległość cosinusowa | "1 minus podobieństwo cosinusowe" | Przekształca podobieństwo cosinusowe na odległość. Zakres od 0 do 2 |
| Iloczyn skalarny | "Nienormalizowany cosinus" | Suma iloczynów składowych. Równa się podobieństwo cosinusowe razy obie długości |
| Odległość Mahalanobisa | "Odległość świadoma korelacji" | Odległość L2 w przestrzeni, która została wybielona (dekorrelowana i znormalizowana) używając macierzy kowariancji danych |
| Podobieństwo Jaccarda | "Przecięcie zbiorów" | Rozmiar przecięcia podzielony przez rozmiar sumy. Dla zbiorów, nie wektorów |
| Odległość edycyjna | "Odległość Levenshteina" | Minimalna liczba wstawień, usunięć i zastąpień do przekształcenia jednego stringa w drugi |
| Dywergencja KL | "Odległość między rozkładami" | Nie prawdziwa odległość (niesymetryczna). Mierzy dodatkowe bity z używania Q do kodowania P |
| Odległość Wassersteina | "Odległość kramarza" | Minimalna praca do przeniesienia masy z jednego rozkładu do drugiego. Prawdziwa metryka |
| Przybliżony najbliższy sąsiad | "Wyszukiwanie ANN" | Algorytmy (HNSW, LSH, IVF) znajdujące w przybliżeniu najbliższe punkty znacznie szybciej niż dokładne wyszukiwanie |
| HNSW | "Algorytm bazy wektorów" | Hierarchiczny graf małego świata. Wielowarstwowy graf do szybkiego przybliżonego wyszukiwania najbliższych sąsiadów |
| Regularyzacja L1 | "Lasso" | Dodanie normy L1 wag do straty. Popycha wagi do zera (rzadkość) |
| Regularyzacja L2 | "Ridge" lub "zanik wag" | Dodanie kwadratu normy L2 wag do straty. Ściska wagi w kierunku zera bez rzadkości |
| Elastic Net | "L1 + L2" | Łączy regularyzację L1 i L2. Lepiej radzi sobie z grupami skorelowanych cech niż każda z osobna |