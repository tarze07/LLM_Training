# Statystyka dla uczenia maszynowego

> Statystyka to sposób, w jaki wiesz, czy twój model faktycznie działa, czy tylko miał szczęście.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minut

## Learning Objectives

- Oblicz statystyki opisowe, korelację Pearsona/Spearmana i macierze kowariancji od podstaw
- Wykonaj testy hipotez (t-test, chi-kwadrat) i poprawnie interpretuj p-wartości oraz przedziały ufności
- Użyj bootstrapu do konstrukcji przedziałów ufności dla dowolnej metryki bez założeń dystrybucyjnych
- Rozróżnij istotność statystyczną od praktycznej używając miar wielkości efektu

## Problem

Wytrenowałeś dwa modele. Model A osiąga 0.87 na zbiorze testowym. Model B osiąga 0.89. Wdrażasz Model B. Trzy tygodnie później metryki produkcyjne są gorsze niż wcześniej. Co się stało?

Model B faktycznie nie był lepszy od modelu A. Różnica 0.02 była szumem. Twój zbiór testowy był zbyt mały, wariancja zbyt wysoka, lub oba. Wypuściłeś losowość przebraną za poprawę.

To zdarza się stale. Przetasowania na tablicy liderów Kaggle. Prace, których nie da się odtworzyć. Testy A/B ogłaszające zwycięzców na podstawie kilkuset próbek. Podstawowa przyczyna jest zawsze ta sama: ktoś pominął statystykę.

Statystyka daje narzędzia do odróżniania sygnału od szumu. Mówi, kiedy różnica jest prawdziwa, jak pewny powinieneś być i ile danych potrzebujesz, zanim możesz zaufać wynikowi. Każdy potok ML, każde porównanie modeli, każdy eksperyment potrzebuje statystyki. Bez niej zgadujesz.

## Koncepcja

### Statystyki opisowe: podsumowanie danych

Zanim cokolwiek zamodelujesz, musisz wiedzieć, jak wyglądają twoje dane. Statystyki opisowe kompresują zbiór danych do kilku liczb, które przechwytują jego kształt.

**Miary tendencji centralnej** odpowiadają "gdzie jest środek?"

```
Średnia:   suma wszystkich wartości / liczba
         mu = (1/n) * sum(x_i)

Mediana: środkowa wartość po posortowaniu
         Odporna na wartości odstające. Jeśli masz [1, 2, 3, 4, 1000], średnia to 202,
         ale mediana to 3.

Moda:   najczęstsza wartość
         Przydatna dla danych kategorycznych. Dla danych ciągłych rzadko informacyjna.
```

Średnia to punkt równowagi. Mediana to punkt środkowy. Gdy się rozchodzą, twój rozkład jest skośny. Rozkłady dochodów mają średnią >> medianę (prawostronna skośność od miliarderów). Rozkłady straty podczas trenowania często mają średnią << medianę (lewostronna skośność od łatwych próbek).

**Miary rozrzutu** odpowiadają "jak rozproszone są dane?"

```
Wariancja:   średnia kwadratów odchyleń od średniej
             sigma^2 = (1/n) * sum((x_i - mu)^2)

Odchylenie standardowe:  pierwiastek kwadratowy z wariancji
                         sigma = sqrt(sigma^2)
                         Te same jednostki co dane, więc bardziej interpretowalne.

Zakres:      max - min
             Wrażliwy na wartości odstające. Prawie nigdy nieprzydatny samodzielnie.

IQR:        Q3 - Q1 (rozstęp międzykwartylowy)
             Zakres środkowych 50% danych.
             Odporny na wartości odstające. Używany do wykresów pudełkowych i wykrywania wartości odstających.
```

**Percentyle** dzielą posortowane dane na 100 równych części. 25. percentyl (Q1) oznacza, że 25% wartości leży poniżej tego punktu. 50. percentyl to mediana. 75. percentyl to Q3.

```
Do monitorowania opóźnień:
  P50 = mediana opóźnienia        (typowe doświadczenie użytkownika)
  P95 = 95. percentyl             (zły, ale nie najgorszy przypadek)
  P99 = 99. percentyl             (opóźnienie ogonowe, często 10x mediany)
```

W ML interesują cię percentyle dla opóźnienia wnioskowania, rozkładów pewności predykcji i zrozumienia rozkładów błędów. Model z niskim średnim błędem, ale fatalnym P99 może być bezużyteczny dla aplikacji bezpieczeństwa krytycznego.

**Statystyki próbki vs populacji.** Obliczając wariancję z próbki, dziel przez (n-1) zamiast n. To korekta Bessela. Kompensuje fakt, że średnia z próbki nie jest prawdziwą średnią populacji. Z n w mianowniku systematycznie zaniżasz prawdziwą wariancję. Z (n-1) estymacja jest nieobciążona.

```
Wariancja populacji: sigma^2 = (1/N) * sum((x_i - mu)^2)
Wariancja próbki:   s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

W praktyce: jeśli n jest duże (tysiące próbek), różnica jest pomijalna. Jeśli n jest małe (dziesiątki próbek), ma znaczenie.

### Korelacja: jak zmienne poruszają się razem

Korelacja mierzy siłę i kierunek liniowej zależności między dwiema zmiennymi.

**Współczynnik korelacji Pearsona** mierzy liniową asociację:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  idealna dodatnia zależność liniowa
r = -1:  idealna ujemna zależność liniowa
r =  0:  brak liniowej zależności (ale może istnieć nieliniowa!)

Zakres: [-1, 1]
```

Pearson zakłada, że zależność jest liniowa i obie zmienne są w przybliżeniu normalnie rozłożone. Jest wrażliwy na wartości odstające. Pojedynczy ekstremalny punkt może przeciągnąć r z 0.1 do 0.9.

**Korelacja rang Spearmana** mierzy monotoniczną asociację:

```
1. Zastąp każdą wartość jej rangą (1, 2, 3, ...)
2. Oblicz korelację Pearsona na rangach

Spearman łapie dowolną monotoniczną zależność, nie tylko liniową.
Jeśli y = x^3, Pearson daje r < 1, ale Spearman daje rho = 1.
```

**Kiedy używać której:**

```
Pearson:    Obie zmienne ciągłe i w przybliżeniu normalne.
            Interesuje cię konkretnie liniowa zależność.
            Brak ekstremalnych wartości odstających.

Spearman:   Dane porządkowe (rankingi, oceny).
            Dane nie są normalnie rozłożone.
            Podejrzewasz monotoniczną, ale nie liniową zależność.
            Wartości odstające są obecne.
```

**Złota zasada:** korelacja nie implikuje przyczynowości. Sprzedaż lodów i utonięcia są skorelowane, bo oba rosną latem. Dokładność twojego modelu i liczba parametrów są skorelowane, ale dodawanie parametrów nie automatycznie poprawia dokładności (patrz: przeuczenie).

### Macierz kowariancji

Kowariancja między dwiema zmiennymi mierzy, jak zmieniają się razem:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X i Y mają tendencję do wzrostu razem
Cov(X, Y) < 0:  gdy X rośnie, Y ma tendencję do malenia
Cov(X, Y) = 0:  brak liniowej współzmienności
```

Dla d cech macierz kowariancji C to macierz d x d, gdzie C[i][j] = Cov(cecha_i, cecha_j). Elementy diagonalne C[i][i] to wariancje każdej cechy.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Własności:
  - Symetryczna: C[i][j] = C[j][i]
  - Dodatnio półokreślona: wszystkie wartości własne >= 0
  - Diagonala = wariancje
  - Poza diagonalą = kowariancje
```

**Związek z PCA.** PCA wykonuje rozkład na wartości własne macierzy kowariancji. Wektory własne to główne składowe (kierunki maksymalnej wariancji). Wartości własne mówią, ile wariancji przechwytuje każda składowa. To dokładnie to, co Lekcja 10 omówiła, ale teraz widzisz, dlaczego macierz kowariancji jest właściwą rzeczą do dekompozycji: koduje wszystkie liniowe zależności parami w twoich danych.

**Związek z korelacją.** Macierz korelacji to macierz kowariancji standaryzowanych zmiennych (każda podzielona przez swoje odchylenie standardowe). Korelacja normalizuje kowariancję, tak by wszystkie wartości mieściły się w [-1, 1].

### Testowanie hipotez

Testowanie hipotez to ramy do podejmowania decyzji w warunkach niepewności. Zaczynasz od twierdzenia, zbierasz dane i określasz, czy dane są zgodne z twierdzeniem.

**Konfiguracja:**

```
Hipoteza zerowa (H0):        domyślne założenie, zwykle "brak efektu"
Hipoteza alternatywna (H1): co próbujesz pokazać

Przykład:
  H0: Model A i Model B mają tę samą dokładność
  H1: Model B ma wyższą dokładność niż Model A
```

**P-wartość** to prawdopodobieństwo zobaczenia danych tak ekstremalnych jak te, które zaobserwowałeś, zakładając że H0 jest prawdziwa. NIE jest to prawdopodobieństwo, że H0 jest prawdziwa. To najczęstsze nieporozumienie w statystyce.

```
p-wartość = P(dane tak ekstremalne | H0 jest prawdziwa)

Jeśli p-wartość < alfa (zwykle 0.05):
    Odrzuć H0. Wynik jest "statystycznie istotny."
Jeśli p-wartość >= alfa:
    Nie masz podstaw do odrzucenia H0.
    To NIE oznacza, że H0 jest prawdziwa.
```

**Przedziały ufności** dają zakres wiarygodnych wartości dla parametru:

```
95% przedział ufności dla średniej:
    x_bar +/- z * (s / sqrt(n))

gdzie z = 1.96 dla 95% ufności

Interpretacja: jeśli powtórzyłbyś ten eksperyment wiele razy, 95% obliczonych
przedziałów zawierałoby prawdziwą średnią. NIE oznacza to, że istnieje
95% prawdopodobieństwo, że prawdziwa średnia znajduje się w tym konkretnym przedziale.
```

Szerokość przedziału ufności mówi o precyzji. Szerokie przedziały oznaczają wysoką niepewność. Wąskie przedziały oznaczają, że twoja estymacja jest precyzyjna (ale niekoniecznie dokładna, jeśli dane są obciążone).

### T-test

T-test porównuje średnie. Istnieje kilka odmian.

**Jednopróbkowy t-test:** czy średnia populacji różni się od hipotetyzowanej wartości?

```
t = (x_bar - mu_0) / (s / sqrt(n))

stopnie swobody = n - 1
```

**Dwuwyjściowy t-test (niezależny):** czy dwie średnie grup są różne?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

To jest t-test Welcha, który nie zakłada równych wariancji.
Zawsze używaj Welcha, chyba że masz konkretny powód do równych wariancji.
```

**Parowany t-test:** gdy pomiary występują w parach (ten sam model oceniany na tych samych podziałach danych):

```
Oblicz d_i = x_i - y_i dla każdej pary
Następnie wykonaj jednopróbkowy t-test na wartościach d_i względem mu_0 = 0
```

W ML parowany t-test jest powszechny: uruchamiasz oba modele na tych samych 10 fałdach walidacji krzyżowej i porównujesz ich wyniki parami.

### Test chi-kwadrat

Test chi-kwadrat sprawdza, czy zaobserwowane częstości zgadzają się z oczekiwanymi. Przydatny dla danych kategorycznych.

```
chi^2 = sum((zaobserwowane - oczekiwane)^2 / oczekiwane)

Przykład: czy rozkład wyjść modelu językowego pasuje do
rozkładu treningowego w poprzek kategorii?

Kategoria    Zaobserwowane   Oczekiwane
Pozytywny       120            100
Negatywny        80            100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

Z 1 stopniem swobody, chi^2 = 8 daje p < 0.005.
Różnica jest istotna.
```

### Testy A/B dla modeli ML

Testy A/B w ML nie są takie same jak testy A/B w internecie. Porównywanie modeli ma specyficzne wyzwania:

```
1. Ten sam zestaw testowy:   Oba modele muszą być oceniane na identycznych danych.
                             Różne zestawy testowe czynią porównanie bez znaczenia.

2. Wiele metryk:             Sama dokładność nie wystarczy. Potrzebujesz precyzji,
                             czułości, F1, opóźnienia i metryk uczciwości.

3. Wariancja:                Użyj walidacji krzyżowej lub bootstrapu do oszacowania
                             wariancji każdej metryki, nie tylko estymacji punktowych.

4. Wyciek danych:            Jeśli zestaw testowy był używany podczas selekcji modelu,
                             twoje porównanie jest obciążone. Odłóż końcowy zestaw testowy.
```

**Procedura:**

```
1. Zdefiniuj metrykę i poziom istotności (alfa = 0.05)
2. Uruchom oba modele na tych samych k fałdach walidacji krzyżowej
3. Zbierz sparowane wyniki: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Oblicz różnice: d_i = b_i - a_i
5. Wykonaj parowany t-test na różnicach
6. Sprawdź: czy średnia różnica jest istotnie różna od 0?
7. Oblicz przedział ufności dla średniej różnicy
8. Oblicz wielkość efektu (d Cohena), by ocenić praktyczną istotność
```

### Istotność statystyczna vs praktyczna

Wynik może być statystycznie istotny, ale praktycznie bez znaczenia. Z wystarczającą ilością danych nawet trywialna różnica staje się statystycznie istotna.

```
Przykład:
  Dokładność modelu A: 0.9234
  Dokładność modelu B: 0.9237
  n = 1 000 000 próbek testowych
  p-wartość = 0.001

Statystycznie istotne? Tak.
Praktycznie istotne? Poprawa o 0.03% nie jest warta
kosztu inżynieryjnego wdrożenia nowego modelu.
```

**Wielkość efektu** określa, jak duża jest różnica, niezależnie od wielkości próbki:

```
d Cohena = (średnia_1 - średnia_2) / połączone_std

d = 0.2:  mały efekt
d = 0.5:  średni efekt
d = 0.8:  duży efekt
```

Zawsze raportuj zarówno p-wartość, jak i wielkość efektu. P-wartość mówi, czy różnica jest prawdziwa. Wielkość efektu mówi, czy ma znaczenie.

### Problem wielokrotnych porównań

Gdy testujesz wiele hipotez, niektóre będą "istotne" przez przypadek. Jeśli testujesz 20 rzeczy przy alfa = 0.05, oczekujesz 1 fałszywie pozytywnego nawet gdy nic nie jest prawdziwe.

```
P(przynajmniej jeden fałszywie pozytywny) = 1 - (1 - alfa)^m

m = 20 testów, alfa = 0.05:
P(fałszywie pozytywny) = 1 - 0.95^20 = 0.64

Masz 64% szansy na przynajmniej jeden fałszywie pozytywny.
```

**Korekcja Bonferroniego:** podziel alfa przez liczbę testów.

```
Dostosowane alfa = alfa / m = 0.05 / 20 = 0.0025

Odrzuć H0 tylko jeśli p-wartość < 0.0025.
Konserwatywna, ale prosta. Działa, gdy testy są niezależne.
```

W ML ma to znaczenie, gdy porównujesz model przez wiele metryk, testujesz wiele konfiguracji hiperparametrów lub oceniasz na wielu zbiorach danych.

### Metody bootstrapowe

Bootstrapping estymuje rozkład z próbkowania statystyki przez losowanie z próbki ze zwracaniem. Żadnych założeń o podstawowym rozkładzie.

**Algorytm:**

```
1. Masz n punktów danych
2. Losuj n próbek ZE ZWROTem (niektóre punkty pojawiają się wiele razy, niektóre wcale)
3. Oblicz swoją statystykę na tej bootstrapowej próbce
4. Powtórz B razy (zwykle B = 1000 do 10000)
5. Rozkład bootstrapowych statystyk przybliża rozkład z próbkowania
```

**Bootstrapowy przedział ufności (metoda percentylowa):**

```
Posortuj B bootstrapowych statystyk
95% przedział ufności = [2.5 percentyl, 97.5 percentyl]
```

**Dlaczego bootstrap ma znaczenie dla ML:**

```
- Dokładność na zbiorze testowym to estymacja punktowa. Bootstrap daje przedziały ufności.
- Nie możesz zakładać, że rozkłady metryk są normalne (szczególnie dla AUC, F1, precyzji przy k).
- Bootstrap działa dla DOWOLNEJ statystyki: mediany, stosunku dwóch średnich, różnicy w AUC między dwoma modelami.
- Żaden wzór w postaci zamkniętej nie jest potrzebny.
```

**Bootstrap do porównywania modeli:**

```
1. Masz predykcje z Modelu A i Modelu B na tym samym zestawie testowym
2. Dla każdej iteracji bootstrapu:
   a. Losuj indeksy testowe ze zwracaniem
   b. Oblicz metrykę_A i metrykę_B na przelotowanym zbiorze
   c. Zapisz diff = metryka_B - metryka_A
3. 95% przedział ufności dla różnicy:
   [2.5 percentyl diff, 97.5 percentyl diff]
4. Jeśli przedział ufności nie zawiera 0, różnica jest istotna
```

To jest bardziej niezawodne niż parowany t-test, ponieważ nie robi założeń dystrybucyjnych.

### Testy parametryczne vs nieparametryczne

**Testy parametryczne** zakładają konkretny rozkład (zwykle normalny):

```
t-test:         zakłada normalnie rozłożone dane (lub duże n przez CTG)
ANOVA:          zakłada normalność i równe wariancje
r Pearsona:     zakłada dwuwymiarową normalność
```

**Testy nieparametryczne** nie robią założeń dystrybucyjnych:

```
Mann-Whitney U:     porównuje dwie grupy (zastępuje niezależny t-test)
Wilcoxon signed-rank: porównuje sparowane dane (zastępuje parowany t-test)
rho Spearmana:       korelacja na rangach (zastępuje Pearsona)
Kruskal-Wallis:      porównuje wiele grup (zastępuje ANOVA)
```

**Kiedy używać nieparametrycznych:**

```
- Mała liczebność próbki (n < 30) i dane wyraźnie nienormalne
- Dane porządkowe (oceny, rankingi)
- Ciężkie wartości odstające, których nie możesz usunąć
- Skośne rozkłady
```

**Kiedy używać parametrycznych:**

```
- Duża liczebność próbki (CTG czyni statystykę testową w przybliżeniu normalną)
- Dane w przybliżeniu symetryczne bez ekstremalnych wartości odstających
- Większa moc statystyczna (lepsza w wykrywaniu prawdziwych różnic)
```

W eksperymentach ML zazwyczaj masz małe n (5 lub 10 fałd walidacji krzyżowej), więc testy nieparametryczne jak Wilcoxon signed-rank są często bardziej odpowiednie niż t-testy.

### Centralne Twierdzenie Graniczne: implikacje praktyczne

CTG mówi, że rozkład średnich z próbek zbliża się do rozkładu normalnego wraz ze wzrostem n, niezależnie od podstawowego rozkładu populacji.

```
Jeśli X_1, X_2, ..., X_n są iid ze średnią mu i wariancją sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    gdy n -> nieskończoność

Działa dla n >= 30 w większości przypadków.
Dla silnie skośnych rozkładów możesz potrzebować n >= 100.
```

**Dlaczego to ma znaczenie dla ML:**

```
1. Uzasadnia przedziały ufności i t-testy na zagregowanych metrykach
2. Wyjaśnia, dlaczego uśrednianie przez fałdy walidacji krzyżowej daje stabilne
   oszacowania, nawet gdy poszczególne fałdy różnią się znacznie
3. Mini-wsadowy spadek gradientowy działa, bo średni gradient nad wsadem
   przybliża prawdziwy gradient (CTG w działaniu)
4. Metody zespołowe: uśrednianie predykcji z wielu modeli daje
   bardziej stabilne wyjście niż każdy pojedynczy model
```

**Czego CTG NIE robi:**

```
- NIE czyni twoich danych normalnymi. Czyni normalną ŚREDNIĄ z próbek.
- NIE działa dla rozkładów z ciężkimi ogonami z nieskończoną wariancją (rozkład Cauchy'ego).
- NIE stosuje się do zależnych danych (szeregi czasowe bez korekty).
```

### Typowe błędy statystyczne w pracach ML

1. **Testowanie na zbiorze treningowym.** Gwarantuje przeuczenie. Zawsze odłóż dane, których model nigdy nie widział podczas trenowania.

2. **Brak przedziałów ufności.** Raportowanie pojedynczej liczby dokładności bez niepewności czyni wyniki nieodtwarzalnymi i nieweryfikowalnymi.

3. **Ignorowanie wielokrotnych porównań.** Testowanie 50 konfiguracji i raportowanie najlepszej bez korekty influje wskaźniki fałszywie pozytywnych.

4. **Mylenie istotności statystycznej i praktycznej.** P-wartość 0.001 przy poprawie dokładności o 0.01% nie jest znacząca.

5. **Używanie dokładności na niezbalansowanych danych.** 99% dokładności na zbiorze z 99% klasy negatywnej oznacza, że model niczego się nie nauczył. Użyj precyzji, czułości, F1 lub AUC.

6. **Wybiórcze raportowanie metryk.** Raportowanie tylko metryki, w której twój model wygrywa. Uczciwa ocena raportuje wszystkie istotne metryki.

7. **Przeciek informacji między podziałami trening/test.** Normalizowanie przed podziałem lub używanie przyszłych danych do przewidywania przeszłości.

8. **Małe zbiory testowe bez estymacji wariancji.** Ocena na 100 próbkach i twierdzenie o 2% poprawie to szum, nie sygnał.

9. **Zakładanie niezależności, gdy dane nie są niezależne.** Obrazy medyczne od tego samego pacjenta, wiele zdań z tego samego dokumentu. Obserwacje w grupie są skorelowane.

10. **P-hacking.** Próbowanie różnych testów, podzbiorów lub kryteriów wykluczenia, aż dostaniesz p < 0.05. Wynik jest artefaktem przeszukiwania.

## Building It

Zaimplementujesz:

1. **Statystyki opisowe od podstaw** (średnia, mediana, moda, odchylenie standardowe, percentyle, IQR)
2. **Funkcje korelacji** (Pearson i Spearman, z macierzą kowariancji)
3. **Testy hipotez** (jednopróbkowy t-test, dwuwyjściowy t-test, test chi-kwadrat)
4. **Bootstrapowe przedziały ufności** (dla dowolnej statystyki, bez założeń)
5. **Symulator testów A/B** (generuj dane, testuj, sprawdzaj błędy typu I i II)
6. **Demo istotności statystycznej vs praktycznej** (pokazujące, że duże n czyni wszystko "istotnym")

Wszystko od podstaw, używając tylko `math` i `random`. Bez numpy, bez scipy.

## Key Terms

| Termin | Definicja |
|---|---|
| Średnia | Suma wartości podzielona przez liczbę. Wrażliwa na wartości odstające. |
| Mediana | Środkowa wartość posortowanych danych. Odporna na wartości odstające. |
| Odchylenie standardowe | Pierwiastek kwadratowy z wariancji. Mierzy rozrzut w oryginalnych jednostkach. |
| Percentyl | Wartość, poniżej której znajduje się dany procent danych. |
| IQR | Rozstęp międzykwartylowy. Q3 minus Q1. Rozrzut środkowych 50%. |
| Korelacja Pearsona | Mierzy liniową zależność między dwiema zmiennymi. Zakres [-1, 1]. |
| Korelacja Spearmana | Mierzy monotoniczną zależność używając rang. |
| Macierz kowariancji | Macierz kowariancji parami między wszystkimi cechami. |
| Hipoteza zerowa | Domyślne założenie braku efektu lub różnicy. |
| P-wartość | Prawdopodobieństwo tak ekstremalnych danych, jeśli hipoteza zerowa jest prawdziwa. |
| Przedział ufności | Zakres wiarygodnych wartości dla parametru przy danym poziomie ufności. |
| t-test | Testuje, czy średnie różnią się istotnie. Używa rozkładu t. |
| Test chi-kwadrat | Testuje, czy zaobserwowane częstości różnią się od oczekiwanych. |
| Wielkość efektu | Wielkość różnicy, niezależna od wielkości próbki. d Cohena jest powszechne. |
| Korekcja Bonferroniego | Dzieli próg istotności przez liczbę testów, by kontrolować fałszywie pozytywne. |
| Bootstrap | Losowanie ze zwracaniem w celu estymacji rozkładów z próbkowania. |
| Błąd typu I | Fałszywie pozytywny. Odrzucenie H0, gdy jest prawdziwa. |
| Błąd typu II | Fałszywie negatywny. Nieodrzucenie H0, gdy jest fałszywa. |
| Moc statystyczna | Prawdopodobieństwo poprawnego odrzucenia fałszywej H0. Moc = 1 minus wskaźnik błędu typu II. |
| Centralne Twierdzenie Graniczne | Średnie z próbek zbiegają do rozkładu normalnego wraz ze wzrostem liczebności próbki. |
| Test parametryczny | Zakłada konkretny rozkład danych (zwykle normalny). |
| Test nieparametryczny | Nie robi założeń dystrybucyjnych. Działa na rangach lub znakach. |