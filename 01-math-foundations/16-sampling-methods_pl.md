# Metody próbkowania

> Próbkowanie to sposób, w jaki AI eksploruje przestrzeń możliwości.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minut

## Learning Objectives

- Zaimplementuj próbkowanie przez odwróconą dystrybuantę, przez odrzucanie i ważne od podstaw używając tylko jednostajnych liczb losowych
- Zbuduj próbkowanie z temperaturą, top-k i top-p (jądrowe) do generowania tokenów modeli językowych
- Wyjaśnij sztuczkę reparametryzacji i dlaczego umożliwia wsteczną propagację przez próbkowanie w VAE
- Uruchom MCMC Metropolisa-Hastingsa do próbkowania z nienormalizowanego rozkładu docelowego

## Problem

Model językowy kończy przetwarzanie twojego promptu i produkuje wektor 50 000 logitów. Jeden dla każdego tokena w swoim słowniku. Teraz musi wybrać jeden. Jak?

Jeśli zawsze wybiera token o najwyższym prawdopodobieństwie, każda odpowiedź jest identyczna. Deterministyczna. Nudna. Jeśli wybiera jednostajnie losowo, wyjście jest bełkotem. Odpowiedź leży gdzieś między tymi ekstremami, a to gdzieś jest kontrolowane przez próbkowanie.

Próbkowanie nie ogranicza się do generowania tekstu. Uczenie przez wzmacnianie estymuje gradienty polityki przez próbkowanie trajektorii. VAE uczą się reprezentacji latentnych przez próbkowanie z nauczonych rozkładów i wsteczną propagację przez losowość. Modele dyfuzyjne generują obrazy przez próbkowanie szumu i iteracyjne odszumianie. Metody Monte Carlo estymują całki, które nie mają rozwiązań w postaci zamkniętej. Algorytmy MCMC eksplorują wysokowymiarowe rozkłady a posteriori, których nie da się wyliczyć.

Każdy generatywny system AI jest systemem próbkującym. Strategia próbkowania determinuje jakość, różnorodność i sterowalność wyjścia. Ta lekcja buduje każdą główną metodę próbkowania od podstaw, zaczynając od jednostajnych liczb losowych, a kończąc na technikach napędzających nowoczesne LLM i modele generatywne.

## Koncepcja

### Dlaczego próbkowanie ma znaczenie

Próbkowanie pojawia się w czterech fundamentalnych rolach w AI i uczeniu maszynowym:

**Generacja.** Modele językowe, modele dyfuzyjne i GAN wszystkie produkują wyjście przez próbkowanie. Algorytm próbkowania bezpośrednio kontroluje kreatywność, spójność i różnorodność. Temperatura, top-k i próbkowanie jądrowe to pokrętła, którymi inżynierowie kręcą codziennie.

**Trenowanie.** Stochastyczny spadek gradientowy próbkuje mini-wsady. Dropout losuje neurony do dezaktywacji. Augmentacja danych losuje losowe przekształcenia. Próbkowanie ważne przeważa próbki, by zmniejszyć wariancję gradientu w uczeniu przez wzmacnianie (PPO, TRPO).

**Estymacja.** Wiele wielkości w ML nie ma rozwiązań w postaci zamkniętej. Oczekiwana strata nad rozkładem danych, funkcja partycji modelu opartego na energii, dowód we wnioskowaniu bayesowskim. Estymacja Monte Carlo przybliża wszystkie te przez uśrednianie po próbkach.

**Eksploracja.** Algorytmy MCMC eksplorują rozkłady a posteriori we wnioskowaniu bayesowskim. Strategie ewolucyjne próbkują perturbacje parametrów. Próbkowanie Thompsona równoważy eksplorację i eksploatację w algorytmach typu bandyta.

Główne wyzwanie: możesz próbkować bezpośrednio tylko z prostych rozkładów (jednostajny, normalny). Dla wszystkiego innego potrzebujesz metody konwersji prostych próbek na próbki z twojego docelowego rozkładu.

### Jednostajne próbkowanie losowe

Każda metoda próbkowania zaczyna się tutaj. Generator jednostajnych liczb losowych produkuje wartości w [0, 1), gdzie każdy podprzedział o równej długości ma równe prawdopodobieństwo.

```
U ~ Jednostajny(0, 1)

P(a <= U <= b) = b - a    dla 0 <= a <= b <= 1

Własności:
  E[U] = 0.5
  Var(U) = 1/12
```

Aby próbkować jednostajnie z dyskretnego zbioru n elementów, wygeneruj U i zwróć floor(n * U). Aby próbkować z ciągłego zakresu [a, b], oblicz a + (b - a) * U.

Kluczową intuicją jest: pojedyncza jednostajna liczba losowa zawiera dokładnie tyle losowości, ile potrzeba do wyprodukowania jednej próbki z dowolnego rozkładu. Sztuczką jest znalezienie odpowiedniego przekształcenia.

### Metoda odwróconej dystrybuanty (próbkowanie przez przekształcenie odwrotne)

Dystrybuanta (CDF) mapuje wartości na prawdopodobieństwa:

```
F(x) = P(X <= x)

Własności:
  F jest niemalejąca
  F(-inf) = 0
  F(+inf) = 1
  F mapuje prostą rzeczywistą na [0, 1]
```

Odwrócona dystrybuanta mapuje prawdopodobieństwa z powrotem na wartości. Jeśli U ~ Jednostajny(0, 1), to X = F_odwrotne(U) podlega docelowemu rozkładowi.

```
Algorytm:
  1. Wygeneruj u ~ Jednostajny(0, 1)
  2. Zwróć F_odwrotne(u)

Dlaczego działa:
  P(X <= x) = P(F_odwrotne(U) <= x) = P(U <= F(x)) = F(x)
```

**Przykład rozkładu wykładniczego:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Rozwiąż F(x) = u dla x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Ponieważ (1 - U) i U mają ten sam rozkład:
  x = -ln(u) / lambda
```

Działa doskonale, gdy możesz zapisać F_odwrotne w postaci zamkniętej. Dla rozkładu normalnego nie ma odwróconej dystrybuanty w postaci zamkniętej, więc używamy innych metod (Boxa-Mullera lub przybliżenia numerycznego).

**Wersja dyskretna:** Dla dyskretnych rozkładów zbuduj CDF jako sumę skumulowaną, wygeneruj U i znajdź pierwszy indeks, gdzie suma skumulowana przekracza U. Tak działa `sample_categorical` z Lekcji 06.

### Próbkowanie przez odrzucanie

Gdy nie możesz odwrócić CDF, ale możesz obliczyć docelową PDF z dokładnością do stałej, działa próbkowanie przez odrzucanie.

```
Rozkład docelowy: p(x)  (można obliczyć, możliwie nienormalizowany)
Rozkład proponujący: q(x)  (można z niego próbkować)
Ograniczenie: M takie, że p(x) <= M * q(x) dla wszystkich x

Algorytm:
  1. Próbkuj x ~ q(x)
  2. Próbkuj u ~ Jednostajny(0, 1)
  3. Jeśli u < p(x) / (M * q(x)), zaakceptuj x
  4. W przeciwnym razie odrzuć i wróć do kroku 1

Współczynnik akceptacji = 1/M
```

Im ciaśniejsze ograniczenie M, tym wyższy współczynnik akceptacji. W niskich wymiarach (1-3) próbkowanie przez odrzucanie działa dobrze. W wysokich wymiarach współczynnik akceptacji spada wykładniczo, ponieważ większość objętości proponującej jest odrzucana. To przekleństwo wymiarowości dla próbkowania przez odrzucanie.

**Przykład: próbkowanie z obciętego rozkładu normalnego.** Użyj jednostajnego rozkładu proponującego nad obciętym zakresem. Obwiednia M to maksimum normalnej PDF w tym zakresie.

**Przykład: próbkowanie z półokręgu.** Proponuj jednostajnie w ograniczającym prostokącie. Zaakceptuj, jeśli punkt wpada do półokręgu. Tak Monte Carlo oblicza pi: współczynnik akceptacji równa się stosunkowi pól pi/4.

### Próbkowanie ważne

Czasami nie potrzebujesz próbek z docelowego rozkładu p(x). Potrzebujesz oszacować wartość oczekiwaną pod p(x) i masz próbki z innego rozkładu q(x).

```
Cel: oszacuj E_p[f(x)] = całka z f(x) * p(x) dx

Przepisz:
  E_p[f(x)] = całka z f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

gdzie w(x) = p(x) / q(x)  to wagi ważności.

Estymator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    gdzie x_i ~ q(x)
```

To jest krytyczne w uczeniu przez wzmacnianie. W PPO (Proximal Policy Optimization) zbierasz trajektorie pod starą polityką pi_stara, ale chcesz optymalizować nową politykę pi_nowa. Waga ważności to pi_nowa(a|s) / pi_stara(a|s). PPO obcina te wagi, by zapobiec zbytniemu oddaleniu się nowej polityki od starej.

Wariancja estymatora próbkowania ważnego zależy od tego, jak podobne są q do p. Jeśli q jest bardzo różne od p, kilka próbek dostaje ogromne wagi i dominuje estymację. Samonormalizujące się próbkowanie ważne dzieli przez sumę wag, by zmniejszyć ten problem:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Estymacja Monte Carlo

Estymacja Monte Carlo przybliża całki przez uśrednianie losowych próbek. Prawo wielkich liczb gwarantuje zbieżność.

```
Cel: oszacuj I = całka z g(x) dx nad dziedziną D

Metoda:
  1. Próbkuj x_1, ..., x_N jednostajnie z D
  2. I ~ (Objętość D / N) * sum(g(x_i))

Błąd: O(1 / sqrt(N))   niezależnie od wymiaru
```

Wskaźnik błędu jest niezależny od wymiaru. Dlatego metody Monte Carlo dominują w wysokich wymiarach, gdzie całkowanie na siatce jest niemożliwe.

**Estymacja pi:**

```
Próbkuj (x, y) jednostajnie z [-1, 1] x [-1, 1]
Policz, ile wpada do okręgu jednostkowego: x^2 + y^2 <= 1
pi ~ 4 * (liczba_w_środku) / (całkowita_liczba)
```

**Estymacja wartości oczekiwanych:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    gdzie x_i ~ p(x)

Średnia z próbki zbiega do prawdziwej wartości oczekiwanej.
Wariancja estymatora = Var(f(X)) / N
```

### Łańcuch Markowa Monte Carlo (MCMC): Metropolis-Hastings

MCMC konstruuje łańcuch Markowa, którego rozkład stacjonarny to docelowy rozkład p(x). Po wystarczającej liczbie kroków próbki z łańcucha są (w przybliżeniu) próbkami z p(x).

```
Cel: p(x)  (znany z dokładnością do stałej normalizującej)
Propozycja: q(x'|x)  (jak zaproponować następny stan na podstawie bieżącego)

Algorytm Metropolisa-Hastingsa:
  1. Zacznij w pewnym x_0
  2. Dla t = 1, 2, ..., T:
     a. Zaproponuj x' ~ q(x'|x_t)
     b. Oblicz współczynnik akceptacji:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Zaakceptuj z prawdopodobieństwem min(1, alpha):
        - Jeśli u < alpha (u ~ Jednostajny(0,1)): x_{t+1} = x'
        - W przeciwnym razie: x_{t+1} = x_t
  3. Odrzuć pierwsze B próbek (burn-in)
  4. Zwróć pozostałe próbki
```

Dla symetrycznych propozycji (q(x'|x) = q(x|x')), stosunek upraszcza się do p(x')/p(x). To oryginalny algorytm Metropolisa.

**Dlaczego działa.** Reguła akceptacji zapewnia równowagę szczegółową: prawdopodobieństwo bycia w x i przejścia do x' równa się prawdopodobieństwu bycia w x' i przejścia do x. Równowaga szczegółowa implikuje, że p(x) jest rozkładem stacjonarnym łańcucha.

**Praktyczne uwagi:**
- Burn-in: odrzuć wczesne próbki, zanim łańcuch osiągnie równowagę
- Rozrzedzanie: zachowaj co k-tą próbkę, by zmniejszyć autokorelację
- Skala propozycji: zbyt mała a łańcuch porusza się wolno (wysoka akceptacja, wolna eksploracja); zbyt duża a większość propozycji jest odrzucana (niska akceptacja, utknięcie w miejscu)
- Optymalny wskaźnik akceptacji dla Gaussa w wysokich wymiarach to około 0.234

### Próbkowanie Gibbsa

Próbkowanie Gibbsa to szczególny przypadek MCMC dla wielowymiarowych rozkładów. Zamiast proponować ruch we wszystkich wymiarach naraz, aktualizuje jedną zmienną na raz z jej rozkładu warunkowego.

```
Cel: p(x_1, x_2, ..., x_d)

Algorytm:
  Dla każdej iteracji t:
    Próbkuj x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Próbkuj x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Próbkuj x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Próbkowanie Gibbsa wymaga, byś mógł próbkować z każdego rozkładu warunkowego p(x_i | x_{-i}). Jest to proste dla wielu modeli:
- Sieci bayesowskie: rozkłady warunkowe wynikają ze struktury grafu
- Mieszanki Gaussów: rozkłady warunkowe są Gaussa
- Modele Isinga: warunkowy każdy spin zależy tylko od sąsiadów

Wskaźnik akceptacji wynosi zawsze 1 (każda propozycja jest przyjęta), ponieważ próbkowanie z dokładnego rozkładu warunkowego automatycznie spełnia równowagę szczegółową.

**Ograniczenie.** Gdy zmienne są silnie skorelowane, próbkowanie Gibbsa miesza się wolno, ponieważ aktualizowanie jednej zmiennej na raz nie może robić dużych diagonalnych ruchów przez rozkład.

### Próbkowanie z temperaturą (używane w LLM)

Modele językowe wyjściowo dają logity z_1, ..., z_V dla każdego tokena w słowniku. Softmax przekształca je na prawdopodobieństwa. Temperatura przeskalowuje logity przed softmaxem:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standardowy softmax (oryginalny rozkład)
T -> 0:  argmax (deterministyczny, zawsze wybiera najwyższy logit)
T -> inf: jednostajny (wszystkie tokeny jednakowo prawdopodobne)
T < 1.0: wyostrza rozkład (bardziej pewny, mniej różnorodny)
T > 1.0: spłaszcza rozkład (mniej pewny, bardziej różnorodny)
```

**Dlaczego działa.** Dzielenie logitów przez T < 1 amplifikuje różnice między logitami. Jeśli z_1 = 2 i z_2 = 1, dzielenie przez T = 0.5 daje z_1/T = 4 i z_2/T = 2, czyniąc różnicę większą. Po softmaxie token z najwyższym logitem dostaje znacznie większy udział.

**W praktyce:**
- T = 0.0: dekodowanie zachłanne, najlepsze do faktograficznych Q&A
- T = 0.3-0.7: lekko kreatywne, dobre do generowania kodu
- T = 0.7-1.0: zrównoważone, dobre do ogólnej konwersacji
- T = 1.0-1.5: kreatywne pisanie, burza mózgów
- T > 1.5: coraz bardziej losowe, rzadko przydatne

Temperatura nie zmienia, które tokeny są możliwe. Zmienia masę prawdopodobieństwa przydzieloną każdemu tokenowi.

### Próbkowanie top-k

Próbkowanie top-k ogranicza zbiór kandydatów do k tokenów z najwyższymi prawdopodobieństwami, a następnie renormalizuje i próbkuje z tego ograniczonego zbioru.

```
Algorytm:
  1. Oblicz prawdopodobieństwa softmax dla wszystkich V tokenów
  2. Posortuj tokeny według prawdopodobieństwa (malejąco)
  3. Zachowaj tylko top k tokenów
  4. Renormalizuj: p_i' = p_i / sum(p_j dla j w top-k)
  5. Próbkuj z renormalizowanego rozkładu

k = 1:  dekodowanie zachłanne
k = V:  bez filtrowania (standardowe próbkowanie)
k = 40: typowe ustawienie, usuwa długi ogon mało prawdopodobnych tokenów
```

Top-k zapobiega wybieraniu przez model skrajnie mało prawdopodobnych tokenów (literówki, nonsensy), które istnieją w długim ogonie rozkładu słownika. Problem: k jest stałe niezależnie od kontekstu. Gdy model jest pewny (jeden token ma 95% prawdopodobieństwa), k = 40 wciąż pozwala na 39 alternatyw. Gdy model jest niepewny (prawdopodobieństwo rozłożone na 1000 tokenów), k = 40 odcina prawdopodobne opcje.

### Próbkowanie top-p (jądrowe)

Próbkowanie top-p dynamicznie dostosowuje rozmiar zbioru kandydatów. Zamiast utrzymywać stałą liczbę tokenów, zachowuje najmniejszy zbiór tokenów, których skumulowane prawdopodobieństwo przekracza p.

```
Algorytm:
  1. Oblicz prawdopodobieństwa softmax dla wszystkich V tokenów
  2. Posortuj tokeny według prawdopodobieństwa (malejąco)
  3. Znajdź najmniejsze k takie, że suma top-k prawdopodobieństw >= p
  4. Zachowaj tylko te k tokenów
  5. Renormalizuj i próbkuj

p = 0.9:  zachowuje tokeny pokrywające 90% masy prawdopodobieństwa
p = 1.0:  bez filtrowania
p = 0.1:  bardzo restrykcyjne, prawie zachłanne
```

Gdy model jest pewny, próbkowanie jądrowe zachowuje mało tokenów (może 2-3). Gdy model jest niepewny, zachowuje wiele (może 200). To adaptacyjne zachowanie jest powodem, dla którego próbkowanie jądrowe generalnie produkuje lepszy tekst niż top-k.

**Typowe kombinacje:**
- Temperatura 0.7 + top-p 0.9: dobre ustawienie ogólnego przeznaczenia
- Temperatura 0.0 (zachłanne): najlepsze do deterministycznych zadań
- Temperatura 1.0 + top-k 50: ustawienie z oryginalnej pracy Fan et al. (2018)

Top-k i top-p można łączyć. Zastosuj top-k najpierw, potem top-p na pozostałym zbiorze.

### Sztuczka reparametryzacji (używana w VAE)

Wariacyjne autoenkodery (VAE) uczą się przez kodowanie wejść w rozkład w przestrzeni latentnej, próbkowanie z tego rozkładu i dekodowanie próbki z powrotem. Problem: nie możesz propagować wstecz przez operację próbkowania.

```
Standardowe próbkowanie (nie różniczkowalne):
  z ~ N(mu, sigma^2)

  Losowość blokuje przepływ gradientu.
  d/d_mu [próbka z N(mu, sigma^2)] = ???
```

Sztuczka reparametryzacji oddziela losowość od parametrów:

```
Reparametryzowane próbkowanie:
  epsilon ~ N(0, 1)          (stały losowy szum, bez parametrów)
  z = mu + sigma * epsilon   (deterministyczna funkcja parametrów)

  Teraz z jest deterministyczną, różniczkowalną funkcją mu i sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradienty płyną przez mu i sigma.
```

Działa to, ponieważ N(mu, sigma^2) ma ten sam rozkład co mu + sigma * N(0, 1). Kluczowa intuicja: przenieś losowość do źródła wolnego od parametrów (epsilon), a następnie wyraź próbkę jako różniczkowalną transformację parametrów.

**W pętli treningowej VAE:**
1. Enkoder wyjściowo daje mu i log(sigma^2) dla każdego wejścia
2. Próbkuj epsilon ~ N(0, 1)
3. Oblicz z = mu + sigma * epsilon
4. Dekoduj z, by odtworzyć wejście
5. Propaguj wstecz przez kroki 4, 3, 2, 1 (możliwe, bo krok 3 jest różniczkowalny)

Bez sztuczki reparametryzacji VAE nie mogą być trenowane standardową wsteczną propagacją. Ta pojedyncza intuicja uczyniła VAE praktycznymi.

### Gumbel-Softmax (różniczkowalne próbkowanie kategoryczne)

Sztuczka reparametryzacji działa dla rozkładów ciągłych (Gaussa). Dla dyskretnych rozkładów kategorycznych potrzebujemy innego podejścia. Gumbel-Softmax dostarcza różniczkowalnego przybliżenia kategorycznego próbkowania.

**Sztuczka Gumbel-Max (nie różniczkowalna):**

```
Aby próbkować z kategorycznego rozkładu z log-prawdopodobieństwami log(p_1), ..., log(p_k):
  1. Próbkuj g_i ~ Gumbel(0, 1) dla każdej kategorii
     (g = -log(-log(u)), gdzie u ~ Jednostajny(0, 1))
  2. Zwróć argmax(log(p_i) + g_i)

To produkuje dokładne kategoryczne próbki.
```

**Gumbel-Softmax (różniczkowalne przybliżenie):**

```
Zastąp twardy argmax miękkim softmaxem:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperatura) kontroluje przybliżenie:
  tau -> 0:  zbliża się do wektora one-hot (twardy kategoryczny)
  tau -> inf: zbliża się do jednostajnego (1/k, 1/k, ..., 1/k)
  tau = 1.0: miękkie przybliżenie
```

Gumbel-Softmax produkuje ciągłą relaksację dyskretnej próbki. Wyjście to wektor prawdopodobieństw (miękki one-hot) zamiast twardego one-hot. Gradienty płyną przez softmax. Podczas przejścia w przód w trenowaniu możesz użyć estymatora "straight-through": użyj twardego argmax dla przejścia w przód, ale miękkich gradientów Gumbel-Softmax dla przejścia wstecznego.

**Zastosowania:**
- Dyskretne zmienne latentne w VAE
- Wyszukiwanie architektury sieci neuronowych (wybór dyskretnych operacji)
- Mechanizmy twardej uwagi
- Uczenie przez wzmacnianie z dyskretnymi akcjami

### Próbkowanie warstwowe

Standardowe próbkowanie Monte Carlo może pozostawić luki w przestrzeni próbek przez przypadek. Próbkowanie warstwowe wymusza równomierne pokrycie przez podział przestrzeni na warstwy i próbkowanie z każdej.

```
Standardowe Monte Carlo:
  Próbkuj N punktów jednostajnie z [0, 1]
  Niektóre regiony mogą mieć skupiska, inne luki

Próbkowanie warstwowe:
  Podziel [0, 1] na N równych warstw: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Próbkuj jeden punkt jednostajnie w każdej warstwie
  x_i = (i + u_i) / N   gdzie u_i ~ Jednostajny(0, 1),  i = 0, ..., N-1
```

Próbkowanie warstwowe zawsze ma mniejszą lub równą wariancję w porównaniu do standardowego Monte Carlo:

```
Var(warstwowe) <= Var(standardowe Monte Carlo)

Poprawa jest największa, gdy f(x) zmienia się gładko.
Dla funkcji przedziałami stałych próbkowanie warstwowe jest dokładne.
```

**Zastosowania:**
- Całkowanie numeryczne (quasi-Monte Carlo)
- Podziały danych treningowych (zapewnienie równowagi klas w każdej fałdzie)
- Próbkowanie ważne z warstwowaniem (łącząc obie techniki)
- NeRF (Neural Radiance Fields) używa próbkowania warstwowego wzdłuż promieni kamery

### Związek z modelami dyfuzyjnymi

Modele dyfuzyjne generują obrazy przez proces próbkowania. Proces w przód dodaje szum Gaussa do obrazu przez T kroków, aż stanie się czystym szumem. Proces odwrotny uczy się odszumiać, odzyskując oryginalny obraz krok po kroku.

```
Proces w przód (znany):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  gdzie epsilon ~ N(0, I)

  Po T krokach: x_T ~ N(0, I)  (czysty szum)

Proces odwrotny (uczony):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  gdzie z ~ N(0, I)

  Każdy krok odszumiania to krok próbkowania.
```

Związek z metodami w tej lekcji:
- Każdy krok odszumiania używa sztuczki reparametryzacji (próbkuj szum, zastosuj deterministyczną transformację)
- Harmonogram szumu {alpha_t} kontroluje formę wyżarzania temperatury
- Trenowanie używa estymacji Monte Carlo do przybliżenia ELBO (dolnej granicy dowodu)
- Próbkowanie ancestralne w modelach dyfuzyjnych to łańcuch Markowa (każdy krok zależy tylko od bieżącego stanu)

Cały proces generowania obrazu to iteracyjne próbkowanie: zacznij od szumu, a na każdym kroku próbkuj nieco mniej zaszumioną wersję warunkową nauczonego modelu odszumiania.

```figure
monte-carlo-pi
```

## Build It

### Krok 1: Próbkowanie jednostajne i przez odwróconą dystrybuantę

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Wygeneruj 10 000 wykładniczych próbek i zweryfikuj, że średnia wynosi 1/lambda.

### Krok 2: Próbkowanie przez odrzucanie

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Użyj próbkowania przez odrzucanie, by losować z obciętego rozkładu normalnego. Zweryfikuj kształt przez histogram próbek.

### Krok 3: Próbkowanie ważne

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Oszacuj E[X^2] pod rozkładem normalnym używając jednostajnej propozycji. Porównaj ze znaną odpowiedzią (mu^2 + sigma^2).

### Krok 4: Estymacja Monte Carlo pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Krok 5: MCMC Metropolisa-Hastingsa

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Próbkuj z dwumodalnego rozkładu (mieszanka dwóch Gaussów). Zwizualizuj trajektorię łańcucha.

### Krok 6: Próbkowanie Gibbsa

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Krok 7: Próbkowanie z temperaturą

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Pokaż, jak temperatura zmienia rozkład wyjściowy dla zestawu logitów tokenów.

### Krok 8: Próbkowanie top-k i top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Krok 9: Sztuczka reparametryzacji

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Zademonstruj, że gradienty płyną przez reparametryzowaną próbkę, ale nie przez bezpośrednie próbkowanie.

### Krok 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Pokaż, jak malejąca temperatura sprawia, że wyjście zbliża się do wektora one-hot.

Pełne implementacje ze wszystkimi wizualizacjami są w `code/sampling.py`.

## Use It

Z NumPy i SciPy, produkcyjne wersje:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Średnia wykładnicza: {exponential_samples.mean():.4f} (oczekiwana 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF w 1.96: {normal.cdf(1.96):.4f}")
print(f"Odwrócona CDF w 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Indeks wybranego tokena: {token}")
```

Do MCMC na dużą skalę używaj dedykowanych bibliotek:
- PyMC: pełne modelowanie bayesowskie z NUTS (adaptacyjny HMC)
- emcee: zespołowy próbnik MCMC
- NumPyro/JAX: MCMC akcelerowany GPU

Zbudowałeś te od podstaw. Teraz wiesz, co robią wywołania bibliotek.

## Ćwiczenia

1. Zaimplementuj próbkowanie przez odwróconą dystrybuantę dla rozkładu Cauchy'ego. CDF to F(x) = 0.5 + arctan(x)/pi. Wygeneruj 10 000 próbek i wykreśl histogram względem prawdziwej PDF. Zwróć uwagę na ciężkie ogony (ekstremalne wartości daleko od środka).

2. Użyj próbkowania przez odrzucanie do generowania próbek z rozkładu Beta(2, 5) używając propozycji Jednostajny(0, 1). Wykreśl zaakceptowane próbki względem prawdziwej PDF Beta. Jaki jest teoretyczny wskaźnik akceptacji?

3. Oszacuj całkę z sin(x) od 0 do pi używając Monte Carlo z 1 000, 10 000 i 100 000 próbkami. Porównaj błąd na każdym poziomie. Zweryfikuj, że błąd skaluje się jak O(1/sqrt(N)).

4. Zaimplementuj Metropolisa-Hastingsa do próbkowania z 2D rozkładu p(x, y) proporcjonalnego do exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Wykreśl próbki i trajektorię łańcucha. Eksperymentuj z różnymi odchyleniami standardowymi propozycji.

5. Zbuduj pełne demo generowania tekstu: dla słownika 10 słów z logitami, wygeneruj sekwencje 20 tokenów używając (a) zachłannie, (b) temperatura=0.7, (c) top-k=3, (d) top-p=0.9. Porównaj różnorodność wyjść przez 5 uruchomień.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| Próbkowanie | "Losowanie wartości" | Generowanie wartości zgodnie z rozkładem prawdopodobieństwa. Mechanizm za wszystkimi generatywnymi AI |
| Rozkład jednostajny | "Wszystkie jednakowo prawdopodobne" | Każda wartość w [a, b] ma równą gęstość prawdopodobieństwa 1/(b-a). Punkt wyjścia dla wszystkich metod próbkowania |
| Odwrócona CDF | "Transformacja prawdopodobieństwa" | F_odwrotne(U) przekształca jednostajną próbkę w próbkę z dowolnego rozkładu ze znaną CDF. Dokładna i wydajna |
| Próbkowanie przez odrzucanie | "Proponuj i akceptuj/odrzucaj" | Generuj z prostej propozycji, akceptuj z prawdopodobieństwem proporcjonalnym do stosunku cel/propozycja. Dokładne, ale marnuje próbki |
| Próbkowanie ważne | "Przeważ próbki" | Oszacuj wartości oczekiwane pod p(x) używając próbek z q(x) przez ważenie każdej próbki przez p(x)/q(x). Rdzeń PPO w RL |
| Monte Carlo | "Średnia losowych próbek" | Przybliż całki jako średnie z próbek. Błąd O(1/sqrt(N)) niezależnie od wymiaru |
| MCMC | "Błądzenie losowe, które zbiega" | Skonstruuj łańcuch Markowa, którego rozkład stacjonarny jest celem. Metropolis-Hastings to podstawowy algorytm |
| Metropolis-Hastings | "Akceptuj pod górę, czasem w dół" | Proponuj ruchy, akceptuj na podstawie stosunku gęstości. Równowaga szczegółowa zapewnia zbieżność do rozkładu docelowego |
| Próbkowanie Gibbsa | "Jedna zmienna na raz" | Aktualizuj każdą zmienną z jej rozkładu warunkowego, utrzymując pozostałe stałe. 100% wskaźnik akceptacji |
| Temperatura | "Pokrętło pewności" | Dzieli logity przez T przed softmaxem. T<1 wyostrza (bardziej pewne), T>1 spłaszcza (bardziej różnorodne) |
| Próbkowanie top-k | "Zachowaj k najlepszych" | Wyzeruj wszystkie poza k najwyższymi tokenami, renormalizuj, próbkuj. Stały rozmiar zbioru kandydatów |
| Próbkowanie jądrowe (top-p) | "Zachowaj prawdopodobne" | Zachowaj najmniejszy zestaw tokenów, których skumulowane prawdopodobieństwo przekracza p. Adaptacyjny rozmiar zbioru kandydatów |
| Sztuczka reparametryzacji | "Przenieś losowość na zewnątrz" | Zapisz z = mu + sigma * epsilon gdzie epsilon ~ N(0,1). Czyni próbkowanie różniczkowalnym. Niezbędne do trenowania VAE |
| Gumbel-Softmax | "Miękkie próbkowanie kategoryczne" | Różniczkowalne przybliżenie kategorycznego próbkowania używając szumu Gumbel + softmax z temperaturą |
| Próbkowanie warstwowe | "Wymuszone pokrycie" | Podziel przestrzeń próbek na warstwy, próbkuj z każdej. Zawsze mniejsza wariancja niż naiwne Monte Carlo |
| Burn-in | "Okres rozgrzewki" | Początkowe próbki MCMC odrzucone, zanim łańcuch osiągnie rozkład stacjonarny |
| Równowaga szczegółowa | "Warunek odwracalności" | p(x) * T(x->y) = p(y) * T(y->x). Warunek wystarczający, by p było rozkładem stacjonarnym łańcucha Markowa |
| Próbkowanie dyfuzyjne | "Iteracyjne odszumianie" | Generuj dane, zaczynając od szumu i stosując uczone kroki odszumiania. Każdy krok to warunkowa operacja próbkowania |