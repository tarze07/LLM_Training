# Prawdopodobieństwo i rozkłady

> Prawdopodobieństwo to język, którym AI wyraża niepewność.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minut

## Learning Objectives

- Zaimplementuj PMF i PDF od podstaw dla rozkładów Bernoulliego, kategorycznego, Poissona, jednostajnego i normalnego
- Oblicz wartość oczekiwaną, wariancję i użyj Centralnego Twierdzenia Granicznego do wyjaśnienia, dlaczego rozkłady Gaussa dominują
- Zbuduj funkcje softmax i log-softmax ze sztuczką numerycznej stabilności (odejmij maksymalny logit)
- Oblicz stratę cross-entropii z logitów i połącz ją z ujemną log-wiarygodnością

## Problem

Klasyfikator wyjściowy `[0.03, 0.91, 0.06]`. Model językowy wybiera następne słowo spośród 50 000 kandydatów. Model dyfuzyjny generuje obrazy przez próbkowanie z nauczonych rozkładów. Wszystkie one to prawdopodobieństwo w działaniu.

Każda predykcja modelu to rozkład prawdopodobieństwa. Każda funkcja straty mierzy, jak daleko przewidywany rozkład jest od prawdziwego. Każdy krok treningowy dostosowuje parametry, by jeden rozkład upodobnić do drugiego. Bez prawdopodobieństwa nie możesz przeczytać żadnej pracy naukowej z ML, zdebugować żadnego modelu ani zrozumieć, dlaczego twoja strata treningowa to NaN.

## Koncepcja

### Zdarzenia, przestrzenie próbek i prawdopodobieństwo

Przestrzeń próbek S to zbiór wszystkich możliwych wyników. Zdarzenie to podzbiór przestrzeni próbek. Prawdopodobieństwo mapuje zdarzenia na liczby między 0 a 1.

```
Rzut monetą:
  S = {O, R}
  P(O) = 0.5,  P(R) = 0.5

Pojedynczy rzut kostką:
  S = {1, 2, 3, 4, 5, 6}
  P(parzyste) = P({2, 4, 6}) = 3/6 = 0.5
```

Trzy aksjomaty definiują całe prawdopodobieństwo:
1. P(A) >= 0 dla dowolnego zdarzenia A
2. P(S) = 1 (coś zawsze się zdarza)
3. P(A lub B) = P(A) + P(B), gdy A i B nie mogą wystąpić jednocześnie

Wszystko inne (twierdzenie Bayesa, wartości oczekiwane, rozkłady) wynika z tych trzech reguł.

### Prawdopodobieństwo warunkowe i niezależność

P(A|B) to prawdopodobieństwo A pod warunkiem, że B zaszło.

```
P(A|B) = P(A i B) / P(B)

Przykład: talia kart
  P(As | Figura) = P(As i Figura) / P(Figura)
                     = (4/52) / (12/52)
                     = 4/12 = 1/3
```

Dwa zdarzenia są niezależne, gdy wiedza o jednym nic nie mówi o drugim:

```
Niezależne:   P(A|B) = P(A)
Równoważne:   P(A i B) = P(A) * P(B)
```

Rzuty monetą są niezależne. Losowanie kart bez zwracania nie jest.

### Funkcje masy prawdopodobieństwa vs funkcje gęstości prawdopodobieństwa

Dyskretne zmienne losowe mają funkcję masy prawdopodobieństwa (PMF). Każdy wynik ma określone prawdopodobieństwo, które możesz odczytać bezpośrednio.

```
PMF: P(X = k)

Uczciwa kostka:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Suma wszystkich prawdopodobieństw = 1
```

Ciągłe zmienne losowe mają funkcję gęstości prawdopodobieństwa (PDF). Gęstość w pojedynczym punkcie nie jest prawdopodobieństwem. Prawdopodobieństwo pochodzi z całkowania gęstości po przedziale.

```
PDF: f(x)

P(a <= X <= b) = całka z f(x) od a do b

f(x) może być większa niż 1 (gęstość, nie prawdopodobieństwo)
całka od -inf do +inf z f(x) dx = 1
```

To rozróżnienie ma znaczenie w ML. Wyjścia klasyfikacji to PMF (dyskretne wybory). Przestrzenie latentne VAE używają PDF (ciągłe).

### Typowe rozkłady

**Bernoulli:** jedna próba, dwa wyniki. Modeluje klasyfikację binarną.

```
P(X = 1) = p
P(X = 0) = 1 - p
Średnia = p,  Wariancja = p(1-p)
```

**Kategoryczny:** jedna próba, k wyników. Modeluje klasyfikację wieloklasową (wyjście softmax).

```
P(X = i) = p_i,  gdzie suma p_i = 1
Przykład: P(kot) = 0.7,  P(pies) = 0.2,  P(ptak) = 0.1
```

**Jednostajny:** wszystkie wyniki jednakowo prawdopodobne. Używany do losowej inicjalizacji.

```
Dyskretny: P(X = k) = 1/n dla k w {1, ..., n}
Ciągły: f(x) = 1/(b-a) dla x w [a, b]
```

**Normalny (Gaussowski):** krzywa dzwonowa. Sparametryzowany przez średnią (mu) i wariancję (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standardowy normalny: mu = 0, sigma = 1
  68% danych w obrębie 1 sigma
  95% w obrębie 2 sigma
  99.7% w obrębie 3 sigma
```

**Poissona:** liczba rzadkich zdarzeń w ustalonym przedziale. Modeluje częstości zdarzeń.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Średnia = lambda,  Wariancja = lambda
```

### Wartość oczekiwana i wariancja

Wartość oczekiwana to ważony średni wynik.

```
Dyskretna:   E[X] = suma x_i * P(X = x_i)
Ciągła:  E[X] = całka x * f(x) dx
```

Wariancja mierzy rozrzut wokół średniej.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Odchylenie standardowe = sqrt(Var(X))
```

W ML wartość oczekiwana pojawia się jako funkcja straty (średnia strata po rozkładzie danych). Wariancja mówi o stabilności modelu. Wysoka wariancja gradientów oznacza głośne trenowanie.

### Rozkłady łączne i brzegowe

Rozkład łączny P(X, Y) opisuje dwie zmienne losowe razem.

Przykład łącznej PMF (X = pogoda, Y = parasol):

| | Y=0 (brak parasola) | Y=1 (parasol) | Brzegowy P(X) |
|---|---|---|---|
| X=0 (słońce) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (deszcz) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Brzegowy P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Rozkład brzegowy sumuje drugą zmienną:

```
P(X = x) = suma po wszystkich y z P(X = x, Y = y)
```

Sumy wierszy i kolumn w powyższej tabeli to rozkłady brzegowe.

### Dlaczego rozkład normalny pojawia się wszędzie

Centralne Twierdzenie Graniczne: suma (lub średnia) wielu niezależnych zmiennych losowych zbiega do rozkładu normalnego, niezależnie od pierwotnego rozkładu.

```
Rzut 1 kostką:  rozkład jednostajny (płaski)
Średnia z 2 kostek:  trójkątny (szczytowy)
Średnia z 30 kostek: prawie idealna krzywa dzwonowa

Działa to dla DOWOLNEGO rozkładu początkowego.
```

Dlatego:
- Błędy pomiarowe są w przybliżeniu normalne (wiele małych niezależnych źródeł)
- Inicjalizacje wag w sieciach neuronowych używają rozkładów normalnych
- Szum gradientowy w SGD jest w przybliżeniu normalny (suma wielu gradientów z próbek)
- Rozkład normalny to rozkład o maksymalnej entropii dla danej średniej i wariancji

### Log-prawdopodobieństwa

Surowe prawdopodobieństwa powodują problemy numeryczne. Mnożenie wielu małych prawdopodobieństw szybko powoduje niedomiar do zera.

```
P(zdanie) = P(słowo1) * P(słowo2) * ... * P(słowo_n)
           = 0.01 * 0.003 * 0.02 * ...
           -> 0.0 (niedomiar po ~30 wyrazach)
```

Log-prawdopodobieństwa rozwiązują to. Mnożenia stają się dodawaniami.

```
log P(zdanie) = log P(słowo1) + log P(słowo2) + ... + log P(słowo_n)
               = -4.6 + -5.8 + -3.9 + ...
               -> skończona liczba (brak niedomiaru)
```

Zasady:
- log(a * b) = log(a) + log(b)
- log-prawdopodobieństwa są zawsze <= 0 (ponieważ 0 < P <= 1)
- Bardziej ujemne = mniej prawdopodobne
- Strata cross-entropii to ujemne log-prawdopodobieństwo poprawnej klasy

### Softmax jako rozkład prawdopodobieństwa

Sieci neuronowe wyjściowo dają surowe wyniki (logity). Softmax przekształca je w poprawny rozkład prawdopodobieństwa.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) dla wszystkich j)

Własności:
  - Wszystkie wyjścia są w (0, 1)
  - Wszystkie wyjścia sumują się do 1
  - Zachowuje względne uporządkowanie wejść
  - exp() wzmacnia różnice między logitami
```

Sztuczka softmax: odejmij maksymalny logit przed potęgowaniem, aby zapobiec nadmiarowi.

```
z = [100, 101, 102]
exp(102) = nadmiar

z_przesunięte = z - max(z) = [-2, -1, 0]
exp(0) = 1  (bezpieczne)

Ten sam wynik, bez nadmiaru.
```

Log-softmax łączy softmax i log dla stabilności numerycznej. PyTorch używa tego wewnętrznie dla straty cross-entropii.

### Próbkowanie

Próbkowanie oznacza losowanie wartości z rozkładu. W ML:
- Dropout losowo wybiera, które neurony wyzerować
- Augmentacja danych losuje losowe przekształcenia
- Modele językowe próbkują następny token z przewidywanego rozkładu
- Modele dyfuzyjne próbkują szum i stopniowo go usuwają

Próbkowanie z dowolnych rozkładów wymaga technik takich jak próbkowanie przez odwróconą dystrybuantę, próbkowanie przez odrzucanie lub sztuczka reparametryzacji (używana w VAE).

```figure
gaussian-pdf
```

## Build It

### Krok 1: Podstawy prawdopodobieństwa

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(Król | Figura) = {p_king_given_face:.4f}")
```

### Krok 2: PMF i PDF od podstaw

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Krok 3: Wartość oczekiwana i wariancja

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Kostka: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Krok 4: Próbkowanie z rozkładów

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Krok 5: Softmax i log-prawdopodobieństwa

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Krok 6: Demonstracja Centralnego Twierdzenia Granicznego

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Krok 7: Wizualizacja

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Pełne implementacje ze wszystkimi wizualizacjami są w `code/probability.py`.

## Use It

Z NumPy i SciPy wszystko powyższe to jednolinijkowce:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Średnia: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Zbudowałeś to od podstaw. Teraz wiesz, co robią wywołania bibliotek.

## Ćwiczenia

1. Zaimplementuj próbkowanie przez odwróconą dystrybuantę dla rozkładu wykładniczego. Zweryfikuj przez próbkowanie 10 000 wartości i porównanie histogramu z prawdziwą PDF.

2. Zbuduj tabelę rozkładu łącznego dla dwóch obciążonych kostek. Oblicz rozkłady brzegowe i sprawdź, czy kostki są niezależne.

3. Oblicz stratę cross-entropii dla klasyfikatora 5-klasowego, który wyjściowo daje logity `[2.0, 0.5, -1.0, 3.0, 0.1]`, gdy poprawna klasa to indeks 3. Następnie zweryfikuj swoją odpowiedź z `nn.CrossEntropyLoss` PyTorcha.

4. Napisz funkcję, która przyjmuje listę log-prawdopodobieństw i zwraca najbardziej prawdopodobną sekwencję, całkowite log-prawdopodobieństwo i równoważne surowe prawdopodobieństwo. Przetestuj ze zdaniem 50 słów, gdzie każde słowo ma prawdopodobieństwo 0.01.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| Przestrzeń próbek | "Wszystkie możliwości" | Zbiór S każdego możliwego wyniku eksperymentu |
| PMF | "Funkcja prawdopodobieństwa" | Funkcja dająca dokładne prawdopodobieństwo każdego dyskretnego wyniku, sumująca się do 1 |
| PDF | "Krzywa prawdopodobieństwa" | Funkcja gęstości dla zmiennych ciągłych. Całkuj po przedziale, by dostać prawdopodobieństwo |
| Prawdopodobieństwo warunkowe | "Prawdopodobieństwo pod warunkiem" | P(A\|B) = P(A i B) / P(B). Podstawa myślenia bayesowskiego i twierdzenia Bayesa |
| Niezależność | "Nie wpływają na siebie" | P(A i B) = P(A) * P(B). Wiedza o jednym zdarzeniu nic nie mówi o drugim |
| Wartość oczekiwana | "Średnia" | Suma wszystkich wyników ważona prawdopodobieństwem. Funkcja straty to wartość oczekiwana |
| Wariancja | "Jak rozrzucone" | Oczekiwane kwadratowe odchylenie od średniej. Wysoka wariancja = głośne, niestabilne oszacowania |
| Rozkład normalny | "Krzywa dzwonowa" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Pojawia się wszędzie dzięki CTG |
| Centralne Twierdzenie Graniczne | "Średnie stają się normalne" | Średnia wielu niezależnych próbek zbiega do rozkładu normalnego niezależnie od źródła |
| Rozkład łączny | "Dwie zmienne razem" | P(X, Y) opisuje prawdopodobieństwo każdej kombinacji wyników X i Y |
| Rozkład brzegowy | "Wysumuj drugą zmienną" | P(X) = suma_y P(X, Y). Odtwarza rozkład jednej zmiennej z łącznego |
| Log-prawdopodobieństwo | "Logarytm prawdopodobieństwa" | log P(x). Zamienia iloczyny w sumy, zapobiegając numerycznemu niedomiarewi w długich sekwencjach |
| Softmax | "Zmień wyniki na prawdopodobieństwa" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Mapuje rzeczywiste logity na poprawny rozkład prawdopodobieństwa |
| Cross-entropia | "Funkcja straty" | -sum(p_true * log(p_predicted)). Mierzy, jak bardzo dwa rozkłady się różnią. Niżej = lepiej |
| Logity | "Surowe wyjścia modelu" | Nienormalizowane wyniki przed softmaxem. Nazwane od funkcji logistycznej |
| Próbkowanie | "Losowanie wartości" | Generowanie wartości zgodnie z rozkładem prawdopodobieństwa. Jak modele generują wyjście |