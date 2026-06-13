# Stabilność numeryczna

> Zmiennoprzecinkowość to cieknąca abstrakcja. Ugryzie cię podczas trenowania, a nie zobaczysz, jak nadchodzi.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minut

## Learning Objectives

- Zaimplementuj numerycznie stabilny softmax i log-sum-exp używając sztuczki odejmowania maksimum
- Zidentyfikuj nadmiar, niedomiar i katastrofalne skracanie w obliczeniach zmiennoprzecinkowych
- Zweryfikuj analityczne gradienty względem numerycznych gradientów używając wyśrodkowanych różnic skończonych
- Wyjaśnij, dlaczego bfloat16 jest preferowane nad float16 do trenowania i jak skalowanie straty zapobiega niedomiarewi gradientów

## Problem

Twój model trenuje przez trzy godziny, po czym strata staje się NaN. Dodajesz print. Logity są w porządku w kroku 9 000. W kroku 9 001 są `inf`. Do kroku 9 002 każdy gradient jest `nan` i trenowanie jest martwe.

Albo: twój model trenuje do końca, ale dokładność jest o 2% gorsza niż twierdzi praca. Sprawdzasz wszystko. Architektura się zgadza. Hiperparametry się zgadzają. Dane się zgadzają. Problem w tym, że praca używała float32, a ty użyłeś float16 bez odpowiedniego skalowania. Trzydzieści dwa bity skumulowanego błędu zaokrągleń po cichu zjadło twoją dokładność.

Albo: implementujesz stratę cross-entropii od podstaw. Działa na małych logitach. Gdy logity przekraczają 100, zwraca `inf`. Softmax przekroczył zakres, bo `exp(100)` jest większe niż float32 może reprezentować. Każdy framework ML obsługuje to dwulinijkową sztuczką. Nie wiedziałeś, że ta sztuczka istnieje.

Stabilność numeryczna nie jest teoretycznym zmartwieniem. To różnica między udanym trenowaniem a cichym niepowodzeniem. Każdy poważny błąd ML, który będziesz debugować, w końcu sprowadza się do zmiennoprzecinkowości.

## Koncepcja

### IEEE 754: Jak komputery przechowują liczby rzeczywiste

Komputery przechowują liczby rzeczywiste jako wartości zmiennoprzecinkowe zgodnie ze standardem IEEE 754. Float ma trzy części: bit znaku, wykładnik i mantysę (znaczącą).

```
Układ Float32 (32 bity razem):
[1 znak] [8 wykładnik] [23 mantysa]

Wartość = (-1)^znak * 2^(wykładnik - 127) * 1.mantysa
```

Mantysa określa precyzję (ile cyfr znaczących). Wykładnik określa zakres (jak duża lub mała może być liczba).

```
Format     Bity   Wykładnik  Mantysa  Cyfr dziesiętnych  Zakres (około)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65 504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 daje około 7 cyfr dziesiętnych precyzji. To znaczy, że może odróżnić 1.0000001 od 1.0000002, ale nie 1.00000001 od 1.00000002. Po 7 cyfrach wszystko jest szumem zaokrągleń.

float16 daje około 3 cyfr. Największa liczba, jaką może reprezentować, to 65 504. To niepokojąco mało dla ML, gdzie logity, gradienty i aktywacje regularnie to przekraczają.

bfloat16 to odpowiedź Google na problem zakresu float16. Ma ten sam 8-bitowy wykładnik co float32 (ten sam zakres, do 3.4e38) ale tylko 7 bitów mantysy (mniej precyzji niż float16). Do trenowania sieci neuronowych zakres ma większe znaczenie niż precyzja, więc bfloat16 zwykle wygrywa.

### Dlaczego 0.1 + 0.2 != 0.3

Liczba 0.1 nie może być reprezentowana dokładnie w binarnym zmiennoprzecinkowym. W systemie binarnym jest to ułamek okresowy:

```
0.1 w binarnym = 0.0001100110011001100110011... (powtarza się w nieskończoność)
```

Float32 uci:na to do 23 bitów mantysy. Przechowywana wartość to około 0.100000001490116. Podobnie 0.2 jest przechowywana jako około 0.200000002980232. Ich suma to 0.300000004470348, nie 0.3.

```
W Pythonie:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

To ma znaczenie dla ML, ponieważ:

1. Porównania strat jak `if loss < threshold` mogą dawać błędne odpowiedzi
2. Akumulacja wielu małych wartości (aktualizacje gradientów przez tysiące kroków) dryfuje od prawdziwej sumy
3. Sumy kontrolne i testy odtwarzalności zawodzą, jeśli porównujesz floaty z `==`

Rozwiązanie: nigdy nie porównuj floatów z `==`. Użyj `abs(a - b) < epsilon` lub `math.isclose()`.

### Katastrofalne skracanie

Gdy odejmujesz dwie prawie równe liczby zmiennoprzecinkowe, cyfry znaczące się skracają i pozostaje szum zaokrągleń wyniesiony do wiodących cyfr.

```
a = 1.0000001    (przechowywane jako 1.00000011920929 w float32)
b = 1.0000000    (przechowywane jako 1.00000000000000 w float32)

Prawdziwa różnica:  0.0000001
Obliczona:         0.00000011920929

Błąd względny: 19.2%
```

To 19% błędu względnego z pojedynczego odejmowania. W ML dzieje się tak, gdy:

- Obliczasz wariancję danych z dużą średnią: `E[x^2] - E[x]^2` gdy E[x] jest duże
- Odejmujesz prawie równe log-prawdopodobieństwa
- Obliczasz gradienty różnic skończonych ze zbyt małym epsilonem

Rozwiązanie: przekształć wzory, by uniknąć odejmowania dużych, prawie równych liczb. Dla wariancji użyj algorytmu Welforda lub wycentruj dane najpierw. Dla log-prawdopodobieństw pracuj w przestrzeni logarytmicznej przez cały czas.

### Nadmiar i niedomiar

Nadmiar występuje, gdy wynik jest zbyt duży do reprezentacji. Niedomiar, gdy jest zbyt mały (bliższy zeru niż najmniejsza reprezentowalna liczba dodatnia).

```
Granice float32:
  Maksimum:  3.4028235e+38
  Minimum dodatnie (normalne): 1.175e-38
  Minimum dodatnie (denormalne): 1.401e-45
  Nadmiar:  cokolwiek > 3.4e38 staje się inf
  Niedomiar: cokolwiek < 1.4e-45 staje się 0.0
```

Funkcja `exp()` jest głównym źródłem nadmiaru w ML:

```
exp(88.7)  = 3.40e+38   (ledwo mieści się w float32)
exp(89.0)  = inf         (nadmiar)
exp(-87.3) = 1.18e-38   (ledwo powyżej niedomiaru)
exp(-104)  = 0.0         (niedomiar do zera)
```

Funkcja `log()` uderza w drugim kierunku:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (w porządku)
log(1e-46) = -inf        (wejście niedomiar do 0, potem log(0) = -inf)
```

W ML `exp()` pojawia się w softmaxie, sigmoidzie i obliczeniach prawdopodobieństwa. `log()` pojawia się w cross-entropii, log-wiarygodnościach i dywergencji KL. Kombinacja `log(exp(x))` to pole minowe bez odpowiednich sztuczek.

### Sztuczka Log-Sum-Exp

Bezpośrednie obliczenie `log(sum(exp(x_i)))` jest numerycznie niebezpieczne. Jeśli któreś `x_i` jest duże, `exp(x_i)` daje nadmiar. Jeśli wszystkie `x_i` są bardzo ujemne, każde `exp(x_i)` daje niedomiar do zera i `log(0)` to `-inf`.

Sztuczka: odejmij maksymalną wartość przed potęgowaniem.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Dlaczego to działa: po odjęciu `max(x)`, największy wykładnik to `exp(0) = 1`. Żaden nadmiar nie jest możliwy. Przynajmniej jeden człon w sumie to 1, więc suma wynosi przynajmniej 1, a `log(1) = 0`. Żaden niedomiar do `-inf` nie jest możliwy.

Dowód:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (dodaj i odejmij c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (wyciągnij exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Ustaw `c = max(x)` i nadmiar jest wyeliminowany.

Ta sztuczka pojawia się wszędzie w ML:
- Normalizacja softmax
- Obliczanie straty cross-entropii
- Sumowanie log-prawdopodobieństw w modelach sekwencyjnych
- Mieszanka Gaussów
- Wnioskowanie wariacyjne

### Dlaczego Softmax potrzebuje sztuczki odejmowania maksimum

Softmax przekształca logity na prawdopodobieństwa:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Bez sztuczki, logity [100, 101, 102] powodują nadmiar:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

Te przepełniają float32 (max ~3.4e38)? Nie, 2.69e43 < 3.4e38? Właściwie:
exp(88.7) jest już przy limicie float32.
exp(100) = inf w float32.
```

Ze sztuczką, odejmij max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Prawdopodobieństwa są identyczne. Obliczenia są bezpieczne. To nie optymalizacja. To wymóg poprawności.

### NaN i Inf: wykrywanie i zapobieganie

`nan` (Not a Number) i `inf` (nieskończoność) rozprzestrzeniają się wirusowo przez obliczenia. Jeden `nan` w aktualizacji gradientu czyni wagę `nan`, co czyni każde kolejne wyjście `nan`. Trenowanie jest martwe w ciągu jednego kroku.

Jak pojawia się `inf`:
- `exp()` dużej dodatniej liczby
- Dzielenie przez zero: `1.0 / 0.0`
- Nadmiar `float32` w akumulacjach

Jak pojawia się `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` ujemnej liczby
- `log()` ujemnej liczby
- Dowolne działanie arytmetyczne z istniejącym `nan`

Wykrywanie:

```python
import math

math.isnan(x)       # True jeśli x to nan
math.isinf(x)       # True jeśli x to +inf lub -inf
math.isfinite(x)    # True jeśli x nie jest ani nan ani inf
```

Strategie zapobiegania:

1. Ogranicz wejścia do `exp()`: `exp(clamp(x, -80, 80))`
2. Dodaj epsilon do mianowników: `x / (y + 1e-8)`
3. Dodaj epsilon wewnątrz `log()`: `log(x + 1e-8)`
4. Użyj stabilnych implementacji (log-sum-exp, stabilny softmax)
5. Obcinanie gradientów, by zapobiec eksplozji wag
6. Sprawdź `nan`/`inf` po każdym przejściu w przód podczas debugowania

### Sprawdzanie gradientów numerycznych

Analityczne gradienty (z wstecznej propagacji) mogą mieć błędy. Numeryczne sprawdzanie gradientów weryfikuje je przez obliczanie gradientów różnicami skończonymi.

Wzór na różnicę wyśrodkowaną:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

To jest dokładne O(h^2), znacznie lepsze niż różnica w przód `(f(x+h) - f(x)) / h` która jest tylko O(h).

Wybór h: za duże i przybliżenie jest błędne. Za małe i katastrofalne skracanie niszczy odpowiedź. `h = 1e-5` do `1e-7` to typowy zakres.

Sprawdzenie: oblicz względną różnicę między analitycznymi a numerycznymi gradientami.

```
względny_błąd = |grad_analityczny - grad_numeryczny| / max(|grad_analityczny|, |grad_numeryczny|, 1e-8)
```

Reguły kciuka:
- względny_błąd < 1e-7: doskonale, gradient jest poprawny
- względny_błąd < 1e-5: akceptowalne, prawdopodobnie poprawne
- względny_błąd > 1e-3: coś jest nie tak
- względny_błąd > 1: gradient jest całkowicie błędny

Zawsze sprawdzaj gradienty przy implementacji nowej warstwy lub funkcji straty. PyTorch dostarcza `torch.autograd.gradcheck()` do tego.

### Trenowanie w mieszanej precyzji

Nowoczesne GPU mają wyspecjalizowany sprzęt (Tensor Cores), który oblicza mnożenie macierzy w float16 2-8x szybciej niż w float32. Trenowanie w mieszanej precyzji wykorzystuje to:

```
1. Utrzymaj float32 główną kopię wag
2. Przejście w przód w float16 (szybko)
3. Oblicz stratę w float32 (zapobiega nadmiarowi)
4. Przejście wsteczne w float16 (szybko)
5. Skaluj gradienty do float32
6. Zaktualizuj float32 główne kopie wag
```

Problem z czystym float16: gradienty są często bardzo małe (1e-8 lub mniejsze). Float16 powoduje niedomiar wszystkiego poniżej ~6e-8 do zera. Twój model przestaje się uczyć, bo wszystkie aktualizacje gradientów są zerowe.

Rozwiązaniem jest skalowanie straty:

```
1. Pomnóż stratę przez duży współczynnik skali (np. 1024)
2. Przejście wsteczne oblicza gradienty (strata * 1024)
3. Wszystkie gradienty są 1024x większe (wypchnięte powyżej niedomiaru float16)
4. Podziel gradienty przez 1024 przed aktualizacją wag
5. Efekt netto: ta sama aktualizacja, ale bez niedomiaru
```

Dynamiczne skalowanie straty dostosowuje współczynnik skali automatycznie. Zacznij z dużą wartością (65536). Jeśli gradienty dają nadmiar do `inf`, zmniejsz o połowę. Jeśli N kroków przejdzie bez nadmiaru, podwój.

### bfloat16 vs float16: Dlaczego bfloat16 wygrywa przy trenowaniu

```
float16:   [1 znak] [5 wykładnik]  [10 mantysa]
bfloat16:  [1 znak] [8 wykładnik]  [7 mantysa]
```

float16 ma większą precyzję (10 bitów mantysy vs 7), ale ograniczony zakres (max ~65 504). bfloat16 ma mniejszą precyzję, ale ten sam zakres co float32 (max ~3.4e38).

Do trenowania sieci neuronowych:

- Aktywacje i logity regularnie przekraczają 65 504 podczas skoków treningowych. float16 daje nadmiar; bfloat16 sobie radzi.
- Skalowanie straty jest wymagane z float16, ale zwykle niepotrzebne z bfloat16, ponieważ jego zakres pokrywa spektrum wielkości gradientów.
- bfloat16 to proste obcięcie float32: odrzuć dolne 16 bitów mantysy. Konwersja jest trywialna i bezstratna w wykładniku.

float16 jest preferowane do wnioskowania, gdzie wartości są ograniczone i precyzja ma większe znaczenie. bfloat16 jest preferowane do trenowania, gdzie zakres ma większe znaczenie. Dlatego TPU i nowoczesne GPU NVIDIA (A100, H100) mają natywną obsługę bfloat16.

### Obcinanie gradientów

Eksplodujące gradienty występują, gdy gradienty rosną wykładniczo przez wiele warstw (częste w RNN, głębokich sieciach i transformerach). Pojedynczy duży gradient może zepsuć wszystkie wagi w jednym kroku.

Dwa typy obcinania:

**Obcięcie według wartości:** ogranicz każdy element gradientu niezależnie.

```
grad = clamp(grad, -max_val, max_val)
```

Proste, ale może zmienić kierunek wektora gradientu.

**Obcięcie według normy:** skaluj cały wektor gradientu, tak by jego norma nie przekraczała progu.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Zachowuje kierunek gradientu. To robi `torch.nn.utils.clip_grad_norm_()`. To standardowy wybór.

Typowe wartości: `max_norm=1.0` dla transformerów, `max_norm=0.5` dla RL, `max_norm=5.0` dla prostszych sieci.

Obcinanie gradientów nie jest hackiem. To mechanizm bezpieczeństwa. Bez niego pojedynczy wsad odstający może wyprodukować gradient wystarczająco duży, by zniszczyć tygodnie trenowania.

### Warstwy normalizacji jako stabilizatory numeryczne

Normalizacja wsadowa, normalizacja warstwowa i normalizacja RMS są zwykle przedstawiane jako regulatory pomagające trenowaniu zbiegać. Są również stabilizatorami numerycznymi.

Bez normalizacji aktywacje mogą rosnąć lub maleć wykładniczo przez warstwy:

```
Warstwa 1: wartości w [0, 1]
Warstwa 5: wartości w [0, 100]
Warstwa 10: wartości w [0, 10 000]
Warstwa 50: wartości w [0, inf]
```

Normalizacja ponownie centruje i skaluje aktywacje na każdej warstwie:

```
LayerNorm(x) = (x - średnia(x)) / (odchylenie standardowe(x) + epsilon) * gamma + beta
```

`epsilon` (zwykle 1e-5) zapobiega dzieleniu przez zero, gdy wszystkie aktywacje są identyczne. Uczone parametry `gamma` i `beta` pozwalają sieci przywrócić dowolną skalę, której potrzebuje.

To utrzymuje wartości w numerycznie bezpiecznym zakresie przez całą sieć, zapobiegając zarówno nadmiarowi w przejściu w przód, jak i eksplozji gradientów w przejściu wstecznym.

### Typowe numeryczne błędy ML

**Błąd: Strata to NaN po kilku epokach.**
Przyczyna: Logity urosły zbyt duże, softmax dał nadmiar. Albo współczynnik uczenia jest zbyt wysoki i wagi się rozeszły.
Naprawa: użyj stabilnego softmax (odejmowanie maksimum), zmniejsz współczynnik uczenia, dodaj obcinanie gradientów.

**Błąd: Strata utknęła na log(liczba_klas).**
Przyczyna: Model wyjściowo daje prawie jednostajne prawdopodobieństwa. Często oznacza to zanikające gradienty lub model w ogóle się nie uczy.
Naprawa: sprawdź, czy etykiety danych są poprawne, zweryfikuj funkcję straty, sprawdź martwe ReLU.

**Błąd: Dokładność walidacji jest niższa niż oczekiwana o 1-3%.**
Przyczyna: Mieszana precyzja bez odpowiedniego skalowania straty. Niedomiar gradientów po cichu zeruje małe aktualizacje.
Naprawa: włącz dynamiczne skalowanie straty lub przełącz na bfloat16.

**Błąd: Normy gradientów są 0.0 dla niektórych warstw.**
Przyczyna: Martwe neurony ReLU (wszystkie wejścia ujemne) lub niedomiar float16.
Naprawa: użyj LeakyReLU lub GELU, użyj skalowania gradientów, sprawdź inicjalizację wag.

**Błąd: Model działa na jednym GPU, ale daje inne wyniki na innym.**
Przyczyna: Niedeterministyczna kolejność akumulacji zmiennoprzecinkowej. Równoległe redukcje GPU sumują w różnych kolejnościach na różnym sprzęcie, a dodawanie zmiennoprzecinkowe nie jest łączne.
Naprawa: zaakceptuj małe różnice (1e-6) lub ustaw `torch.use_deterministic_algorithms(True)` i zaakceptuj karę szybkości.

**Błąd: `exp()` zwraca `inf` w obliczaniu straty.**
Przyczyna: Surowe logity przekazane do `exp()` bez sztuczki odejmowania maksimum.
Naprawa: użyj `torch.nn.functional.log_softmax()` które implementuje log-sum-exp wewnętrznie.

**Błąd: Trenowanie rozbiega się po przejściu z float32 na float16.**
Przyczyna: float16 nie może reprezentować wielkości gradientów poniżej 6e-8 ani aktywacji powyżej 65 504.
Naprawa: użyj mieszanej precyzji ze skalowaniem straty (AMP) lub użyj bfloat16.

```figure
logsumexp-stability
```

## Build It

### Krok 1: Zademonstruj ograniczenia precyzji zmiennoprzecinkowej

```python
print("=== Precyzja zmiennoprzecinkowa ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Różnica: {(0.1 + 0.2) - 0.3:.2e}")
```

### Krok 2: Zaimplementuj naiwny vs stabilny softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naiwny:  {softmax_naive(safe_logits)}")
print(f"Stabilny: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stabilny: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) zwróciłby [nan, nan, nan]
```

### Krok 3: Zaimplementuj stabilny log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naiwny:  {logsumexp_naive(safe):.6f}")
print(f"Stabilny: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stabilny: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) zwraca inf
```

### Krok 4: Zaimplementuj stabilną cross-entropię

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naiwny:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stabilny: {cross_entropy_stable(true_class, logits):.6f}")
```

### Krok 5: Sprawdzanie gradientów

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analityczna={a:.8f} numeryczna={n:.8f} "
              f"wzgl_błąd={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Symulacja mieszanej precyzji

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Obcinanie gradientów

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Oryginalna norma: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Obcięta norma:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Kierunek zachowany: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Wykrywanie NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"OSTRZEŻENIE {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("dobry", [1.0, 2.0, 3.0])
check_tensor("zły",  [1.0, float('nan'), 3.0])
check_tensor("brzydki", [1.0, float('inf'), 3.0])
```

Zobacz `code/numerical.py` dla kompletnych implementacji ze wszystkimi przypadkami skrajnymi.

## Ship It

Ta lekcja produkuje:
- `code/numerical.py` ze stabilnym softmaxem, log-sum-exp, cross-entropią, sprawdzaniem gradientów i symulacją mieszanej precyzji
- `outputs/prompt-numerical-debugger.md` do diagnozowania NaN/Inf i problemów numerycznych w trenowaniu

Te stabilne implementacje pojawiają się ponownie w Phase 3 przy budowaniu pętli treningowej i w Phase 4 przy implementacji mechanizmów uwagi.

## Ćwiczenia

1. **Katastrofalne skracanie.** Oblicz wariancję [1000000.0, 1000001.0, 1000002.0] używając naiwnego wzoru `E[x^2] - E[x]^2` w float32. Następnie oblicz ją używając algorytmu online Welforda. Porównaj błędy względem prawdziwej wariancji (0.6667).

2. **Polowanie na precyzję.** Znajdź najmniejszą dodatnią wartość float32 `x` taką, że `1.0 + x == 1.0` w Pythonie. To jest epsilon maszynowy. Zweryfikuj, że zgadza się z `numpy.finfo(numpy.float32).eps`.

3. **Przypadki skrajne log-sum-exp.** Przetestuj swoją funkcję `logsumexp_stable` z: (a) wszystkimi wartościami równymi, (b) jedną wartością znacznie większą od reszty, (c) wszystkimi wartościami bardzo ujemnymi (-1000). Zweryfikuj, że daje poprawne wyniki tam, gdzie naiwna wersja zawodzi.

4. **Sprawdzanie gradientów warstwy sieci neuronowej.** Zaimplementuj pojedynczą warstwę liniową `y = Wx + b` i jej analityczne przejście wsteczne. Użyj `numerical_gradient` do weryfikacji poprawności dla macierzy wag 3x2.

5. **Eksperyment skalowania straty.** Zasymuluj trenowanie z float16: stwórz losowe gradienty w zakresie [1e-9, 1e-3], przekonwertuj na float16 i zmierz, jaki ułamek staje się zerem. Następnie zastosuj skalowanie straty (pomnóż przez 1024), przekonwertuj na float16, odskaluj i zmierz ułamek zer ponownie.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| IEEE 754 | "Standard float" | Międzynarodowy standard definiujący formaty binarne zmiennoprzecinkowe, reguły zaokrąglania i specjalne wartości (inf, nan). Każde nowoczesne CPU i GPU go implementuje. |
| Epsilon maszynowy | "Limit precyzji" | Najmniejsza wartość e taka, że 1.0 + e != 1.0 w danym formacie float. Dla float32 to około 1.19e-7. |
| Katastrofalne skracanie | "Strata precyzji z odejmowania" | Gdy odejmujesz prawie równe liczby zmiennoprzecinkowe, cyfry znaczące się skracają, a szum zaokrągleń dominuje wynik. |
| Nadmiar | "Liczba za duża" | Wynik przekracza maksymalną reprezentowalną wartość i staje się inf. exp(89) daje nadmiar w float32. |
| Niedomiar | "Liczba za mała" | Wynik jest bliższy zeru niż najmniejsza reprezentowalna liczba dodatnia i staje się 0.0. exp(-104) daje niedomiar w float32. |
| Sztuczka log-sum-exp | "Odejmij najpierw max" | Obliczanie log(sum(exp(x))) przez wyciągnięcie exp(max(x)), by zapobiec nadmiarowi i niedomiarewi. Używane w softmaxie, cross-entropii i log-prawdopodobieństwach. |
| Stabilny softmax | "Softmax, który nie wybucha" | Odejmowanie max(logity) przed potęgowaniem. Numerycznie identyczny wynik, brak możliwego nadmiaru. |
| Sprawdzanie gradientów | "Zweryfikuj swoją wsteczną propagację" | Porównanie analitycznych gradientów z wstecznej propagacji z numerycznymi gradientami z różnic skończonych, by wyłapać błędy implementacji. |
| Mieszana precyzja | "Float16 w przód, float32 wstecz" | Używanie niższej precyzji floatów dla operacji krytycznych szybkościowo i wyższej precyzji dla operacji wrażliwych numerycznie. Typowe przyspieszenie to 2-3x. |
| Skalowanie straty | "Zapobiegaj niedomiarewi gradientów" | Mnożenie straty przez dużą stałą przed wsteczną propagacją, by gradienty pozostały w reprezentowalnym zakresie float16, a następnie dzielenie przez tę samą stałą przed aktualizacją wag. |
| bfloat16 | "Brain floating point" | 16-bitowy format Google z 8 bitami wykładnika (ten sam zakres co float32) i 7 bitami mantysy (mniej precyzji niż float16). Preferowany do trenowania. |
| Obcinanie gradientów | "Ogranicz normę gradientu" | Skalowanie wektora gradientu tak, by jego norma nie przekraczała progu. Zapobiega psuciu wag przez eksplodujące gradienty. |
| NaN | "Not a Number" | Specjalna wartość float z niezdefiniowanych operacji (0/0, inf-inf, sqrt(-1)). Propaguje się przez całą późniejszą arytmetykę. |
| Inf | "Nieskończoność" | Specjalna wartość float z nadmiaru lub dzielenia przez zero. Może łączyć się, by produkować NaN (inf - inf, inf * 0). |
| Gradient numeryczny | "Pochodna na brute force" | Przybliżenie pochodnej przez obliczenie f(x+h) i f(x-h) i podzielenie przez 2h. Wolne, ale niezawodne do weryfikacji. |