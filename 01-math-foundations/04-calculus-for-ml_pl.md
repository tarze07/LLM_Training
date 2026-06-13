# Rachunek różniczkowy dla uczenia maszynowego

> Pochodne mówią ci, w którą stronę jest w dół. Tylko tyle potrzebuje sieć neuronowa, by się uczyć.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minut

## Learning Objectives

- Obliczaj numeryczne i analityczne pochodne dla typowych funkcji ML (x^2, sigmoid, cross-entropia)
- Zaimplementuj spadek gradientowy od podstaw, aby minimalizować funkcję straty w 1D i 2D
- Wyprowadź gradient modelu regresji liniowej i trenuj go przez ręczną aktualizację wag
- Wyjaśnij macierz Hessianu, przybliżenia szeregiem Taylora i ich związek z metodami optymalizacji

## Problem

Masz sieć neuronową z milionami wag. Każda waga to pokrętło. Musisz ustalić, w którą stronę przekręcić każde pokrętło, by model był nieco mniej błędny. Rachunek różniczkowy daje ci ten kierunek.

Bez rachunku różniczkowego trenowanie sieci neuronowej oznaczałoby próbowanie losowych zmian i liczenie na najlepsze. Z pochodnymi wiesz dokładnie, jak każda waga wpływa na błąd. Przekręcasz każde pokrętło we właściwą stronę, za każdym razem.

## Koncepcja

### Czym jest pochodna?

Pochodna mierzy tempo zmian. Dla funkcji y = f(x), pochodna f'(x) mówi: jeśli poruszysz x o malutką wartość, o ile zmieni się y?

Geometrycznie pochodna to nachylenie stycznej w punkcie.

**f(x) = x^2:**

| x | f(x) | f'(x) (nachylenie) |
|---|------|---------------|
| 0 | 0    | 0 (płasko, na dnie) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (nachylenie stycznej w tym punkcie) |
| 3 | 9    | 6 |

W x=2 nachylenie wynosi 4. Jeśli przesuniesz x odrobinę w prawo, y zwiększy się około 4 razy tyle. W x=0 nachylenie wynosi 0. Jesteś na dnie doliny.

Formalna definicja:

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

W kodzie pomijasz granicę i używasz bardzo małego h. To jest pochodna numeryczna.

### Pochodne cząstkowe: jedna zmienna na raz

Prawdziwe funkcje mają wiele wejść. Strata sieci neuronowej zależy od tysięcy wag. Pochodna cząstkowa utrzymuje wszystkie zmienne stałe z wyjątkiem jednej, a następnie bierze pochodną względem tej jednej.

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (traktuj y jako stałą)
df/dy = 3x + 2y     (traktuj x jako stałą)
```

Każda pochodna cząstkowa odpowiada na pytanie: jeśli poruszę tylko tę jedną wagę, jak zmieni się strata?

### Gradient: wektor wszystkich pochodnych cząstkowych

Gradient zbiera każdą pochodną cząstkową w jeden wektor. Dla funkcji f(x, y, z) gradient to:

```
grad f = [ df/dx, df/dy, df/dz ]
```

Gradient wskazuje kierunek największego wzrostu. Aby zminimalizować funkcję, idź w przeciwnym kierunku.

**Mapa warstwicowa f(x,y) = x^2 + y^2:**

Funkcja tworzy kształt misy z koncentrycznymi okręgami jako liniami warstwicowymi. Minimum znajduje się w (0, 0).

| Punkt | grad f | -grad f (kierunek spadku) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (wskazuje pod górę, od minimum) | [-2, -2] (wskazuje w dół, do minimum) |
| (0, 0) | [0, 0] (płasko, w minimum) | [0, 0] |

To jest spadek gradientowy na obrazku. Oblicz gradient, zaneguj go, zrób krok.

### Związek z optymalizacją

Trenowanie sieci neuronowej to optymalizacja. Masz funkcję straty L(w1, w2, ..., wn), która mierzy, jak bardzo model się myli. Chcesz ją zminimalizować.

```
Reguła aktualizacji spadku gradientowego:

  w_new = w_old - współczynnik_uczenia * dL/dw

Dla każdej wagi:
  1. Oblicz pochodną cząstkową straty względem tej wagi
  2. Odejmij małą jej wielokrotność od wagi
  3. Powtórz
```

Współczynnik uczenia kontroluje wielkość kroku. Za duży i przeskoczysz. Za mały i będziesz się czołgać.

**Krajobraz straty (przekrój 1D):**

Funkcja straty L(w) tworzy krzywą ze szczytami i dolinami, gdy waga w się zmienia.

| Cecha | Opis |
|---------|-------------|
| Minimum globalne | Najniższy punkt na całej krzywej – najlepsze rozwiązanie |
| Minimum lokalne | Dolina niższa od sąsiadów, ale nie najniższa ogólnie |
| Nachylenie | Spadek gradientowy podąża za nachyleniem w dół z dowolnego punktu początkowego |

Spadek gradientowy podąża za nachyleniem w dół. Może utknąć w minimach lokalnych, ale w przestrzeniach wysokowymiarowych (miliony wag) jest to rzadko praktycznym problemem.

### Pochodne numeryczne vs analityczne

Są dwa sposoby obliczania pochodnej.

Analityczny: zastosuj reguły rachunku różniczkowego ręcznie. Dla f(x) = x^2, pochodna to f'(x) = 2x. Dokładna. Szybka.

Numeryczny: przybliż używając definicji. Oblicz f(x+h) i f(x-h) dla małego h, a następnie użyj różnicy.

```
Numeryczny (różnica centralna):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 działa dobrze w praktyce
```

Pochodne numeryczne są wolniejsze, ale działają dla dowolnej funkcji. Pochodne analityczne są szybkie, ale wymagają wyprowadzenia wzoru. Frameworki sieci neuronowych używają trzeciego podejścia: automatycznego różniczkowania, które oblicza dokładne pochodne mechanicznie. Zobaczysz to w Phase 3.

### Ręczne pochodne dla prostych funkcji

To są pochodne, które będziesz widzieć wielokrotnie w ML.

```
Funkcja        Pochodna       Zastosowanie
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Funkcje straty (MSE)
f(x) = wx + b  f'(w) = x        Warstwa liniowa (gradient względem wagi)
                f'(b) = 1        Warstwa liniowa (gradient względem biasu)
                f'(x) = w        Warstwa liniowa (gradient względem wejścia)
f(x) = e^x     f'(x) = e^x     Softmax, uwaga
f(x) = ln(x)   f'(x) = 1/x     Strata cross-entropii
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Aktywacja sigmoidalna
```

Dla f(x) = x^2:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   znaczenie
  -2    4      -4      nachylenie w lewo (malejące)
  -1    1      -2      nachylenie w lewo (malejące)
   0    0       0      płasko (minimum!)
   1    1       2      nachylenie w prawo (rosnące)
   2    4       4      nachylenie w prawo (rosnące)
```

Dla f(w) = wx + b z x=3, b=1:

```
f(w) = 3w + 1    f'(w) = 3

Pochodna względem w to po prostu x.
Jeśli x jest duże, mała zmiana w powoduje dużą zmianę na wyjściu.
```

### Reguła łańcuchowa

Gdy funkcje są złożone, reguła łańcuchowa mówi, jak różniczkować.

```
Jeśli y = f(g(x)), to dy/dx = f'(g(x)) * g'(x)

Przykład: y = (3x + 1)^2
  zewnętrzna: f(u) = u^2       f'(u) = 2u
  wewnętrzna: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

Sieci neuronowe to łańcuchy funkcji: wejście -> liniowa -> aktywacja -> liniowa -> aktywacja -> strata. Wsteczna propagacja to reguła łańcuchowa stosowana wielokrotnie od wyjścia do wejścia. To cały algorytm.

### Macierz Hessianu

Gradient mówi o nachyleniu. Hessian mówi o krzywiźnie.

Hessian to macierz pochodnych cząstkowych drugiego rzędu. Dla funkcji f(x1, x2, ..., xn), element (i, j) Hessianu to:

```
H[i][j] = d^2f / (dx_i * dx_j)
```

Dla funkcji 2-zmiennych f(x, y):

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**Co Hessian mówi w punkcie krytycznym (gdzie gradient = 0):**

| Własność Hessianu | Znaczenie | Przykład powierzchni |
|-----------------|---------|-----------------|
| Dodatnio określony (wszystkie wartości własne > 0) | Minimum lokalne | Misa skierowana w górę |
| Ujemnie określony (wszystkie wartości własne < 0) | Maksimum lokalne | Misa skierowana w dół |
| Nieokreślony (mieszane wartości własne) | Punkt siodłowy | Kształt siodła |

**Przykład:** f(x, y) = x^2 - y^2 (funkcja siodłowa)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Wartości własne: 2 i -2 (jedna dodatnia, jedna ujemna)
--> Punkt siodłowy w (0, 0)
```

Porównaj z f(x, y) = x^2 + y^2 (misa):

```
H = | 2  0 |
    | 0  2 |

Wartości własne: 2 i 2 (obie dodatnie)
--> Minimum lokalne w (0, 0)
```

**Dlaczego Hessian ma znaczenie w ML:**

Metoda Newtona używa Hessianu do lepszych kroków optymalizacyjnych niż spadek gradientowy. Zamiast tylko podążać za nachyleniem, uwzględnia krzywiznę:

```
Aktualizacja Newtona:    w_new = w_old - H^(-1) * gradient
Spadek gradientowy:      w_new = w_old - lr * gradient
```

Metoda Newtona zbiega się szybciej, ponieważ Hessian "przeskalowuje" gradient -- strome kierunki dostają mniejsze kroki, płaskie kierunki dostają większe.

Haczyk: dla sieci neuronowej z N parametrami, Hessian to N x N. Model z 1 milionem parametrów potrzebowałby macierzy o bilionie elementów. Dlatego używamy przybliżeń.

| Metoda | Czego używa | Koszt | Zbieżność |
|--------|-------------|------|-------------|
| Spadek gradientowy | Tylko pierwsze pochodne | O(N) na krok | Wolna (liniowa) |
| Metoda Newtona | Pełny Hessian | O(N^3) na krok | Szybka (kwadratowa) |
| L-BFGS | Przybliżony Hessian z historii gradientów | O(N) na krok | Średnia (nadliniowa) |
| Adam | Adaptacyjne współczynniki na parametr (diagonalne przybliżenie Hessianu) | O(N) na krok | Średnia |
| Naturalny gradient | Macierz informacji Fishera (statystyczny Hessian) | O(N^2) na krok | Szybka |

W praktyce Adam jest domyślnym optymalizatorem dla głębokiego uczenia. Przybliża informację drugiego rzędu tanio, śledząc średnią bieżącą i wariancję gradientów na parametr.

### Przybliżenie szeregiem Taylora

Dowolną gładką funkcję można przybliżyć lokalnie wielomianem:

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

Im więcej członów uwzględnisz, tym lepsze przybliżenie -- ale tylko blisko punktu x.

**Dlaczego szereg Taylora ma znaczenie dla ML:**

- **Taylor pierwszego rzędu = spadek gradientowy.** Gdy używasz f(x + h) ~ f(x) + f'(x)*h, robisz liniowe przybliżenie. Spadek gradientowy minimalizuje ten model liniowy, wybierając h = -lr * f'(x).

- **Taylor drugiego rzędu = metoda Newtona.** Używając f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2, dostajesz model kwadratowy. Jego minimalizacja daje h = -f'(x)/f''(x) -- krok Newtona.

- **Projektowanie funkcji straty.** MSE i cross-entropia są gładkie, co oznacza, że ich rozwinięcia Taylora są dobrze uwarunkowane. To nie przypadek. Gładkie straty sprawiają, że optymalizacja jest przewidywalna.

```
Rząd przybliżenia    Co przechwytuje    Metoda optymalizacji
-------------------   -----------------   -------------------
0 rząd (stała)        Tylko wartość       Szukanie losowe
1 rząd (liniowy)      Nachylenie          Spadek gradientowy
2 rząd (kwadratowy)   Krzywizna           Metoda Newtona
Wyższe rzędy          Drobniejsza struktura   Rzadko używane w ML
```

Kluczowa intuicja: cała optymalizacja oparta na gradientach polega na lokalnym przybliżaniu funkcji straty i przejściu do minimum tego przybliżenia.

### Całki w ML

Pochodne mówią o tempie zmian. Całki obliczają akumulacje -- pole pod krzywą.

W ML rzadko obliczasz całki ręcznie, ale koncepcja jest wszechobecna:

**Prawdopodobieństwo.** Dla ciągłej zmiennej losowej z gęstością p(x):
```
P(a < X < b) = całka od a do b z p(x) dx
```
Pole pod krzywą gęstości prawdopodobieństwa między a i b to prawdopodobieństwo znalezienia się w tym zakresie.

**Wartość oczekiwana.** Średni wynik ważony prawdopodobieństwem:
```
E[f(X)] = całka z f(x) * p(x) dx
```
Oczekiwana strata względem rozkładu danych to całka. Trenowanie minimalizuje empiryczne przybliżenie tego.

**Dywergencja KL.** Mierzy, jak bardzo dwa rozkłady się różnią:
```
KL(p || q) = całka z p(x) * log(p(x) / q(x)) dx
```
Używana w VAE, dystylacji wiedzy i wnioskowaniu bayesowskim.

**Stałe normalizacyjne.** We wnioskowaniu bayesowskim:
```
p(w | dane) = p(dane | w) * p(w) / całka z p(dane | w) * p(w) dw
```
Mianownik to całka po wszystkich możliwych wartościach parametrów. Często jest nieobliczalna, dlatego używamy przybliżeń jak MCMC i wnioskowanie wariacyjne.

| Koncepcja całkowa | Gdzie pojawia się w ML |
|-----------------|----------------------|
| Pole pod krzywą | Prawdopodobieństwo z funkcji gęstości |
| Wartość oczekiwana | Funkcje straty, minimalizacja ryzyka |
| Dywergencja KL | VAE, optymalizacja polityki, dystylacja |
| Normalizacja | Rozkłady a posteriori Bayesa, mianownik softmaxa |
| Wiarygodność brzegowa | Porównywanie modeli, dolna granica dowodu (ELBO) |

### Wielowymiarowa reguła łańcuchowa w grafie obliczeniowym

Reguła łańcuchowa nie dotyczy tylko funkcji skalarnych w linii. W sieci neuronowej zmienne rozgałęziają się i łączą. Oto jak gradienty płyną przez proste przejście w przód:

```mermaid
graph LR
    x["x (wejście)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"f. straty"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

Przejście wsteczne oblicza gradienty od prawej do lewej:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

Każda strzałka mnoży przez lokalną pochodną. Gradient dla dowolnego parametru to iloczyn wszystkich lokalnych pochodnych na ścieżce od straty do tego parametru. Gdy ścieżki się rozgałęziają i łączą, sumujesz wkłady (wielowymiarowa reguła łańcuchowa).

To wszystko, czym jest wsteczna propagacja: reguła łańcuchowa zastosowana systematycznie przez graf obliczeniowy, od wyjścia do wejść.

### Macierz Jacobiego

Gdy funkcja mapuje wektor na wektor (jak warstwa sieci neuronowej), jej pochodna jest macierzą. Jacobian zawiera każdą pochodną cząstkową każdego wyjścia względem każdego wejścia.

Dla f: R^n -> R^m, Jacobian J jest macierzą m x n:

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

Nie będziesz obliczać Jacobianów ręcznie dla sieci neuronowych. PyTorch to obsługuje. Ale wiedza o ich istnieniu pomaga zrozumieć kształty we wstecznej propagacji: jeśli warstwa mapuje R^n na R^m, jej Jacobian to m x n. Gradient płynie wstecz przez transpozycję tej macierzy.

### Dlaczego to ma znaczenie dla sieci neuronowych

Każda waga w sieci neuronowej dostaje gradient. Gradient mówi, jak dostosować tę wagę, by zmniejszyć stratę.

```mermaid
graph LR
    subgraph Forward["Przejście w przód"]
        I["wejście"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["strata"]
    end
```

```mermaid
graph RL
    subgraph Backward["Przejście wsteczne"]
        dL["dL/dstrata"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

Każda aktualizacja wagi:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

Przejście w przód oblicza predykcję i stratę. Przejście wsteczne oblicza gradient straty względem każdej wagi. Następnie każda waga robi mały krok w dół. Powtórz miliony razy. To jest głębokie uczenie.

```figure
derivative-tangent
```

## Build It

### Krok 1: Pochodna numeryczna od podstaw

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numeryczna={numerical:.6f}  analityczna={analytical:.1f}")
```

Pochodna numeryczna zgadza się z analityczną do wielu miejsc po przecinku.

### Krok 2: Pochodne cząstkowe i gradienty

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Gradient numeryczny w (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Gradient analityczny w (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### Krok 3: Spadek gradientowy do znalezienia minimum f(x) = x^2

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"krok {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

Zaczynając w x=5, każdy krok przybliża do x=0 (minimum).

### Krok 4: Spadek gradientowy na funkcji 2D

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"krok {step:2d}  punkt=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### Krok 5: Porównanie pochodnych numerycznych i analitycznych

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Funkcja':<12} {'Numeryczna':>12} {'Analityczna':>12} {'Błąd':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### Krok 6: Obliczanie Hessianu numerycznie

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Hessian siodłowy: {H_saddle}")  # [[2, 0], [0, -2]] -- mieszane znaki
print(f"Hessian misy:   {H_bowl}")    # [[2, 0], [0, 2]]  -- oba dodatnie
```

Hessian funkcji siodłowej ma wartości własne 2 i -2 (mieszane znaki, potwierdzające punkt siodłowy). Misa ma wartości własne 2 i 2 (oba dodatnie, potwierdzające minimum).

### Krok 7: Przybliżenie Taylora w akcji

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  rząd1={t1:.4f}  rząd2={t2:.4f}")
```

Blisko x0=0, sin(x) ~ x (Taylor pierwszego rzędu). Przybliżenie jest doskonałe dla małych h, ale załamuje się dla dużych h. Dlatego spadek gradientowy działa najlepiej z małymi współczynnikami uczenia -- każdy krok zakłada, że liniowe przybliżenie jest dokładne.

### Krok 8: Dlaczego to ma znaczenie dla sieci neuronowej

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoka {epoch:3d}  w={w:.4f}  b={b:.4f}  strata={total_loss:.6f}")

print(f"\nNauczone: y = {w:.2f}x + {b:.2f}")
print(f"Rzeczywiste:  y = 2x + 1")
```

Każda pętla treningowa oparta na gradientach podąża za tym wzorcem: przewiduj, oblicz stratę, oblicz gradienty, zaktualizuj wagi.

## Use It

Z NumPy te same operacje są szybsze i bardziej zwięzłe:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Nauczone: y = {w:.2f}x + {b:.2f}")
```

Właśnie zbudowałeś spadek gradientowy od podstaw. PyTorch automatyzuje obliczanie gradientów, ale pętla aktualizacji jest identyczna.

## Ćwiczenia

1. Zaimplementuj `numerical_second_derivative(f, x)` używając `numerical_derivative` wywołanej dwukrotnie. Zweryfikuj, że druga pochodna x^3 w x=2 wynosi 12.
2. Użyj spadku gradientowego do znalezienia minimum f(x, y) = (x - 3)^2 + (y + 1)^2. Zacznij od (0, 0). Odpowiedź powinna zbiec do (3, -1).
3. Dodaj pęd do pętli spadku gradientowego: utrzymuj wektor prędkości, który akumuluje przeszłe gradienty. Porównaj szybkość zbieżności z pędem i bez na f(x) = x^4 - 3x^2.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| Pochodna | "Nachylenie" | Tempo zmian funkcji w punkcie. Mówi, o ile zmienia się wyjście na jednostkę zmiany wejścia. |
| Pochodna cząstkowa | "Pochodna jednej zmiennej" | Pochodna względem jednej zmiennej, podczas gdy wszystkie pozostałe są utrzymywane stałe. |
| Gradient | "Kierunek największego wzrostu" | Wektor wszystkich pochodnych cząstkowych. Wskazuje kierunek, w którym funkcja rośnie najszybciej. |
| Spadek gradientowy | "Idź w dół" | Odejmij gradient (razy współczynnik uczenia) od parametrów, aby zmniejszyć stratę. Rdzeń trenowania sieci neuronowych. |
| Współczynnik uczenia | "Wielkość kroku" | Skalar kontrolujący, jak duży jest każdy krok spadku gradientowego. Za duży: rozbieżność. Za mały: wolna zbieżność. |
| Reguła łańcuchowa | "Pomnóż pochodne" | Reguła różniczkowania funkcji złożonych: df/dx = df/dg * dg/dx. Matematyczna podstawa wstecznej propagacji. |
| Jacobian | "Macierz pochodnych" | Gdy funkcja mapuje wektory na wektory, Jacobian to macierz wszystkich pochodnych cząstkowych wyjść względem wejść. |
| Pochodna numeryczna | "Różnice skończone" | Przybliżenie pochodnej przez obliczenie funkcji w dwóch pobliskich punktach i obliczenie nachylenia między nimi. |
| Wsteczna propagacja | "Automatyczne różniczkowanie wsteczne" | Obliczanie gradientów warstwa po warstwie od wyjścia do wejścia używając reguły łańcuchowej. Jak sieci neuronowe się uczą. |
| Hessian | "Macierz drugich pochodnych" | Macierz wszystkich pochodnych cząstkowych drugiego rzędu. Opisuje krzywiznę funkcji. Dodatnio określony Hessian w punkcie krytycznym oznacza minimum lokalne. |
| Szereg Taylora | "Przybliżenie wielomianowe" | Przybliżanie funkcji w pobliżu punktu za pomocą jej pochodnych: f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... Podstawa zrozumienia, dlaczego spadek gradientowy i metoda Newtona działają. |
| Całka | "Pole pod krzywą" | Akumulacja wielkości w zakresie. W ML całki definiują prawdopodobieństwa, wartości oczekiwane i dywergencję KL. |
