# Perceptron

> Perceptron jest atomem sieci neuronowych. Rozłóż go na części, a znajdziesz wagi, obciążenie i decyzję.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Zaimplementuj perceptron od zera w Pythonie, włączając regułę aktualizacji wag i funkcję aktywacji skokowej
- Wyjaśnij, dlaczego pojedynczy perceptron może rozwiązywać tylko problemy liniowo separowalne i zademonstruj przypadek niepowodzenia XOR
- Zbuduj wielowarstwowy perceptron, łącząc bramki OR, NAND i AND, aby rozwiązać XOR
- Wytrenuj dwuwarstwową sieć z aktywacją sigmoidalną i wsteczną propagacją, aby automatycznie nauczyć się XOR

## The Problem

Znasz wektory i iloczyny skalarne. Wiesz, że macierz przekształca wejścia w wyjścia. Ale w jaki sposób maszyna *uczy się*, jakiego przekształcenia użyć?

Perceptron odpowiada na to pytanie. To najprostsza możliwa maszyna ucząca się: weź kilka wejść, pomnóż przez wagi, dodaj obciążenie i podejmij binarną decyzję. Następnie dostosuj. To wszystko. Każda kiedykolwiek zbudowana sieć neuronowa to warstwy tego pomysłu ułożone jedna na drugiej.

Zrozumienie perceptronu oznacza zrozumienie, co "uczenie się" naprawdę oznacza w kodzie: dostosowywanie liczb, aż wynik będzie zgodny z rzeczywistością.

## The Concept

### Jeden Neuron, Jedna Decyzja

Perceptron przyjmuje n wejść, mnoży każde przez wagę, sumuje je, dodaje obciążenie i przepuszcza wynik przez funkcję aktywacji.

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

Funkcja skokowa jest bezwzględna: jeśli ważona suma plus obciążenie jest >= 0, wyjście wynosi 1. W przeciwnym razie wyjście wynosi 0.

```
step(z) = 1  if z >= 0
           0  if z < 0
```

To jest klasyfikator liniowy. Wagi i obciążenie definiują linię (lub hiperpłaszczyznę w wyższych wymiarach), która dzieli przestrzeń wejściową na dwa regiony.

### Granica Decyzyjna

Dla dwóch wejść perceptron rysuje linię w przestrzeni 2D:

```
  x2
  ┤
  │  Klasa 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Klasa 2
  │            /        (1)
  ┼───────────/──────────── x1
```

Wszystko po jednej stronie linii daje wynik 0. Wszystko po drugiej stronie daje wynik 1. Trenowanie przesuwa tę linię, aż prawidłowo rozdzieli klasy.

### Reguła Uczenia

Reguła uczenia perceptronu jest prosta:

```
Dla każdego przykładu treningowego (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    Dla każdej wagi:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

Jeśli przewidywanie jest poprawne, error = 0, nic się nie zmienia. Jeśli przewiduje 0, ale powinno być 1, wagi rosną. Jeśli przewiduje 1, ale powinno być 0, wagi maleją. Współczynnik uczenia kontroluje, jak duża jest każda korekta.

### Problem XOR

Tutaj następuje załamanie. Spójrz na te bramki logiczne:

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND i OR są liniowo separowalne: możesz narysować pojedynczą linię, aby oddzielić 0 od 1. XOR nie jest. Żadna pojedyncza linia nie może oddzielić [0,1] i [1,0] od [0,0] i [1,1].

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       linia działa!          żadna linia nie działa!
```

To jest fundamentalne ograniczenie. Pojedynczy perceptron może rozwiązywać tylko problemy liniowo separowalne. Minsky i Papert udowodnili to w 1969 roku i prawie zabili badania nad sieciami neuronowymi na dekadę.

Rozwiązanie: ułóż perceptrony w warstwy. Wielowarstwowy perceptron może rozwiązać XOR, łącząc dwie liniowe decyzje w nieliniową.

```figure
perceptron-boundary
```

## Build It

### Krok 1: Klasa Perceptron

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### Krok 2: Trenowanie na bramkach logicznych

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### Krok 3: Obserwuj, jak XOR zawodzi

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

To nigdy nie osiągnie zbieżności. To jest twardy dowód, że pojedynczy perceptron nie może nauczyć się XOR.

### Krok 4: Rozwiąż XOR za pomocą dwóch warstw

Sztuczka: XOR = (x1 OR x2) AND NOT (x1 AND x2). Połącz trzy perceptrony:

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

Wszystkie cztery przypadki poprawne. Układanie perceptronów w warstwy tworzy granice decyzyjne, których żaden pojedynczy perceptron nie jest w stanie wyprodukować.

### Krok 5: Trenuj Dwuwarstwową Sieć

Krok 4 ręcznie ustawił wagi. To działa dla XOR, ale nie dla rzeczywistych problemów, gdzie nie znasz poprawnych wag z góry. Rozwiązanie: zastąp funkcję skokową sigmoidą i naucz się wag automatycznie poprzez wsteczną propagację.

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

Dwie kluczowe różnice w stosunku do Kroku 4. Po pierwsze, sigmoida zastępuje funkcję skokową -- jest gładka, więc istnieją gradienty. Po drugie, metoda `train` propaguje błąd wstecz od wyjścia do ukrytej warstwy, dostosowując każdą wagę proporcjonalnie do jej wkładu w błąd. To jest wsteczna propagacja w 20 liniach.

To jest most do Lekcji 03. Matematyka stojąca za `d_output` i `hidden_deltas` to reguła łańcuchowa zastosowana do grafu sieci. Wyprowadzimy ją tam poprawnie.

## Use It

Wszystko, co właśnie zbudowałeś od zera, istnieje w jednym imporcie:

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

Pięć linii. Twoja 30-liniowa klasa `Perceptron` robi to samo. Wersja sklearn dodaje sprawdzanie zbieżności, wiele funkcji strat i obsługę rzadkich wejść -- ale główna pętla jest identyczna: ważona suma, funkcja skokowa, aktualizacja wag przy błędzie.

Prawdziwa różnica pojawia się przy skali. Co zmienia się w produkcyjnych sieciach:

- Funkcja skokowa staje się sigmoidą, ReLU lub innymi gładkimi aktywacjami
- Wagi są uczone automatycznie przez wsteczną propagację (Lekcja 03)
- Warstwy stają się głębsze: 3, 10, 100+ warstw
- Ta sama zasada obowiązuje: każda warstwa tworzy nowe cechy z wyjść poprzedniej warstwy

Pojedynczy perceptron może rysować tylko proste linie. Ułóż je w stos, a możesz narysować dowolny kształt.

## Ship It

Ta lekcja produkuje:
- `outputs/skill-perceptron.md` - umiejętność opisująca, kiedy potrzebne są architektury jednowarstwowe vs wielowarstwowe

## Exercises

1. Wytrenuj perceptron na bramce NAND (bramka uniwersalna -- każdy układ logiczny może być zbudowany z NAND). Zweryfikuj, że jego wagi i obciążenie tworzą prawidłową granicę decyzyjną.
2. Zmodyfikuj klasę Perceptron, aby śledzić granicę decyzyjną (w1*x1 + w2*x2 + b = 0) w każdej epoce. Wypisz, jak linia przesuwa się podczas trenowania na bramce AND.
3. Zbuduj perceptron z 3 wejściami, który zwraca 1 tylko wtedy, gdy co najmniej 2 z 3 wejść to 1 (funkcja większościowa). Czy jest to liniowo separowalne? Dlaczego?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|----------------------|
| Perceptron | "Sztuczny neuron" | Klasyfikator liniowy: iloczyn skalarny wejść i wag, plus obciążenie, przez funkcję skokową |
| Weight (waga) | "Jak ważne jest wejście" | Mnożnik skalujący wkład każdego wejścia w decyzję |
| Bias (obciążenie) | "Próg" | Stała przesuwająca granicę decyzyjną, pozwalająca perceptronowi zadziałać nawet przy zerowych wejściach |
| Activation function (funkcja aktywacji) | "To, co ściska wartości" | Funkcja zastosowana po ważonej sumie -- funkcja skokowa dla perceptronów, sigmoida/ReLU dla nowoczesnych sieci |
| Linearly separable (liniowo separowalne) | "Możesz narysować linię między nimi" | Zbiór danych, w którym pojedyncza hiperpłaszczyzna może idealnie rozdzielić klasy |
| XOR problem (problem XOR) | "To, czego perceptrony nie potrafią zrobić" | Dowód, że jednowarstwowe sieci nie mogą nauczyć się funkcji nieliniowo separowalnych |
| Decision boundary (granica decyzyjna) | "Gdzie klasyfikator się przełącza" | Hiperpłaszczyzna w*x + b = 0 dzieląca przestrzeń wejściową na dwie klasy |
| Multi-layer perceptron (wielowarstwowy perceptron) | "Prawdziwa sieć neuronowa" | Perceptrony ułożone w warstwy, gdzie wyjście każdej warstwy zasila wejście następnej |

## Further Reading

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- oryginalna praca, która zapoczątkowała wszystko
- Minsky & Papert, "Perceptrons" (1969) -- książka, która udowodniła, że XOR jest nierozwiązywalny przez jednowarstwowe sieci i zabiła badania nad perceptronami na dekadę
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- darmowe online, najlepsze wizualne wyjaśnienie, jak perceptrony łączą się w sieci