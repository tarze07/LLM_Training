# Zbuduj Własny Mini Framework

> Zbudowałeś neurony, warstwy, sieci, wsteczną propagację, aktywacje, funkcje strat, optymalizatory, regularyzację, inicjalizację i harmonogramy LR. Wszystko jako osobne części. Teraz połącz je w framework. Nie PyTorch. Nie TensorFlow. Twój własny.

**Type:** Build
**Languages:** Python
**Prerequisites:** All of Phase 03 (Lessons 01-09)
**Time:** ~120 minutes

## Learning Objectives

- Zbuduj kompletny framework głębokiego uczenia (~500 linii) z Module, Linear, ReLU, Sigmoid, Dropout, BatchNorm, Sequential, funkcjami strat, optymalizatorami i DataLoader
- Wyjaśnij abstrakcję Module (forward, backward, parameters) i dlaczego przełączanie między trybem train/eval jest konieczne
- Połącz wszystkie komponenty w działającą pętlę treningową, która trenuje 4-warstwową sieć na klasyfikacji kołowej
- Odwzoruj każdy komponent swojego frameworka na jego odpowiednik w PyTorch (nn.Module, nn.Sequential, optim.Adam, DataLoader)

## The Problem

Masz dziesięć lekcji bloków konstrukcyjnych rozrzuconych po osobnych plikach. Klasa `Value` tutaj, pętla treningowa tam, inicjalizacja wag w innym pliku, harmonogramy tempa uczenia w jeszcze innym. Aby wytrenować sieć, kopiujesz-wklejasz z pięciu różnych lekcji i łączysz je ręcznie.

To właśnie rozwiązują frameworki. PyTorch daje Ci `nn.Module`, `nn.Sequential`, `optim.Adam`, `DataLoader` i wzorzec pętli treningowej, który łączy je w całość. TensorFlow daje Ci `keras.Layer`, `keras.Sequential`, `keras.optimizers.Adam`. To nie jest magia. To wzorce organizacyjne, które umożliwiają definiowanie, trenowanie i ocenianie sieci bez wymyślania infrastruktury na nowo za każdym razem.

Zbudujesz to samo w ~500 liniach Pythona. Bez numpy. Bez zewnętrznych zależności. Framework, który może zdefiniować dowolną sieć feedforward, trenować ją z SGD lub Adamem, dzielić dane na batche, stosować dropout i normalizację batchową, używać dowolnej aktywacji i harmonogramować tempo uczenia.

Kiedy skończysz, będziesz dokładnie rozumieć, co się dzieje, gdy piszesz `model = nn.Sequential(...)` w PyTorch. Będziesz rozumieć, dlaczego istnieją `model.train()` i `model.eval()`. Będziesz rozumieć, dlaczego `optimizer.zero_grad()` jest osobnym wywołaniem. Będziesz to wszystko rozumieć, ponieważ sam to zbudowałeś.

## The Concept

### The Module Abstraction

Każda warstwa w PyTorch dziedziczy po `nn.Module`. Moduł ma trzy obowiązki:

1. **forward()** -- oblicza wyjście na podstawie wejść
2. **parameters()** -- zwraca wszystkie trenowalne wagi
3. **backward()** -- oblicza gradienty (obsługiwane przez autograd w PyTorch, u nas jawne)

Warstwa Linear to Moduł. Aktywacja ReLU to Moduł. Warstwa dropout to Moduł. Warstwa normalizacji batchowej to Moduł. Wszystkie mają ten sam interfejs.

### Sequential Container

`nn.Sequential` łączy Moduły w łańcuch. Przejście do przodu: przepuść dane przez Moduł 1, potem Moduł 2, potem Moduł 3. Przejście wstecz: odwróć łańcuch. Sam kontener jest Modułem -- ma forward(), parameters() i backward(). To wzorzec kompozytu: sekwencja Modułów sama jest Modułem.

### Training vs Evaluation Mode

Dropout losowo zeruje neurony podczas trenowania, ale przepuszcza wszystko podczas ewaluacji. Normalizacja batchowa używa statystyk batcha podczas trenowania, ale średnich kroczących podczas ewaluacji. Metody `train()` i `eval()` przełączają to zachowanie. Każdy Moduł ma flagę `training`.

### Optimizer

Optymalizator aktualizuje parametry za pomocą ich gradientów. SGD: `param -= lr * grad`. Adam: utrzymuje estymaty momentum i wariancji, a następnie aktualizuje. Optymalizator nie zna architektury sieci -- widzi tylko płaską listę parametrów i ich gradientów.

### DataLoader

Dzielenie na batche ma znaczenie z dwóch powodów. Po pierwsze, nie możesz zmieścić całego zbioru danych w pamięci dla dużych problemów. Po drugie, gradientowa descent mini-batch dostarcza szumu, który pomaga uciekać z lokalnych minimów. DataLoader dzieli dane na batche i opcjonalnie tasuje między epokami.

### Framework Architecture

```mermaid
graph TD
    subgraph "Moduły"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>losowa maska zerująca"]
        BatchNorm["BatchNorm<br/>normalizuj aktywacje"]
    end

    subgraph "Kontenery"
        Sequential["Sequential<br/>łączy moduły w łańcuch"]
    end

    subgraph "Funkcje strat"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binarna entropia krzyżowa"]
    end

    subgraph "Optymalizatory"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>adaptacyjne momenty"]
    end

    subgraph "Dane"
        DataLoader["DataLoader<br/>batchowanie + tasowanie"]
    end

    Sequential --> |"zawiera"| Linear
    Sequential --> |"zawiera"| ReLU
    Sequential --> |"forward/backward"| MSE
    SGD --> |"aktualizuje"| Sequential
    DataLoader --> |"zasila"| Sequential
```

### Training Loop

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    loop Każda epoka
        DL->>M: batch wejść
        M->>M: forward pass (warstwa po warstwie)
        M->>L: przewidywania
        L->>L: oblicz stratę
        L->>M: backward pass (gradienty)
        M->>O: parametry + gradienty
        O->>M: zaktualizowane parametry
        O->>O: wyzeruj gradienty
    end
```

### Module Hierarchy

```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

```figure
gradient-clipping
```

## Build It

### Step 1: Module Base Class

Abstrakcyjny interfejs, który implementuje każda warstwa.

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### Step 2: Linear Layer

Podstawowy element konstrukcyjny. Przechowuje wagi i biasy, oblicza Wx + b w przód oraz gradienty wag/wejść w tył.

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### Step 3: Activation Modules

ReLU, Sigmoid i Tanh jako Moduły. Każdy zapisuje w pamięci podręcznej to, czego potrzebuje do przejścia wstecznego.

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### Step 4: Dropout Module

Losowo zeruje elementy podczas trenowania. Skaluje pozostałe elementy przez 1/(1-p), aby wartości oczekiwane pozostały takie same. Nic nie robi podczas ewaluacji.

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### Step 5: BatchNorm Module

Normalizuje aktywacje do średniej zero i jednostkowej wariancji na cechę w obrębie batcha. Utrzymuje kroczące statystyki dla trybu ewaluacji.

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### Step 6: Sequential Container

Łączy moduły w łańcuch. Forward idzie od lewej do prawej, backward od prawej do lewej.

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### Step 7: Loss Functions

MSE i binarna entropia krzyżowa. Każda zwraca wartość straty i udostępnia backward() zwracający gradient.

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### Step 8: SGD and Adam Optimizers

Oba przyjmują listę parametrów i aktualizują wagi za pomocą gradientów.

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### Step 9: DataLoader

Dzieli dane na batche, opcjonalnie tasuje każdą epokę.

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### Step 10: Train a 4-Layer Network on Circle Classification

Połącz wszystko razem. Zdefiniuj model, wybierz stratę, wybierz optymalizator, uruchom pętlę treningową.

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoka {epoch:3d} | Strata: {avg_loss:.6f} | Dokładność trening: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nDokładność test: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## Use It

Oto odpowiednik w PyTorch tego, co właśnie zbudowałeś:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

Struktura jest identyczna. `Sequential`, `Linear`, `ReLU`, `Sigmoid`, `BCELoss`, `Adam`, `zero_grad`, `backward`, `step`, `train`, `eval`. Każda koncepcja odwzorowuje się jeden do jednego. Różnica polega na tym, że PyTorch obsługuje autograd automatycznie (nie trzeba implementować backward() w każdym module), działa na GPU i był optymalizowany przez lata. Ale kości są te same.

Teraz, gdy widzisz kod PyTorch, dokładnie wiesz, co się dzieje w każdej linii. To zrozumienie jest całym sensem.

## Ship It

Ta lekcja produkuje:
- `outputs/prompt-framework-architect.md` -- prompt do projektowania architektur sieci neuronowych przy użyciu abstrakcji frameworka

## Exercises

1. Dodaj klasę `SoftmaxCrossEntropyLoss` do klasyfikacji wieloklasowej. Zastosuj softmax do przewidywań, oblicz stratę entropii krzyżowej i obsłuż łączone przejście wsteczne. Przetestuj na 3-klasowym zbiorze spiralnym.

2. Zaimplementuj harmonogramowanie tempa uczenia w optymalizatorze: dodaj metodę `set_lr()` i podłącz harmonogram cosinusowy z Lekcji 09. Trenuj klasyfikator kołowy z rozgrzewką + cosinus i porównaj ze stałym LR.

3. Dodaj metody `save()` i `load()` do Sequential, które serializują wszystkie wagi do pliku JSON i wczytują je z powrotem. Zweryfikuj, że wczytany model produkuje te same przewidywania co oryginał.

4. Zaimplementuj zanik wag (regularyzację L2) w optymalizatorze Adam. Dodaj parametr `weight_decay`, który przybliża wagi do zera w każdym kroku. Porównaj trenowanie z decay=0 vs decay=0.01.

5. Zastąp pętlę treningową pojedynczych próbek właściwą akumulacją gradientów mini-batch: akumuluj gradienty przez wszystkie próbki w batchu, następnie podziel przez rozmiar batcha i wykonaj jeden krok optymalizatora. Zmierz, czy zmienia to szybkość zbieżności.

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|----------------------|
| Module | "Warstwa" | Podstawowa abstrakcja w frameworku -- wszystko, co ma forward(), backward() i parameters() |
| Sequential | "Układaj warstwy po kolei" | Kontener łączący moduły w łańcuch, stosując je w kolejności dla forward i odwrotnej dla backward |
| Forward pass | "Uruchom sieć" | Obliczanie wyjścia poprzez przepuszczenie wejścia przez każdy moduł po kolei |
| Backward pass | "Oblicz gradienty" | Propagowanie gradientu straty przez każdy moduł w odwrotnej kolejności w celu obliczenia gradientów parametrów |
| Parameters | "Trenowalne wagi" | Wszystkie wartości w sieci, które optymalizator może aktualizować -- wagi i biasy |
| Optimizer | "Rzecz, która aktualizuje wagi" | Algorytm używający gradientów do aktualizacji parametrów, implementujący SGD, Adam lub inne reguły |
| DataLoader | "Rzecz, która podaje dane" | Iterator dzielący zbiór danych na batche, opcjonalnie tasujący między epokami |
| Training mode | "model.train()" | Flaga włączająca stochastyczne zachowanie, takie jak dropout i normalizacja batchowa ze statystykami batcha |
| Evaluation mode | "model.eval()" | Flaga wyłączająca dropout i używająca kroczących statystyk dla normalizacji batchowej |
| Zero grad | "Wyczyść gradienty" | Resetowanie wszystkich gradientów parametrów do zera przed obliczeniem gradientów następnego batcha |

## Further Reading

- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019) -- praca opisująca decyzje projektowe PyTorch
- Chollet, "Deep Learning with Python, Second Edition" (2021) -- Rozdział 3 omawia wewnętrzne działanie Keras z tą samą abstrakcją modułu/warstwy
- Johnson, "Tiny-DNN" (https://github.com/tiny-dnn/tiny-dnn) -- nagłówkowy framework głębokiego uczenia w C++ do zrozumienia wewnętrznych mechanizmów frameworka