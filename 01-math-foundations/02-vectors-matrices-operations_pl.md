# Wektory, Macierze i Operacje

> Każda sieć neuronowa to tylko mnożenie macierzy z dodatkowymi krokami.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minut

## Learning Objectives

- Zbuduj klasę Matrix z operacjami elementarnymi, mnożeniem macierzy, transpozycją, wyznacznikiem i odwracaniem
- Rozróżnij mnożenie elementarne od mnożenia macierzy i wyjaśnij, kiedy stosować każde z nich
- Zaimplementuj pojedynczą gęstą warstwę sieci neuronowej (`relu(W @ x + b)`) używając tylko własnoręcznie napisanej klasy Matrix
- Wyjaśnij zasady broadcastingu i jak dodawanie biasu działa w frameworkach sieci neuronowych

## Problem

Chcesz zbudować sieć neuronową. Czytasz kod i widzisz to:

```
output = activation(weights @ input + bias)
```

Ten `@` to mnożenie macierzy. `weights` to macierz. `input` to wektor. Jeśli nie wiesz, co te operacje robią, ta linia jest magią. Jeśli wiesz, to jest całe przejście w przód warstwy w trzech operacjach.

Każdy obraz przetwarzany przez twój model to macierz wartości pikseli. Każdy embedding słowa to wektor. Każda warstwa każdej sieci neuronowej to przekształcenie macierzowe. Nie możesz budować systemów AI bez biegłości w operacjach macierzowych, tak jak nie możesz pisać kodu bez rozumienia zmiennych.

Ta lekcja buduje tę biegłość od podstaw.

## Koncepcja

### Wektory: uporządkowane listy liczb

Wektor to lista liczb z kierunkiem i długością. W AI wektory reprezentują punkty danych, cechy lub parametry.

```
v = [3, 4]        -- wektor 2D
w = [1, 0, -2]    -- wektor 3D
```

Wektor 2D `[3, 4]` wskazuje na współrzędne (3, 4) na płaszczyźnie. Jego długość to 5 (trójkąt 3-4-5).

### Macierze: siatki liczb

Macierz to dwuwymiarowa siatka. Wiersze i kolumny. Macierz m x n ma m wierszy i n kolumn.

```
A = | 1  2  3 |     -- macierz 2x3 (2 wiersze, 3 kolumny)
    | 4  5  6 |
```

W sieciach neuronowych macierze wag przekształcają wektory wejściowe w wektory wyjściowe. Warstwa z 784 wejściami i 128 wyjściami używa macierzy wag 128x784.

### Dlaczego kształty mają znaczenie

Mnożenie macierzy ma ścisłą regułę: `(m x n) @ (n x p) = (m x p)`. Wewnętrzne wymiary muszą być zgodne.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  wagi          wejście      wyjście

Wewnętrzne wymiary: 784 = 784  -- poprawne
```

Jeśli dostajesz błąd niezgodności kształtów w PyTorch, to dlatego.

### Mapa operacji

| Operacja | Co robi | Zastosowanie w sieciach neuronowych |
|-----------|-------------|-------------------|
| Dodawanie | Łączenie elementarne | Dodawanie biasu do wyjścia |
| Mnożenie przez skalar | Skalowanie każdego elementu | Współczynnik uczenia * gradienty |
| Mnożenie macierzy | Przekształcanie wektorów | Przejście w przód warstwy |
| Transpozycja | Odwracanie wierszy i kolumn | Wsteczna propagacja |
| Wyznacznik | Pojedyncza liczba podsumowująca | Sprawdzanie odwracalności |
| Odwrotność | Cofanie przekształcenia | Rozwiązywanie układów liniowych |
| Jednostkowa | Macierz "nic nierobiąca" | Inicjalizacja, połączenia resztkowe |

### Mnożenie elementarne vs macierzowe

To rozróżnienie stale dezorientuje początkujących.

Elementarne: pomnóż odpowiadające sobie pozycje. Obie macierze muszą mieć ten sam kształt.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Mnożenie macierzy: iloczyny skalarne wierszy i kolumn. Wewnętrzne wymiary muszą być zgodne.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Różne operacje, różne wyniki, różne reguły.

### Broadcasting

Gdy dodajesz wektor biasu do macierzy wyjść, kształty nie są zgodne. Broadcasting rozciąga mniejszą tablicę, by pasowała.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting rozciąga wektor w poprzek wierszy:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Każdy nowoczesny framework robi to automatycznie. Zrozumienie tego zapobiega dezorientacji, gdy kształty wydają się niepoprawne, ale kod działa.

```figure
vector-projection
```

## Build It

### Krok 1: Klasa Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Krok 2: Klasa Matrix z podstawowymi operacjami

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Krok 3: Zobacz, jak to działa

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Krok 4: Połącz z sieciami neuronowymi

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

To jest pojedyncza gęsta warstwa: `output = relu(W @ x + b)`. Każda gęsta warstwa w każdej sieci neuronowej robi dokładnie to.

## Use It

NumPy robi wszystko powyżej w mniejszej liczbie linii i o rzędy wielkości szybciej.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Operator `@` w Pythonie wywołuje `__matmul__`. NumPy implementuje go zoptymalizowanymi procedurami BLAS napisanymi w C i Fortranie. Ta sama matematyka, 100x szybciej.

Broadcasting w NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy automatycznie rozgłasza 1D bias w poprzek obu wierszy. Tak działa dodawanie biasu w każdym frameworku sieci neuronowych.

## Ship It

Ta lekcja produkuje prompt do nauczania operacji macierzowych przez geometryczną intuicję. Zobacz `outputs/prompt-matrix-operations.md`.

Klasa Matrix zbudowana tutaj jest fundamentem dla mini frameworka sieci neuronowych, który budujemy w Phase 3, Lesson 10.

## Ćwiczenia

1. **Zweryfikuj odwrotność.** Pomnóż `A @ A.inverse_2x2()` i potwierdź, że otrzymujesz macierz jednostkową. Wypróbuj z trzema różnymi macierzami 2x2. Co się dzieje, gdy wyznacznik wynosi zero?

2. **Zaimplementuj odwrotność 3x3.** Rozszerz klasę Matrix, aby obliczać odwrotności dla macierzy 3x3 używając metody adjugaty. Przetestuj przeciwko `np.linalg.inv` z NumPy.

3. **Zbuduj dwuwarstwową sieć.** Używając tylko swojej klasy Matrix (bez NumPy), stwórz dwuwarstwową sieć neuronową: wejście (3) -> ukryta (4) -> wyjście (2). Zainicjalizuj losowe wagi, wykonaj przejście w przód i zweryfikuj, że wszystkie kształty są poprawne.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|----------------------|
| Wektor | "Strzałka" | Uporządkowana lista liczb. W AI: punkt w przestrzeni wysokowymiarowej. |
| Macierz | "Tabela liczb" | Przekształcenie liniowe. Mapuje wektory z jednej przestrzeni do drugiej. |
| Mnożenie macierzy | "Po prostu pomnóż liczby" | Iloczyny skalarne między każdym wierszem pierwszej macierzy a każdą kolumną drugiej. Kolejność ma znaczenie. |
| Transpozycja | "Odwróć" | Zamień wiersze i kolumny. Zmienia macierz m x n w n x m. Kluczowe we wstecznej propagacji. |
| Wyznacznik | "Jakaś liczba z macierzy" | Mierzy, jak bardzo macierz skaluje pole (2D) lub objętość (3D). Zero oznacza, że przekształcenie zgniata wymiar. |
| Odwrotność | "Cofnij macierz" | Macierz odwracająca przekształcenie. Istnieje tylko wtedy, gdy wyznacznik nie jest zerem. |
| Macierz jednostkowa | "Nudna macierz" | Macierzowy odpowiednik mnożenia przez 1. Używana w połączeniach resztkowych (ResNety). |
| Broadcasting | "Magiczne dopasowanie kształtów" | Rozciąganie mniejszej tablicy, by pasowała do większej przez powtarzanie wzdłuż brakujących wymiarów. |
| Elementarne | "Zwykłe mnożenie" | Pomnóż odpowiadające sobie pozycje. Obie tablice muszą mieć ten sam kształt (lub być broadcastowalne). |
