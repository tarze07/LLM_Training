# Wprowadzenie do JAX

> PyTorch mutuje tensory. TensorFlow buduje grafy. JAX kompiluje czyste funkcje. Ta ostatnia zmienia sposób, w jaki myślisz o głębokim uczeniu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Learning Objectives

- Pisz kod sieci neuronowych w stylu czystych funkcji używając funkcyjnego API JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Wyjaśnij kluczową różnicę projektową między zachłanną mutacją PyTorch a modelem kompilacji funkcyjnej JAX
- Zastosuj kompilację jit i wektoryzację vmap, aby przyspieszyć pętle treningowe w porównaniu do naiwnego Pythona
- Trenuj prostą sieć w JAX i skontrastuj jawne zarządzanie stanem z podejściem obiektowym PyTorch

## The Problem

Umiesz budować sieci neuronowe w PyTorch. Definiujesz `nn.Module`, wywołujesz `.backward()`, wykonujesz krok optymalizatora. Działa. Miliony ludzi go używają.

Ale PyTorch ma ograniczenie wbudowane w swoje DNA: śledzi operacje zachłannie, jedną na raz, w Pythonie. Każde `tensor + tensor` to osobne uruchomienie jądra. Każdy krok treningowy interpretuje na nowo ten sam kod Pythona. Działa to dobrze, dopóki nie musisz trenować modelu z 540 miliardami parametrów na 2 048 TPU. Wtedy narzut cię zabija.

Google DeepMind trenuje Gemini na JAX. Anthropic trenował Claude na JAX. To nie są małe operacje – to największe na Ziemi przebiegi trenowania sieci neuronowych. Wybrali JAX, ponieważ traktuje twoją pętlę treningową jako kompilowalny program, a nie sekwencję wywołań Pythona.

JAX to NumPy z trzema supermocami: automatyczne różniczkowanie, kompilacja JIT do XLA i automatyczna wektoryzacja. Piszesz funkcję, która przetwarza jeden przykład. JAX daje ci funkcję, która przetwarza partię, oblicza gradienty, kompiluje do kodu maszynowego i działa na wielu urządzeniach. Bez zmieniania oryginalnej funkcji.

## The Concept

### The JAX Philosophy

JAX to framework funkcyjny. Żadnych klas, żadnego mutowalnego stanu, żadnej metody `.backward()`. Zamiast tego:

| PyTorch | JAX |
|---------|-----|
| Klasa `nn.Module` ze stanem | Czysta funkcja: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Zachłanne wykonanie | Kompilacja JIT przez XLA |
| Ręczna pętla `for x in batch:` | `jax.vmap(f)` auto-wektoryzacja |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-równoległość |
| Mutowalne `model.parameters()` | Niemutowalne pytree tablic |

To nie jest preferencja stylistyczna. To ograniczenie kompilatora. Kompilacja JIT wymaga czystych funkcji – te same wejścia zawsze dają te same wyjścia, bez efektów ubocznych. To ograniczenie umożliwia 100-krotne przyspieszenia.

### jax.numpy: The Familiar Surface

JAX implementuje na nowo API NumPy na akceleratorach:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Te same nazwy funkcji. Te same zasady broadcastingu. Ta sama semantyka wycinków. Ale tablice żyją na GPU/TPU, a każda operacja jest śledzona przez kompilator.

Jedna krytyczna różnica: tablice JAX są niemutowalne. Nie ma `a[0] = 5`. Zamiast tego: `a = a.at[0].set(5)`. To wydaje się dziwne przez tydzień, potem klikasz – niemutowalność sprawia, że transformacje takie jak `grad`, `jit` i `vmap` są komponowalne.

### jax.grad: Functional Autodiff

PyTorch dołącza gradienty do tensorów (`.grad`). JAX dołącza gradienty do funkcji.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` przyjmuje funkcję i zwraca nową funkcję, która oblicza gradient. Bez wywołania `.backward()`. Bez grafu obliczeniowego przechowywanego na tensorach. Gradient jest po prostu kolejną funkcją, którą możesz wywołać, komponować lub kompilować JIT.

To komponuje się dowolnie:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Drugie pochodne. Trzecie pochodne. Jacobiany. Hesjany. Wszystko przez komponowanie `grad`. PyTorch też może to zrobić (`torch.autograd.functional.hessian`), ale jest to doczepione. W JAX jest to fundament.

Ograniczenie: `grad` działa tylko na czystych funkcjach. Żadnych print w środku (działają podczas śledzenia, nie wykonania). Żadnej mutacji stanu zewnętrznego. Żadnego generowania liczb losowych bez jawnego zarządzania kluczami.

### jit: Compile to XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Przy pierwszym wywołaniu JAX śledzi funkcję – rejestruje, które operacje mają miejsce, bez ich wykonywania. Następnie przekazuje to śledzenie do XLA (Accelerated Linear Algebra), kompilatora Google dla TPU i GPU. XLA scala operacje, eliminuje zbędne kopie pamięci i generuje zoptymalizowany kod maszynowy.

Kolejne wywołania całkowicie pomijają Pythona. Skompilowany kod działa na akceleratorze z szybkością C++.

Kiedy JIT pomaga:
- Kroki treningowe (ta sama wielokrotnie powtarzana operacja)
- Inferencja (ten sam model, różne wejścia)
- Dowolna funkcja wywoływana więcej niż raz z wejściami o podobnym kształcie

Kiedy JIT szkodzi:
- Funkcje z przepływem sterowania Pythona zależnym od wartości (`if x > 0` gdzie x jest śledzoną tablicą)
- Obliczenia jednorazowe (narzut kompilacji przekracza czas wykonania)
- Debugowanie (śledzenie ukrywa faktyczne wykonanie)

Ograniczenie przepływu sterowania jest realne. `jax.lax.cond` zastępuje `if/else`. `jax.lax.scan` zastępuje pętle `for`. To nie jest opcjonalne – to cena kompilacji.

### vmap: Automatic Vectorization

Piszesz funkcję, która przetwarza jeden przykład:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` podnosi ją do przetwarzania partii:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` oznacza: nie grupuj po `params` (współdzielone), grupuj po osi 0 dla `x`. Żadnej ręcznej pętli `for`. Żadnego przekształcania. Żadnego przeciągania wymiaru partii. JAX znajduje wymiar partii i wektoryzuje całe obliczenie.

To nie jest lukier składniowy. `vmap` generuje scalony kod wektoryzowany, który działa 10-100 razy szybciej niż pętla Pythona. I komponuje się z `jit` i `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradienty dla każdego przykładu. Jedna linia. To jest prawie niemożliwe w PyTorch bez hacków.

### pmap: Data Parallelism Across Devices

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` replikuje funkcję na wszystkich dostępnych urządzeniach (GPU/TPU) i dzieli partię. Wewnątrz funkcji `jax.lax.pmean` i `jax.lax.psum` synchronizują gradienty między urządzeniami.

Google trenuje Gemini na tysiącach układów TPU v5e używając `pmap` (i jego następcy `shard_map`). Model programowania: napisz wersję na jedno urządzenie, opakuj w `pmap`, gotowe.

### Pytrees: The Universal Data Structure

JAX operuje na "pytrees" – zagnieżdżonych kombinacjach list, krotek, słowników i tablic. Parametry twojego modelu to pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Każda transformacja JAX – `grad`, `jit`, `vmap` – wie, jak przemierzać pytrees. `jax.tree.map(f, tree)` stosuje `f` do każdego liścia. Tak optymalizatory aktualizują wszystkie parametry naraz:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Bez metody `.parameters()`. Bez rejestracji parametrów. Struktura drzewa jest modelem.

### Functional vs Object-Oriented

PyTorch przechowuje stan wewnątrz obiektów:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX używa czystych funkcji z jawnym stanem:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Parametry są przekazywane. Nic nie jest przechowywane. Nic nie jest mutowane. To sprawia, że każda funkcja jest testowalna, komponowalna i kompilowalna. Oznacza to również, że sam zarządzasz parametrami – albo używasz biblioteki takiej jak Flax lub Equinox.

### The JAX Ecosystem

JAX daje prymitywy. Biblioteki dają ergonomię:

| Biblioteka | Rola | Styl |
|---------|------|-------|
| **Flax** (Google) | Warstwy sieci neuronowych | `nn.Module` z jawnym stanem |
| **Equinox** (Patrick Kidger) | Warstwy sieci neuronowych | Oparte na pytree, Pythoniczne |
| **Optax** (DeepMind) | Optymalizatory + harmonogramy LR | Komponowalne transformacje gradientów |
| **Orbax** (Google) | Punkt kontrolny | Zapisywanie/przywracanie pytrees |
| **CLU** (Google) | Metryki + logowanie | Narzędzia pętli treningowej |

Optax to standardowa biblioteka optymalizatorów. Oddziela transformację gradientów (Adam, SGD, przycinanie) od aktualizacji parametrów, co czyni trywialnym komponowanie:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch

| Czynnik | JAX | PyTorch |
|--------|-----|---------|
| Wsparcie TPU | Pierwsza klasa (Google zbudował oba) | Utrzymywane przez społeczność (torch_xla) |
| Wsparcie GPU | Dobre (CUDA przez XLA) | Najlepsze w klasie (natywne CUDA) |
| Debugowanie | Trudne (śledzenie + kompilacja) | Łatwe (zachłanne, linijka po linijce) |
| Ekosystem | Skoncentrowany na badaniach (Flax, Equinox) | Masywny (HuggingFace, torchvision, itd.) |
| Zatrudnienie | Niszowe (Google/DeepMind/Anthropic) | Główny nurt (wszędzie) |
| Trenowanie na dużą skalę | Lepsze (XLA, pmap, mesh) | Dobre (FSDP, DeepSpeed) |
| Szybkość prototypowania | Wolniejsza (narzut funkcyjny) | Szybsza (mutuj i działaj) |
| Produkcyjna inferencja | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Kto tego używa | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Szczera odpowiedź: używaj PyTorch, chyba że masz konkretny powód, aby używać JAX. Te powody to – dostęp do TPU, potrzeba gradientów dla każdego przykładu, trenowanie na wielu urządzeniach na ogromną skalę lub praca w Google/DeepMind/Anthropic.

### Random Numbers in JAX

JAX nie ma globalnego stanu losowego. Każda operacja losowa wymaga jawnego klucza PRNG:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Na początku jest to denerwujące. Ale gwarantuje odtwarzalność między urządzeniami i kompilacjami – właściwości, której `torch.manual_seed` w PyTorch nie może zagwarantować w ustawieniach z wieloma GPU.

```figure
batchnorm-effect
```

## Build It

### Step 1: Setup and Data

Będziemy trenować 3-warstwowy MLP na MNIST używając JAX i Optax. 784 wejścia, dwie ukryte warstwy po 256 i 128 neuronów, 10 klas wyjściowych.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters

Żadnej klasy. Tylko funkcja, która zwraca pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Inicjalizacja He, wykonana ręcznie. Trzy klucze PRNG podzielone z jednego ziarna. Każda waga jest niemutowalną tablicą w zagnieżdżonym słowniku.

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Czyste funkcje. Parametry wchodzą, przewidywanie wychodzi. Żadnego `self`, żadnego przechowywanego stanu. `loss_fn` oblicza entropię krzyżową od zera -- softmax, log, ujemna średnia.

### Step 4: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` zwraca zarówno wartość straty, jak i gradienty w jednym przebiegu. Dekorator `@jax.jit` kompiluje obie funkcje do XLA. Po pierwszym wywołaniu każdy krok treningowy działa bez dotykania Pythona.

### Step 5: Training Loop

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epok. ~97% dokładności testowej. Pierwsza epoka jest wolna (kompilacja JIT). Epoki 2-10 są szybkie.

Zwróć uwagę, czego brakuje: żadnego `.zero_grad()`, żadnego `.backward()`, żadnego `.step()`. Cała aktualizacja to jedno wywołanie złożonej funkcji. Gradienty są obliczane, przekształcane przez Adama i stosowane do parametrów – wszystko wewnątrz `train_step`.

## Use It

### Flax: The Google Standard

Flax to najczęściej używana biblioteka sieci neuronowych dla JAX. Dodaje z powrotem `nn.Module`, ale z jawnym zarządzaniem stanem:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Taka sama struktura jak PyTorch, ale `params` jest oddzielone od modelu. `model.init()` tworzy parametry. `model.apply(params, x)` uruchamia forward pass. Obiekt modelu nie ma stanu.

### Equinox: The Pythonic Alternative

Equinox (autorstwa Patricka Kidgera) reprezentuje modele jako pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Sam model jest pytree. Nie trzeba `.apply()`. Parametry są po prostu liśćmi modelu. To jest bliższe sposobowi myślenia JAX.

### Optax: Composable Optimizers

Optax oddziela transformację gradientów od aktualizacji:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Przycinanie gradientów, rozgrzewanie tempa uczenia, spadek wagi – wszystko złożone jako łańcuch transformacji. Każda transformacja widzi gradienty, modyfikuje je i przekazuje do następnej. Żadnej monolitycznej klasy optymalizatora.

## Ship It

**Instalacja:**

```bash
pip install jax jaxlib optax flax
```

Dla wsparcia GPU:

```bash
pip install jax[cuda12]
```

Dla TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Pułapki wydajnościowe:**

- Pierwsze wywołanie JIT jest wolne (kompilacja). Rozgrzej przed benchmarkowaniem.
- Unikaj pętli Pythona po tablicach JAX wewnątrz JIT. Użyj `jax.lax.scan` lub `jax.lax.fori_loop`.
- `jax.debug.print()` działa wewnątrz JIT. Zwykły `print()` nie.
- Profiluj z `jax.profiler` lub TensorBoard. Kompilacja XLA może ukryć wąskie gardła.
- JAX domyślnie prealokuje 75% pamięci GPU. Ustaw `XLA_PYTHON_CLIENT_PREALLOCATE=false`, aby wyłączyć.

**Punkty kontrolne:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Ta lekcja produkuje:**
- `outputs/prompt-jax-optimizer.md` -- prompt do wyboru odpowiedniej konfiguracji optymalizatora JAX
- `outputs/skill-jax-patterns.md` -- umiejętność obejmująca wzorce funkcyjne w JAX

## Exercises

1. Dodaj dropout do MLP. W JAX dropout wymaga klucza PRNG -- przeciągnij klucz przez forward pass i podziel go dla każdej warstwy dropout. Porównaj dokładność testową z dropoutem i bez.

2. Użyj `jax.vmap`, aby obliczyć gradienty dla każdego przykładu dla partii 32 obrazów MNIST. Oblicz normę gradientu dla każdego przykładu. Które przykłady mają największe gradienty i dlaczego?

3. Zastąp ręczną funkcję forward ogólnym `mlp_forward(params, x)`, który działa dla dowolnej liczby warstw. Użyj `jax.tree.leaves`, aby automatycznie określić głębokość.

4. Porównaj benchmark kroku treningowego z `@jax.jit` i bez. Zmierz czas 100 kroków każdego. Jak duże jest przyspieszenie na twoim sprzęcie? Jaki jest narzut kompilacji przy pierwszym wywołaniu?

5. Zaimplementuj przycinanie gradientów przez złożenie `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Trenuj z przycinaniem i bez. Wykreśl normę gradientu podczas trenowania, aby zobaczyć efekt.

## Key Terms

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| XLA | "To, co sprawia, że JAX jest szybki" | Accelerated Linear Algebra -- kompilator, który scala operacje i generuje zoptymalizowane jądra GPU/TPU z grafu obliczeniowego |
| JIT | "Kompilacja just-in-time" | JAX śledzi funkcję przy pierwszym wywołaniu, kompiluje do XLA, a następnie uruchamia skompilowaną wersję przy kolejnych wywołaniach |
| Czysta funkcja | "Bez efektów ubocznych" | Funkcja, której wynik zależy tylko od wejść -- brak stanu globalnego, mutacji, losowości bez jawnych kluczy |
| vmap | "Auto-grupowanie" | Przekształca funkcję przetwarzającą jeden przykład w funkcję przetwarzającą partię, bez przepisywania |
| pmap | "Auto-równoległość" | Replikuje funkcję na wielu urządzeniach i dzieli wejściową partię |
| Pytree | "Zagnieżdżony słownik tablic" | Dowolna zagnieżdżona struktura list, krotek, słowników i tablic, którą JAX może przemierzać i przekształcać |
| Śledzenie | "Rejestrowanie obliczeń" | JAX wykonuje funkcję z abstrakcyjnymi wartościami, aby zbudować graf obliczeniowy, bez obliczania rzeczywistych wyników |
| Funkcyjna autodyf | "pochodna funkcji" | Obliczanie pochodnych przez przekształcanie funkcji, a nie przez dołączanie pamięci gradientów do tensorów |
| Optax | "Biblioteka optymalizatorów JAX" | Komponowalna biblioteka transformacji gradientów -- Adam, SGD, przycinanie, harmonogramy -- które łączą się w łańcuch |
| Flax | "nn.Module JAX" | Biblioteka sieci neuronowych Google dla JAX, dodająca abstrakcje warstw przy zachowaniu jawnego stanu |

## Further Reading

- JAX documentation: https://jax.readthedocs.io/ -- oficjalna dokumentacja, z doskonałymi tutorialami o grad, jit i vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- oryginalna praca wyjaśniająca filozofię projektową
- Flax documentation: https://flax.readthedocs.io/ -- biblioteka sieci neuronowych Google dla JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Pythoniczna alternatywa dla Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- standardowa biblioteka optymalizatorów
- "You Don't Know JAX" (Colin Raffel, 2020) -- praktyczny przewodnik po pułapkach i wzorcach JAX od jednego z autorów T5