# Wielogłowicowa Uwaga (Multi-Head Attention)

> Jedna głowa uwagi uczy się jednej relacji na raz. Osiem głów uczy się ośmiu. Głowy są darmowe. Bierz ich więcej.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Problem

Pojedyncza głowa samo-uwagi oblicza jedną macierz uwagi. Ta macierz przechwytuje jeden rodzaj relacji — zwykle ten, który minimalizuje stratę dla danego sygnału treningowego. Jeśli twoje dane zawierają zgodność podmiotu z czasownikiem, koreferencję, dyskurs dalekiego zasięgu i frazowanie składniowe, wszystko splecione razem, pojedyncza głowa rozmazuje je w jeden rozkład softmax i traci połowę sygnału.

Rozwiązanie z artykułu Vaswani z 2017 roku: uruchom kilka funkcji uwagi równolegle, każdą z własnymi projekcjami Q, K, V, i połącz wyniki. Każda głowa działa w mniejszej podprzestrzeni o wymiarze `d_model / n_heads`. Całkowita liczba parametrów pozostaje taka sama. Moc ekspresyjna rośnie.

Wielogłowicowa uwaga jest domyślną opcją, z którą każdy transformer w 2026 roku jest dostarczany. Jedynym argumentem jest *ile* głów i czy klucze i wartości dzielą projekcje (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Koncepcja

![Wielogłowicowa uwaga dzieli, skupia się, łączy](../assets/multi-head-attention.svg)

**Podziel.** Weź `X` o kształcie `(N, d_model)`. Przejdź do Q, K, V, każdy o kształcie `(N, d_model)`. Przekształć do `(N, n_heads, d_head)`, gdzie `d_head = d_model / n_heads`. Transponuj do `(n_heads, N, d_head)`.

**Skup się równolegle.** Uruchom skalowaną uwagę iloczynową wewnątrz każdej głowy. Każda głowa produkuje `(N, d_head)`. Głowy operują na różnych podprzestrzeniach osadzenia i nigdy nie komunikują się podczas samego obliczania uwagi.

**Połącz i przekształć.** Złóż głowy z powrotem do `(N, d_model)` i pomnóż przez wyuczoną macierz wyjściową `W_o` o kształcie `(d_model, d_model)`. `W_o` to miejsce, gdzie głowy mogą się mieszać.

**Dlaczego to działa.** Każda głowa może się wyspecjalizować bez konkurowania z innymi o budżet reprezentacyjny. Badania probingowe z lat 2019–2024 pokazują odrębne role głów: głowy pozycyjne, głowa skupiająca się na poprzednim tokenie, głowy kopiujące, głowy jednostek nazwanych, głowy indukcyjne (które leżą u podstaw uczenia się w kontekście).

**Linia odmian w 2026 roku:**

| Wariant | Głowy Q | Głowy K/V | Używane przez |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (np. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | skompresowane do niskiego rzędu | DeepSeek-V2, V3 |

GQA jest współczesnym domyślnym wyborem, ponieważ redukuje pamięć pamięci podręcznej KV o czynnik `N/G`, zachowując prawie pełną jakość. MLA idzie dalej, kompresując K/V do przestrzeni latentnej, a następnie rzutując z powrotem w czasie obliczeń — kosztuje FLOPy, oszczędza znacznie więcej pamięci.

```figure
multihead-split
```

## Zbuduj To

### Krok 1: podziel głowy z jednogłowicowej uwagi, którą już mamy

Weź `SelfAttention` z Lekcji 02 i opakuj ją parą split/concat. Zobacz `code/main.py` dla implementacji w numpy; logika jest następująca:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Jedno przekształcenie i jedna transpozycja. Żadnej pętli. To jest dokładnie to, co robi PyTorch w `nn.MultiheadAttention`.

### Krok 2: uruchom skalowaną uwagę iloczynową na głowę

Każda głowa dostaje swój własny wycinek Q, K, V. Uwaga staje się wsadowym mnożeniem macierzy:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Na prawdziwym sprzęcie `Qh @ Kh.transpose(...)` to jeden `bmm`. GPU widzi pojedyncze wsadowe mnożenie macierzy o kształcie `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Dodawanie głów jest darmowe.

### Krok 3: Wariant Grouped-Query Attention

Zmieniają się tylko projekcje klucza i wartości. Q dostaje `n_heads` grup; K i V dostają `n_kv_heads < n_heads` grup i są powtarzane, by dopasować:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Podczas wnioskowania oszczędza to pamięć, ponieważ tylko `n_kv_heads` kopii znajduje się w pamięci podręcznej KV, a nie `n_heads`. Llama 3 70B używa 64 głów zapytań z 8 głowami KV — 8× zmniejszenie pamięci podręcznej.

### Krok 4: zbadaj, czego nauczyła się każda głowa

Uruchom MHA na krótkim zdaniu z 4 głowami. Dla każdej głowy wyświetl macierz uwagi `(N, N)`. Zobaczysz, że różne głowy wyłapują różne struktury nawet przy losowej inicjalizacji — to częściowo sygnał, częściowo symetria obrotowa w podprzestrzeniach.

## Użyj Tego

W PyTorch, wersja jednowierszowa:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA od PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Ile głów?** Reguły kciuka z produkcyjnych modeli w 2026 roku:

| Rozmiar modelu | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Mały (~125M) | 768 | 12 | 64 |
| Bazowy (~350M) | 1024 | 16 | 64 |
| Duży (~1B) | 2048 | 16 | 128 |
| Pograniczny (~70B) | 8192 | 64 | 128 |

`d_head` prawie zawsze ląduje na 64 lub 128. To jednostka tego, jak wiele jedna głowa może "zobaczyć." Zejdź poniżej 32, a głowy zaczynają walczyć ze współczynnikiem skalowania `sqrt(d_head)`; idź powyżej 256, a tracisz korzyść "wielu małych specjalistów."

## Dostarcz To

Zobacz `outputs/skill-mha-configurator.md`. Umiejętność zaleca liczbę głów, liczbę głów KV i strategię projekcji dla nowego transformera, biorąc pod uwagę budżet parametrów, długość sekwencji i cel wdrożenia.

## Ćwiczenia

1. **Łatwe.** Weź MHA z `code/main.py` i zmień `n_heads` z 1 na 16 przy stałym `d_model=64`. Narysuj wykres straty małego modelu jednowarstwowego na syntetycznym zadaniu kopiowania. Czy więcej głów pomaga, osiąga plateau, czy szkodzi?
2. **Średnie.** Zaimplementuj MQA (jedna głowa KV współdzielona między wszystkimi głowami zapytań). Zmierz, jak bardzo spada liczba parametrów w porównaniu do pełnego MHA. Oblicz, jak bardzo rozmiar pamięci podręcznej KV zmniejsza się podczas wnioskowania dla N=2048.
3. **Trudne.** Zaimplementuj małą wersję Multi-head Latent Attention: skompresuj K,V do latentu rzędu `r`, przechowuj latent w pamięci podręcznej KV, dekompresuj w czasie uwagi. Przy jakim `r` pamięć podręczna spada poniżej 1/8 pełnego MHA, podczas gdy jakość pozostaje w granicach 1 bitu walidacyjnego ppl?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|-----------------|-----------------------|
| Głowa (Head) | "Pojedynczy obwód uwagi" | Jedna projekcja Q/K/V o wymiarze `d_head = d_model / n_heads` z własną macierzą uwagi. |
| d_head | "Wymiar głowy" | Szerokość ukryta na głowę; prawie zawsze 64 lub 128 w produkcji. |
| Podział / łączenie (Split / combine) | "Sztuczki z przekształcaniem" | `(N, d_model) ↔ (n_heads, N, d_head)` przekształć+transponuj wokół uwagi. |
| W_o | "Projekcja wyjściowa" | Macierz `(d_model, d_model)` zastosowana po połączeniu głów; miejsce, gdzie głowy się mieszają. |
| MQA | "Jedna głowa KV" | Multi-Query Attention: pojedyncza współdzielona projekcja K/V. Najmniejsza pamięć podręczna KV, pewna utrata jakości. |
| GQA | "Domyślna od Llamy 2" | Grouped-Query Attention z `n_kv_heads < n_heads`; powtarza, by dopasować Q. |
| MLA | "Sztuczka DeepSeek" | Multi-head Latent Attention: K,V skompresowane do latentu niskiego rzędu, dekompresowane w czasie uwagi. |
| Głowa indukcyjna (Induction head) | "Obwód stojący za uczeniem się w kontekście" | Para głów, które wykrywają poprzednie wystąpienia i kopiują to, co po nich nastąpiło. |

## Dalsza Lektura

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — oryginalna specyfikacja wielogłowicowa.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — artykuł MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — jak przekonwertować MHA na GQA po treningu.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA i dlaczego bije MHA/GQA na pamięci podręcznej.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistyczne spojrzenie na to, co głowy faktycznie robią.