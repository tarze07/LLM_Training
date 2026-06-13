# Mechanizm Attention — Przełom

> Dekoder przestaje mrużyć oczy na skompresowane podsumowanie i zaczyna patrzeć na całe źródło. Wszystko po tym to attention plus inżynieria.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## Problem

Lekcja 09 zakończyła się zmierzoną porażką. GRU enkoder-dekoder wytrenowany na zabawkowym zadaniu kopiowania spada z 89% dokładności przy długości 5 do blisko przypadku przy długości 80. Przyczyna jest strukturalna, a nie błąd treningowy: każda informacja, którą enkoder wydobył, musi zmieścić się w jednym stanie ukrytym o stałym rozmiarze, a dekoder nigdy nie widzi niczego innego.

Bahdanau, Cho i Bengio opublikowali trzywierszową poprawkę w 2014. Zamiast dawać dekoderowi tylko końcowy stan enkodera, zachowaj każdy stan enkodera. W każdym kroku dekodera oblicz ważoną średnią stanów enkodera, gdzie wagi mówią "jak bardzo dekoder musi teraz patrzeć na pozycję enkodera `i`?" Ta ważona średnia to kontekst i zmienia się w każdym kroku dekodera.

To cały pomysł. Transformery go rozszerzyły. Self-attention zastosowało go do pojedynczej sekwencji. Multi-head attention uruchomiło go równolegle. Ale wersja z 2014 już przełamała wąskie gardło, a gdy już ją masz, przejście do transformerów to inżynieria, a nie koncepcja.

## Koncepcja

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

W każdym kroku dekodera `t`:

1. Użyj poprzedniego stanu ukrytego dekodera `s_{t-1}` jako **zapytania (query)**.
2. Oceń je względem każdego stanu ukrytego enkodera `h_1, ..., h_T`. Jeden skalar na pozycję enkodera.
3. Softmax z wyników daje wagi attention `α_{t,1}, ..., α_{t,T}`, które sumują się do 1.
4. Wektor kontekstu `c_t = Σ α_{t,i} * h_i`. Ważona średnia stanów enkodera.
5. Dekoder przyjmuje `c_t` plus poprzedni token wyjściowy i produkuje następny token.

Ważona średnia jest sednem. Gdy dekoder potrzebuje przetłumaczyć "Je" na "I", waży stan enkodera nad "Je" wysoko, a inne nisko. Gdy potrzebuje "not", waży "pas" wysoko. Wektor kontekstu zmienia kształt w każdym kroku.

## Kształty (rzecz, która gryzie każdego)

To miejsce, gdzie każda implementacja attention idzie źle za pierwszym razem. Czytaj powoli.

| Rzecz | Kształt | Uwagi |
|-------|-------|-------|
| Stany ukryte enkodera `H` | `(T_enc, d_h)` | Jeśli BiLSTM, `d_h = 2 * d_hidden` |
| Stan ukryty dekodera `s_{t-1}` | `(d_s,)` | Jeden wektor |
| Wynik attention `e_{t,i}` | skalar | Jeden na pozycję enkodera |
| Waga attention `α_{t,i}` | skalar | Po softmax po wszystkich `i` |
| Wektor kontekstu `c_t` | `(d_h,)` | Taki sam kształt jak stan enkodera |

**Bahdanau (addytywny) wynik.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` ma kształt `(d_s,)`, `h_i` ma kształt `(d_h,)`.
- `W_a` ma kształt `(d_attn, d_s)`. `U_a` ma kształt `(d_attn, d_h)`.
- Ich suma wewnątrz tanh ma kształt `(d_attn,)`.
- `v_α` ma kształt `(d_attn,)`. Iloczyn skalarny z `v_α` zwija się do skalara. **To właśnie robi `v_α`.** To nie magia. To projekcja, która zamienia wektor o wymiarze attention w skalarny wynik.

**Luong (multiplikatywny) wynik.** Trzy warianty:

- `dot`: `e_{t,i} = s_t^T * h_i`. Wymaga `d_s == d_h`. Trudne ograniczenie. Pomiń, jeśli twój enkoder jest dwukierunkowy.
- `general`: `e_{t,i} = s_t^T * W * h_i` z `W` o kształcie `(d_s, d_h)`. Usuwa ograniczenie równych wymiarów.
- `concat`: zasadniczo forma Bahdanau. Rzadko używane, ponieważ pierwsze dwa są tańsze.

**Jeden haczyk Bahdanau / Luong wart wymienienia.** Bahdanau używa `s_{t-1}` (stan dekodera *przed* wygenerowaniem bieżącego słowa). Luong używa `s_t` (stan *po*). Pomieszanie ich produkuje subtelnie błędne gradienty, które są niezwykle trudne do debugowania. Wybierz jeden artykuł i trzymaj się jego konwencji.

```figure
attention-heatmap
```

## Zbuduj to

### Krok 1: addytywne (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Sprawdź swoje kształty względem tabeli powyżej. `encoder_states` ma kształt `(T_enc, d_h)`. `projected_enc` ma kształt `(T_enc, d_attn)`. `projected_dec` ma kształt `(d_attn,)` i broadcastuje. `combined` ma kształt `(T_enc, d_attn)`. `scores` ma kształt `(T_enc,)`. `weights` ma kształt `(T_enc,)`. `context` ma kształt `(d_h,)`. Gotowe.

### Krok 2: Luong dot i general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Po trzy linie każda. Dlatego artykuł Luonga trafił w sedno. Ta sama dokładność w większości zadań, znacznie mniej kodu.

### Krok 3: przepracowany przykład numeryczny

Mając trzy stany enkodera (z grubsza "cat", "sat", "mat") i stan dekodera, który najbardziej pasuje do pierwszego, rozkład attention koncentruje się na pozycji 0. Jeśli stan dekodera przesunie się, by pasować do ostatniego, attention przesuwa się na pozycję 2. Wektor kontekstu podąża.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Pierwszy wiersz wygrywa. Następnie przesuń stan dekodera bliżej trzeciego stanu enkodera i obserwuj, jak wagi się zmieniają. To wszystko. Attention to jawne dopasowanie.

### Krok 4: dlaczego to jest most do transformerów

Przetłumacz powyższy język na Q/K/V:

- **Query (Zapytanie)** = stan dekodera `s_{t-1}`
- **Key (Klucz)** = stany enkodera (przeciw którym punktujemy)
- **Value (Wartość)** = stany enkodera (które ważymy i sumujemy)

W klasycznym attention klucze i wartości są tym samym. Self-attention je rozdziela: możesz odpytować sekwencję względem samej siebie, z różnymi uczonymi projekcjami dla K i V. Multi-head attention uruchamia to równolegle z różnymi uczonymi projekcjami. Transformery układają cały etap wielokrotnie i porzucają RNN.

Matematyka jest taka sama. Kształty są takie same. Pedagogiczny skok od attention Bahdanau do scaled dot-product attention to głównie kwestia notacji.

## Użyj tego

PyTorch i TensorFlow dostarczają attention bezpośrednio.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

To jest warstwa attention transformera. Partia zapytań o 5 pozycjach, partia kluczy/wartości o 10 pozycjach, 128-wymiarowe, 8 głowic. `output` to nowe zapytania wzbogacone o kontekst. `weights` to macierz dopasowania 5x10, którą możesz zwizualizować.

### Kiedy klasyczne attention wciąż ma znaczenie

- Pedagogika. Wersja z pojedynczą głowicą, pojedynczą warstwą, oparta na RNN czyni każdą koncepcję widoczną.
- Zadania sekwencyjne na urządzeniu, gdzie transformery się nie mieszczą.
- Każdy artykuł z lat 2014-2017. Źle go odczytasz, nie znając konwencji Bahdanau.
- Analiza drobnoziarnistego dopasowania w MT. Surowe wagi attention są narzędziem interpretowalności nawet na modelach transformerowych, a ich czytanie wymaga wiedzy, czym są.

### Pułapka wag-attention-jako-wyjaśnienie

Wagi attention wyglądają na interpretowalne. To wagi, które sumują się do jedności na pozycjach; możesz je wykreślić; wysoka wartość oznacza "spojrzało na to". Recenzenci je uwielbiają.

Nie są tak interpretowalne, jak wyglądają. Jain i Wallace (2019) pokazali, że rozkłady attention mogą być permutowane i zastępowane dowolnymi alternatywami bez zmiany predykcji modelu dla niektórych zadań. Nigdy nie zgłaszaj wag attention jako dowodu rozumowania bez testu ablacji lub kontrfaktycznego.

## Dostarcz to

Zapisz jako `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj maskowanie `softmax`, aby tokeny dopełniające w enkoderze dostały wagę attention zero. Przetestuj na partii z sekwencjami o zmiennej długości.
2. **Średnie.** Dodaj multi-head attention do formy Luong `general`. Podziel `d_h` na `n_heads` grup, uruchom attention na głowicę, połącz. Zweryfikuj, że przypadek z jedną głowicą odpowiada twojej wcześniejszej implementacji.
3. **Trudne.** Wytrenuj GRU enkoder-dekoder z attention Bahdanau na zabawkowym zadaniu kopiowania z lekcji 09. Narysuj wykres dokładności vs. długość sekwencji. Porównaj z modelem bazowym bez attention. Powinieneś zobaczyć, jak luka się powiększa wraz ze wzrostem długości, potwierdzając, że attention usuwa wąskie gardło.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Attention | Patrzenie na rzeczy | Ważona średnia sekwencji wartości, wagi obliczone z podobieństwa query-key. |
| Query, Key, Value | QKV | Trzy projekcje: Q pyta, K jest tym, do czego dopasować, V jest tym, co zwrócić. |
| Addytywne attention | Bahdanau | Wynik z feed-forward: `v^T tanh(W q + U k)`. |
| Multiplikatywne attention | Luong dot / general | Wynik to `q^T k` lub `q^T W k`. Tańsze, ta sama dokładność w większości zadań. |
| Macierz dopasowania | Ładny obrazek | Wagi attention jako siatka `(T_dec, T_enc)`. Przeczytaj ją, aby zobaczyć, na co model zwrócił uwagę. |

## Dalsze czytanie

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — artykuł.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — trzy warianty wyników i ich porównanie.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — zastrzeżenie dotyczące interpretowalności.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — wykonalny przewodnik z PyTorch.