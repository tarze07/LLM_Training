# Punkt kontrolny gradientu i ponowne obliczanie aktywacji

> Wsteczna propagacja przechowuje każdą pośrednią aktywację. Przy 70B parametrach i kontekście 128K to 3 TB aktywacji na rangę. Punkt kontrolny (checkpointing) wymienia FLOPsy na pamięć: przelicz ponownie zamiast zapisywać. Pytanie brzmi, które segmenty pominąć, a odpowiedź nie brzmi „wszystkie".

**Type:** Build
**Languages:** Python (z numpy, opcjonalnie torch)
**Prerequisites:** Phase 10 Lesson 04 (pretrenowanie Mini-GPT), Phase 10 Lesson 05 (skalowanie i rozproszenie)
**Time:** ~70 minut

## Problem

Trenowanie transformera przechowuje, dla każdej warstwy, wejścia do każdej operacji, która jest różniczkowana wstecz: wejścia attention, projekcje Q/K/V, wyjście softmax, wejścia FFN, wyjścia normalizacji i strumień resztkowy. Dla warstwy o rozmiarze ukrytym `d`, długości sekwencji `L`, partii `B`, jest to rzędu `12 * B * L * d` liczb zmiennoprzecinkowych na warstwę.

Dla `d=8192, L=8192, B=1` to 800 MB/warstwę w BF16. Model 64-warstwowy to 51 GB aktywacji — i to przed pomnożeniem przez rozmiar mikro-partii, przed dodaniem pośrednich wartości softmax attention (`L^2` na głowicę) i przed uwzględnieniem częściowych kopii równoległości tensorowej.

Dwustronny rachunek: wagi BF16 plus stan optymalizatora mogą zmieścić się w 80GB, ale aktywacje wypychają cię poza limit. Punkt kontrolny gradientu (inaczej ponowne obliczanie aktywacji) to standardowe rozwiązanie. Odrzuć większość aktywacji; powtórz forward podczas backward, aby je odzyskać. Koszt: dodatkowe FLOPsy. Korzyść: pamięć spada o stosunek segmentów punktów kontrolnych do całkowitej liczby warstw.

Wykonane naiwnie, punktowanie kosztuje około 33% więcej FLOPsów forward na krok. Wykonane dobrze — selektywne punktowanie zgodnie z „inteligentnym wyborem" Korthikanti i in. — oszczędzasz 5x pamięć przy narzucie FLOPsów poniżej 5%. A przy mnożeniach macierzy FP8, odciążaniu FSDP i MoE z równoległością ekspertów ma to ogromne znaczenie: nie możesz sobie pozwolić ani na pamięć, ani na zmarnowane obliczenia.

## Koncepcja

### Czego faktycznie potrzebuje backward

`output = layer(input)`. Backward chce `grad_input` i `grad_params`. Aby je obliczyć, potrzebuje:

- `input` (aby obliczyć `grad_params = input.T @ grad_output` dla warstw liniowych)
- niektórych pośrednich pochodnych aktywacji (pochodna ReLU/GELU/softmax zależy od wartości aktywacji)

Forward przechowuje je automatycznie w grafie autograd. Każde `tensor.retain_grad()` i każda operacja, która potrzebuje swojego wejścia, przechowuje referencję.

### Naiwne pełne punktowanie

Podziel sieć na `N` segmentów. Podczas forward przechowuj tylko *wejście* do każdego segmentu. Gdy backward potrzebuje wartości pośrednich, uruchom ponownie forward segmentu, aby je zmaterializować, a następnie różniczkuj.

Przykład: 32-warstwowy transformer podzielony na 32 segmenty po 1 warstwie każdy.

- Pamięć: 32 wejścia warstw (małe) vs 32 * (objętość aktywacji na warstwę) (ogromne).
- Dodatkowe obliczenia: 1 dodatkowy forward na segment, tj. ~33% więcej FLOPsów forward łącznie (ponieważ backward to 2x forward, pełny krok to 1 + 1 + 2 = 4 jednostki zamiast 1 + 2 = 3).

To oryginalna recepta Chen i in. 2016: jeden punkt kontrolny co `sqrt(L)` warstw, aby zrównoważyć pamięć i obliczenia. Dla L=64 to 8 punktów kontrolnych.

### Selektywne punktowanie (Korthikanti 2022)

Nie wszystkie aktywacje kosztują tyle samo. Wyjście softmax attention to `B*L*L*heads` i rośnie *kwadratowo* z długością sekwencji. Ukryta aktywacja FFN to `B*L*4d` i rośnie liniowo. Dla długich sekwencji softmax dominuje.

Selektywne punktowanie zachowuje tanie do przechowywania aktywacje (projekcje liniowe, residua) i przelicza tylko drogie (attention). Płacisz minimalne FLOPsy za przeliczenie, ale oszczędzasz pamięć O(L^2).

Megatron-Core implementuje to jako „selektywne" ponowne obliczanie aktywacji. Używane w większości granicznych trenowań 2024+.

### Odciążanie

Alternatywa dla przeliczania: wyślij aktywacje do pamięci RAM CPU między forward a backward. Wymaga przepustowości PCIe; opłacalne, gdy bezczynna przepustowość przewyższa koszt ponownej materializacji. Mieszane strategie są powszechne: punktuj niektóre warstwy, odciążaj inne.

FSDP2 dostarcza odciążanie jako opcję pierwszej klasy. Odciążanie sprawdza się, gdy GPU jest ograniczone pamięcią, ale transfer CPU-GPU ma zapas.

### Model kosztów przeliczania

FLOPsy na krok z naiwnym punktowaniem co `k` warstw z `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # jeden dodatkowy forward na warstwę w segmencie
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Przy selektywnym punktowaniu przeliczasz tylko jądro attention, a nie całą warstwę:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Model oszczędności pamięci

Objętość aktywacji na warstwę: `A`. Dla `L` warstw, całkowita pamięć aktywacji: `L * A`.

Pełne punktowanie (rozmiar segmentu 1): przechowuj tylko `L * input_volume` (~`L * 1/10 A` dla standardowego transformera). Oszczędza ~`9 * L * A * 1/10`.

Punktowanie co `k` warstw: przechowuj `L/k * A` plus `k-1` warstw w aktywnym segmencie.

Przy `k = sqrt(L)`, pamięć i koszt przeliczania skalują się z `sqrt(L)` — optymalny kompromis dla warstw o jednolitym koszcie.

### Kiedy nie punktować

- Najbardziej wewnętrzne warstwy etapu potoku już w locie. Muszą i tak skończyć.
- Pierwsze i ostatnie warstwy, jeśli dominują obliczenia etapu (rzadkie w transformerach).
- Jądra attention już używające FlashAttention — Flash już szybko przelicza softmax, więc dodatkowe punktowanie na poziomie warstwy dodaje niewiele.

### Wzorce implementacyjne

1. **Otoka funkcji:** owiń segment w `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch przechowuje tylko `input`, przelicza wszystko inne podczas backward.

2. **Oparta na dekoratorach:** oznacz warstwy jako możliwe do punktowania; trener decyduje w czasie konfiguracji, które segmenty zostaną owinięte.

3. **Ręczne jawne przeliczanie:** napisz backward samodzielnie, wywołując niestandardowy `recompute_forward`, który duplikuje forward z zapisanym wejściem.

Wszystkie trzy dają ten sam funkcjonalny wynik. Otoki są standardowym idiomem.

### Interakcja z TP / PP / FP8

- **Równoległość tensorowa:** wejścia punktów kontrolnych muszą być zebrane lub rozrzucone przy przeliczaniu; uwzględnij koszt komunikacji.
- **Równoległość potokowa:** typowy wzorzec to punktowanie forward każdego etapu potoku, aby mikro-partie w odwrotnej kolejności mogły ponownie użyć pamięci aktywacji.
- **Przeliczanie FP8:** historie amax aktualizowane podczas przeliczania muszą pasować do oryginalnego forward, w przeciwnym razie skala FP8 dryfuje. Większość frameworków robi migawkę skali.

## Zbuduj

### Krok 1: Zabawkowy model z segmentami

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Krok 2: Naiwny backward potrzebujący wszystkich aktywacji

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Krok 3: Pamięć z punktowaniem co k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Krok 4: Model kosztów

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Krok 5: Estymator pamięci

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Krok 6: Optymalny rozmiar segmentu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Krok 7: Decyzja o selektywnym punktowaniu

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Użyj

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — kanoniczna otoka w PyTorch. Owija funkcję; przechowuje tylko wejścia, przelicza podczas backward.
- **Megatron-Core activation recomputation**: obsługuje tryby `selective`, `full` i `block`. Standard w granicznych trenowaniach 2024+.
- **FSDP2 offload**: `module.to_empty(device="cpu")` z `offload_policy` w FSDP2 dzieli aktywacje na CPU zamiast przeliczać.
- **DeepSpeed ZeRO-Offload**: odciążanie CPU dla stanów optymalizatora i aktywacji, uzupełniające punktowanie.

## Dostarcz

Ta lekcja produkuje `outputs/prompt-activation-recompute-policy.md` — podpowiedź, która przyjmuje konfigurację modelu (warstwy, hidden, seq, batch) i dostępną pamięć GPU i emituje politykę przeliczania na warstwę (brak / selektywne / pełne / odciążanie).

## Ćwiczenia

1. Zweryfikuj poprawność. Uruchom `model_forward` + `model_backward` (pełne aktywacje) vs `model_forward_checkpointed` + `model_backward_checkpointed` (segmenty). Gradienty parametrów muszą być identyczne z dokładnością maszynową.

2. Przeskanuj rozmiar segmentu `k` od 1 do `L`. Wykreśl narzut FLOPsów i pamięć. Znajdź kolano krzywej.

3. Zaimplementuj selektywne punktowanie: przechowuj wejście modułu attention, ale nie jego wartości pośrednie. Zmierz narzut FLOPsów w porównaniu do punktowania pełnych warstw dla modelu 32-warstwowego przy seq=8192.

4. Dodaj odciążanie. Zapisz wejścia segmentów do symulowanego „bufora CPU" (osobna lista). Zmierz „przepustowość PCIe" jako bajty/czas i znajdź punkt równowagi między odciążaniem a przeliczaniem.

5. Porównaj prawdziwy transformer PyTorch z i bez `torch.utils.checkpoint`. Zmierz pamięć (przez `torch.cuda.max_memory_allocated`) i czas kroku.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|----------------------|
| Punkt kontrolny gradientu | „Oszczędź pamięć, powtarzając forward" | Przechowuj tylko wejścia segmentów; przelicz wartości pośrednie podczas backward, aby uzyskać tensory wspierające gradient |
| Ponowne obliczanie aktywacji | „To samo co punktowanie" | Nazwa w stylu HPC dla tej samej techniki |
| Rozmiar segmentu (k) | „Ile warstw na punkt kontrolny" | Liczba warstw, których wartości pośrednie są odrzucane i ponownie materializowane razem |
| Selektywne punktowanie | „Sztuczka Korthikanti" | Przeliczaj tylko drogie do przechowywania aktywacje (softmax attention); zachowaj tanie |
| Pełne punktowanie | „Naiwna wersja" | Przeliczaj wartości pośrednie każdej warstwy w każdym segmencie |
| Punktowanie blokowe | „Gruboziarniste" | Punktuj całe bloki transformera; największa granularność |
| Narzut FLOPsów | „Podatek obliczeniowy" | Dodatkowe FLOPsy na krok = (FLOPsy przeliczenia) / (FLOPsy fwd + bwd); 33% naiwne, 5% selektywne |
| Odciążanie aktywacji | „Wyślij do CPU" | Przenieś aktywacje do pamięci RAM CPU między forward a backward; alternatywa dla przeliczania |
| Reguła sqrt-L | „Klasyczne optimum" | Dla warstw o jednolitym koszcie, optymalny odstęp punktów kontrolnych to sqrt(L) warstw |
| Objętość softmax attention | „Problem O(L^2)" | L^2 * heads * batch liczb zmiennoprzecinkowych; dominuje pamięć aktywacji przy długich kontekstach |

## Dalsze czytanie

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- oryginalny artykuł, który sformalizował punkt kontrolny gradientu
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selektywne ponowne obliczanie aktywacji i formalna analiza kosztów
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- alternatywne podejście ze stałą pamięcią poprzez ponowną materializację w trybie odwrotnym
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- odciążanie aktywacji na dużą skalę
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- standardowe API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- tryby selektywny, pełny i blokowy