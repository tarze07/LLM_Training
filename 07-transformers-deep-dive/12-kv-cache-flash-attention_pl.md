# Pamięć podręczna KV, Flash Attention i Optymalizacja Wnioskowania

> Trenowanie jest równoległe i ograniczone FLOP-ami. Wnioskowanie jest szeregowe i ograniczone pamięcią. Inne wąskie gardło, inne sztuczki.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## Problem

Naiwny autoregresyjny dekoder wykonuje `O(N²)` pracy, aby wygenerować `N` tokenów: na każdym kroku przelicza uwagę na całym prefiksie. Dla odpowiedzi o długości 4K tokenów to 16M operacji uwagi, z których większość jest zbędna. Każdy stan ukryty tokena prefiksu jest deterministyczny po obliczeniu — potrzebujesz tylko uruchomić zapytanie nowego tokena względem zapisanych w pamięci kluczy i wartości wszystkiego, co było wcześniej.

Na dodatek sama uwaga przenosi dużo danych. Standardowa uwaga materializuje macierz wyników N×N, wyjście softmax N×d, końcowe wyjście N×d — zbyt wiele odczytów i zapisów do HBM. Dla N≥2K, uwaga staje się ograniczona pamięcią, zanim stanie się ograniczona FLOP-ami. Klasyczne jądra uwagi wykorzystują nowoczesne GPU w 4–10× gorszym stopniu.

Dwie optymalizacje, obie od Dao i in., przesunęły wnioskowanie na granicy możliwości z "wolnego" na "szybkie":

1. **Pamięć podręczna KV.** Przechowuj wektory K i V każdego tokena prefiksu. Uwaga każdego nowego tokena to jedno zapytanie względem zapisanych kluczy. Wnioskowanie redukuje się z `O(N²)` do `O(N)` na krok generacji.
2. **Flash Attention.** Podziel obliczenia uwagi na kafelki, aby pełna macierz N×N nigdy nie trafiła do HBM. Cały softmax + mnożenie macierzy odbywa się w SRAM. 2–4× przyspieszenie ścienne na A100; 5–10× na H100 z FP8.

Do 2026 roku obie są uniwersalne. Każdy produkcyjny stos wnioskowania (vLLM, TensorRT-LLM, SGLang, llama.cpp) zakłada ich istnienie. Każdy model na granicy możliwości jest dostarczany z włączonym Flash Attention.

## Koncepcja

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### Matematyka pamięci podręcznej KV

Na warstwę dekodera, na token, na głowę:

```
bajty_na_token_na_warstwę = 2 * d_head * rozmiar_typu
                          ^
                          K i V
```

Dla modelu 7B z 32 warstwami, 32 głowami, d_head=128, fp16:

```
na token na warstwę = 2 * 128 * 2 = 512 bajtów
na token (32 warstwy) = 16 KB
na kontekst 32K = 512 MB
```

Dla Llama 3 70B (80 warstw, d_head=128, GQA z 8 głowami KV):

```
na token na warstwę = 2 * 8 * 128 * 2 = 4096 bajtów (4 KB)
na kontekst 32K = 10.4 GB
```

Te 10 GB to powód, dla którego Llama 3 70B przy kontekście 128K potrzebuje większości z 40 GB A100 tylko dla pamięci podręcznej KV przy rozmiarze partii 1.

**GQA to zwycięstwo dla pamięci podręcznej KV.** MHA z 64 głowami wymagałoby 32 GB. MLA kompresuje jeszcze bardziej.

Przeciągnij wymiary i obserwuj, jak zmienia się rozmiar pamięci. Zwiększ długość sekwencji lub rozmiar partii i zobacz, jak szybko przekracza pojedyncze GPU:

```figure
kv-cache-sizer
```

### Flash Attention — sztuczka z kafelkowaniem

Standardowa uwaga:

```
S = Q @ K^T          (odczyt HBM, N×N, zapis HBM)
P = softmax(S)       (odczyt HBM, zapis HBM)
O = P @ V            (odczyt HBM, zapis HBM)
```

Trzy podróże do HBM. Na H100 przepustowość HBM wynosi 3 TB/s; SRAM to 30 TB/s. Każda podróż do HBM to spowolnienie o czynnik 10 w porównaniu do trzymania wszystkiego na chipie.

Flash Attention:

```
dla każdego bloku Q (rozmiar kafelka ~128 × 128):
    załaduj Q_kaf do SRAM
    dla każdego bloku K, V:
        załaduj K_kaf, V_kaf do SRAM
        oblicz S_kaf = Q_kaf @ K_kaf^T     (SRAM)
        bieżąca agregacja softmax           (SRAM)
        akumuluj do O_kaf                    (SRAM)
    zapisz O_kaf do HBM
```

Jedna podróż do HBM na kafelek. Całkowity ślad pamięciowy spada z `O(N²)` do `O(N)`. Przejście wsteczne przelicza niektóre wartości z przejścia w przód zamiast je przechowywać — kolejna oszczędność pamięci.

**Sztuczka numeryczna.** Bieżący softmax utrzymuje `(max, suma)` w poprzek kafelków, więc końcowa normalizacja jest dokładna. To nie przybliżenie — Flash Attention oblicza wyjście identyczne bitowo ze standardową uwagą (poza nieasocjatywnością fp16).

**Ewolucja wersji:**

| Wersja | Rok | Kluczowa zmiana | Przyspieszenie na referencyjnym sprzęcie |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Kafelkowe jądro SRAM | 2× na A100 |
| Flash 2 | 2023 | Lepsza równoległość, porządek causal-first | 3× na A100 |
| Flash 3 | 2024 | Asynchroniczność Hopper, FP8 | 1.5–2× na H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | 5-stopniowy potok Blackwell, software exp2 | Wnioskowanie jako pierwsze (początkowo tylko forward) |

Flash 4 jest początkowo dostępny tylko dla przejścia w przód. Trenowanie wciąż używa Flash 3. Obsługa GQA i varlen dla Flash 4 jest w przygotowaniu (połowa 2026).

### Dekodowanie spekulacyjne — drugie zwycięstwo w opóźnieniu

Tani model proponuje N tokenów. Duży model weryfikuje wszystkie N równolegle. Jeśli weryfikacja akceptuje k tokenów, zapłaciłeś 1 przepust dużego modelu w przód za k generacji. Typowe k=3–5 w kodzie i prozie.

Domyślne ustawienia 2026:
- **EAGLE 2 / Medusa.** Zintegrowane głowy propozycji, które współdzielą stany ukryte weryfikatora. 2–3× przyspieszenie bez utraty jakości.
- **Dekodowanie spekulacyjne z modelem propozycji.** 2–4× przyspieszenie na sprzęcie konsumenckim.
- **Dekodowanie Lookahead.** Iteracja Jacobiego; nie wymaga modelu propozycji. Niszowe, ale darmowe.

### Batching ciągły

Klasyczne wnioskowanie wsadowe: czekaj, aż najwolniejsza sekwencja się zakończy, a następnie rozpocznij nową partię. Marnuje GPU, gdy krótkie odpowiedzi kończą się wcześnie.

Batching ciągły (pierwszy raz dostarczony w Orca, obecnie w vLLM, TensorRT-LLM, SGLang): zamieniaj nowe żądania do partii, gdy tylko stare się zakończą. 5–10× wzrost przepustowości dla typowych obciążeń czatu.

### PagedAttention — pamięć podręczna KV jako pamięć wirtualna

Flagowa funkcja vLLM. Pamięć podręczna KV jest alokowana w 16-tokenowych blokach; tablica stron mapuje logiczne pozycje na fizyczne bloki. Pozwala współdzielić KV między równoległymi próbkami (przeszukiwanie wiązkowe, równoległe próbkowanie), hot-swapować prefiksy do buforowania promptów i defragmentować pamięć. 4× poprawa przepustowości w porównaniu do naiwnej alokacji ciągłej.

```figure
flash-attention-memory
```

## Zbuduj To

Zobacz `code/main.py`. Implementujemy:

1. Naiwny `O(N²)` inkrementalny dekoder.
2. Dekoder z pamięcią podręczną KV `O(N)`.
3. Kafelkowy softmax symulujący algorytm bieżącego maksimum Flash Attention.

### Krok 1: pamięć podręczna KV

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Proste: stale powiększaj wektory K, V na token w listach na warstwę i na głowę.

### Krok 2: kafelkowy softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Wyjście identyczne bitowo z `softmax(qK) V` w jednym przejściu, ale w dowolnym momencie zbiór roboczy to blok `kafelek × d_head`, a nie pełny `N × d_head`.

### Krok 3: porównaj naiwne a buforowane dekodowanie na 100-tokenowej generacji

Policz operacje uwagi. Naiwne: `O(N²)` = 5050. Buforowane: `O(N)` = 100. Kod wypisuje obie wartości.

## Użyj Tego

```python
# HuggingFace transformers automatycznie włącza pamięć podręczną KV w decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # użyj FA3 jeśli Hopper
    torch_dtype="bfloat16",
)
# generate() używa pamięci podręcznej KV automatycznie
```

Produkcyjne vLLM:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Buforowanie prefiksów między żądaniami to duże zwycięstwo w 2026 roku — ten sam prompt systemowy, przykłady few-shot lub długi dokument kontekstowy ponownie używa KV między wywołaniami. Dla obciążeń agentów z powtarzającymi się promptami narzędzi, buforowanie prefiksów rutynowo daje 5× wzrost przepustowości.

## Dostarcz To

Zobacz `outputs/skill-inference-optimizer.md`. Umiejętność wybiera implementację uwagi, strategię pamięci podręcznej KV, kwantyzację i dekodowanie spekulacyjne dla nowego wdrożenia wnioskowania.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź, że naiwny i buforowany dekoder produkują to samo wyjście; zanotuj różnicę w liczbie operacji.
2. **Średnie.** Zaimplementuj buforowanie prefiksów: mając prompt P i kilka dokończeń, wykonaj jeden przepust w przód przez P, aby wypełnić pamięć podręczną KV, a następnie rozgałęź na każde dokończenie. Zmierz przyspieszenie względem ponownego kodowania P dla każdego.
3. **Trudne.** Zaimplementuj zabawkowy PagedAttention: pamięć podręczna KV w stałych 16-tokenowych blokach z listą wolnych. Gdy sekwencja się kończy, zwróć jej bloki do puli. Zasymuluj 1,000 dokończeń czatu o różnej długości. Porównaj fragmentację pamięci z alokacją ciągłą.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Pamięć podręczna KV (KV cache) | "Sztuczka, która przyspiesza dekodowanie" | Przechowywane K i V z każdego tokena prefiksu; nowe zapytania odnoszą się do nich zamiast przeliczać. |
| HBM | "Główna pamięć GPU" | High Bandwidth Memory; 80 GB na H100, 192 GB na B200. ~3 TB/s przepustowości. |
| SRAM | "Pamięć na chipie" | Szybka pamięć na SM, ~256 KB na SM na H100. ~30 TB/s przepustowości. |
| Flash Attention | "Kafelkowe jądro uwagi" | Oblicza uwagę bez materializowania N×N w HBM. |
| Batching ciągły (continuous batching) | "Batching bez czekania" | Zamień zakończone sekwencje na nowe bez opróżniania partii. |
| PagedAttention | "Flagowa funkcja vLLM" | Pamięć podręczna KV alokowana w stałych blokach z tablicą stron; eliminuje fragmentację. |
| Buforowanie prefiksów (prefix caching) | "Ponowne użycie długich promptów" | Buforuj KV dla współdzielonego prefiksu między żądaniami; znaczące obniżenie kosztów dla agentów. |
| Dekodowanie spekulacyjne (speculative decoding) | "Propozycja + weryfikacja" | Tani model proponuje tokeny; duży model weryfikuje k w jednym przejściu. |

## Dalsza Lektura

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — 5-stopniowy potok Blackwell i sztuczka software-exp2; przeczytaj README repozytorium po szczegóły o ograniczeniach początkowego uruchomienia tylko dla forward, o których wspomina ta lekcja.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — artykuł vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — dekodowanie spekulacyjne.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — artykuł EAGLE-1/2 o podejściu zintegrowanej propozycji, na które powołuje się lekcja.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — podejście Medusa, o którym mowa obok EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — kanoniczne szczegółowe omówienie projektu 16-tokenowego bloku i tablicy stron.