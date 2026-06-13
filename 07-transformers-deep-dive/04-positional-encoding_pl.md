# Kodowanie Pozycyjne — Sinusoidalne, RoPE, ALiBi

> Uwaga jest niezmienna względem permutacji. "The cat sat on the mat" i "mat the on sat cat the" dają to samo wyjście bez sygnału pozycyjnego. Trzy algorytmy to naprawiają — każdy z innym założeniem co do tego, co oznacza "pozycja."

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Problem

Skalowana uwaga iloczynowa jest ślepa na kolejność. Macierz uwagi `softmax(Q K^T / √d) V` jest obliczana z parami podobieństw. Pomieszaj wiersze `X`, a wiersze wyjścia będą pomieszane w ten sam sposób. Nic wewnątrz uwagi nie przejmuje się pozycją.

To nie jest błąd w modelu worka słów. Dla języka, kodu, audio, wideo — wszystkiego, gdzie kolejność niesie znaczenie — jest to fatalne.

Rozwiązaniem jest wstrzyknięcie pozycji do osadzeń w jakiś sposób. Trzy ery odpowiedzi:

1. **Absolutne sinusoidalne** (Vaswani 2017). Dodaj `sin/cos` pozycji do osadzenia. Proste, bezuczeniowe, źle ekstrapoluje poza wytrenowane długości.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Obróć wektory Q i K o kąt proporcjonalny do pozycji. Koduje *względną* pozycję bezpośrednio w iloczynie skalarnym. Dominujące w 2026 roku.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Pomiń osadzenia całkowicie; dodaj liniową karę na głowę do wyników uwagi opartą na odległości. Doskonała ekstrapolacja długości.

Stan na 2026 rok, zasadniczo każdy pograniczny otwarty model używa RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Garstka modeli z długim kontekstem używa ALiBi lub jego nowoczesnych wariantów. Absolutne sinusoidalne jest historyczne.

## Koncepcja

![Sinusoidalne absolutne vs obroty RoPE vs odległościowe obciążenie ALiBi](../assets/positional-encoding.svg)

### Absolutne sinusoidalne

Wstępnie oblicz stałą macierz `PE` o kształcie `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Następnie `X' = X + PE[:N]` przed uwagą. Każdy wymiar jest sinusoidą o innej częstotliwości. Model uczy się odczytywać pozycję ze wzorca fazy. Zawodzi poza `max_len`: nic nie powiedziało modelowi, co dzieje się na pozycji 2048, gdy widział tylko pozycje 0–2047.

### RoPE

Obróć wektory Q i K (nie osadzenia). Dla pary wymiarów `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 domyślnie
```

Zastosuj ten sam obrót do kluczy z pozycją `pos_k`. Iloczyn skalarny `q'_m · k'_n` staje się funkcją samego `(m - n)`. To znaczy: **wynik uwagi zależy tylko od względnej odległości**, mimo że obrót był oparty na absolutnych pozycjach. Piękna sztuczka.

Rozszerzanie RoPE: `base` może być skalowane (NTK-aware, YaRN, LongRoPE), aby ekstrapolować do dłuższych kontekstów bez ponownego treningu. Llama 3 rozszerzyła w ten sposób kontekst z 8K do 128K.

### ALiBi

Pomiń sztuczkę z osadzeniem. Obciąż wyniki uwagi bezpośrednio:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Gdzie `m_h` to nachylenie specyficzne dla głowy (np. `1 / 2^(8·h/H)`). Bliższe tokeny są wzmacniane; dalekie tokeny są karane. Żadnego kosztu w czasie treningu. Artykuł pokazuje, że ekstrapolacja długości bije sinusoidalne i dorównuje RoPE na oryginalnej wytrenowanej długości.

### Co wybrać w 2026 roku

| Wariant | Ekstrapolacja | Koszt treningu | Używane przez |
|---------|---------------|---------------|---------|
| Absolutne sinusoidalne | słaba | darmowe | oryginalny transformer, wczesny BERT |
| Wyuczone absolutne | brak | minimalny | GPT-2, GPT-3 |
| RoPE | dobra ze skalowaniem | darmowe | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | doskonała | etap dostrajania | Qwen2-1M, Llama 3.1 128K |
| ALiBi | doskonała | darmowe | BLOOM, MPT, Baichuan |

RoPE wygrało, ponieważ wchodzi w uwagę bez zmiany architektury, koduje względną pozycję, a jego hiperparametr `base` daje czyste pokrętło do dostrajania długiego kontekstu.

```figure
rope-explorer
```

## Zbuduj To

### Krok 1: kodowanie sinusoidalne

Zobacz `code/main.py`. Obliczenie w 4 liniach:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Dodaj to do macierzy osadzenia przed pierwszą warstwą uwagi.

### Krok 2: RoPE zastosowane do Q, K

RoPE działa w miejscu na Q i K. Dla każdej pary wymiarów:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Kluczowe: zastosuj tę samą funkcję do Q na pozycji `m` i K na pozycji `n`. Ich iloczyn skalarny zyskuje czynnik `cos((m-n)·θ_i)` na każdej parze współrzędnych. Uwaga uczy się względnej pozycji za darmo.

### Krok 3: Nachylenia ALiBi i obciążenie

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Dodaj `bias[h]` do macierzy wyników uwagi `(seq_len, seq_len)` głowy `h`, a następnie softmax.

### Krok 4: zweryfikuj właściwość względnej odległości RoPE

Wybierz dwa losowe wektory `a, b`. Obróć o `(pos_a, pos_b)`. Następnie o `(pos_a + k, pos_b + k)`. Oba iloczyny skalarne muszą być zgodne w granicach błędu zmiennoprzecinkowego. Ta właściwość jest całym sensem RoPE — jest niezmienna względem absolutnego przesunięcia, liczy się tylko względna różnica.

## Użyj Tego

PyTorch 2.5+ dostarcza narzędzia RoPE w `torch.nn.functional`. Większość produkcyjnego kodu używa `flash_attn` lub `xformers`, gdzie RoPE jest stosowane wewnątrz jądra uwagi.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Sztuczki długiego kontekstu w 2026 roku:**

- **Interpolacja NTK-aware.** Przeskaluj `base` do `base * (scale_factor)^(d/(d-2))` podczas rozszerzania z 4K do 16K+.
- **YaRN.** Mądrzejsza interpolacja, która zachowuje entropię uwagi na długich kontekstach. Llama 3.1 128K go używa.
- **LongRoPE.** Metoda Microsoftu z 2024 roku, która używa wyszukiwania ewolucyjnego do doboru współczynników skalowania na wymiar. Phi-3-Long go używa.
- **Interpolacja pozycji + dostrajanie.** Po prostu zmniejsz pozycje o współczynnik rozszerzenia i dostrajaj przez 1–5B tokenów. Zaskakująco skuteczne.

## Dostarcz To

Zobacz `outputs/skill-positional-encoding-picker.md`. Umiejętność wybiera strategię kodowania dla nowego modelu, biorąc pod uwagę docelową długość kontekstu, potrzeby ekstrapolacji i budżet treningowy.

## Ćwiczenia

1. **Łatwe.** Narysuj sinusoidalną macierz `PE` jako mapę ciepła dla `max_len=512, d=128`. Potwierdź wzór "paski stają się szersze wraz ze wzrostem indeksu wymiaru."
2. **Średnie.** Zaimplementuj skalowanie RoPE NTK-aware. Wytrenuj mały model językowy na sekwencjach długości 256, a następnie przetestuj na długości 1024 ze skalowaniem i bez. Zmierz perplexity.
3. **Trudne.** Zaimplementuj ALiBi i RoPE w tym samym module uwagi. Wytrenuj 4-warstwowy transformer na zadaniu kopiowania z sekwencjami długości 512. Ekstrapoluj do 2048 w czasie testowania. Porównaj degradację.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|-----------------|-----------------------|
| Kodowanie pozycyjne (Positional encoding) | "Mówi uwadze o kolejności" | Dowolny sygnał dodany do osadzeń lub uwagi, który koduje pozycję. |
| Sinusoidalne | "To oryginalne" | `sin/cos` przy geometrycznych częstotliwościach dodane do osadzeń; nie ekstrapoluje. |
| RoPE | "Obroty" | Obróć Q, K o kąt zależny od pozycji; iloczyn skalarny koduje względną odległość. |
| ALiBi | "Sztuczka z liniowym obciążeniem" | Dodaj `-m·\|i-j\|` do wyników uwagi; brak potrzeby osadzenia, świetna ekstrapolacja. |
| base | "Pokrętło RoPE" | Skaler częstotliwości w RoPE; zwiększ, aby rozszerzyć kontekst podczas wnioskowania. |
| NTK-aware | "Sztuczka skalowania RoPE" | Przeskaluj `base`, aby wymiary wysokich częstotliwości nie były ściskane, gdy kontekst się rozszerza. |
| YaRN | "Wymyślne" | Interpolacja+ekstrapolacja na wymiar, która zachowuje entropię uwagi. |
| Ekstrapolacja (Extrapolation) | "Działa poza wytrenowaną długością" | Czy schemat pozycji może dostarczyć poprawne wyjście poza `max_len` widzianym w treningu? |

## Dalsza Lektura

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — oryginalne sinusoidalne.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — artykuł RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — najnowocześniejsze skalowanie RoPE.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — artykuł Meta o długim kontekście Llamy 2.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — metoda Microsoftu używana przez Phi-3-Long i cytowana w sekcji Użyj Tego.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — produkcyjne implementacje każdego schematu skalowania RoPE (default, linear, dynamic, YaRN, LongRoPE, Llama-3).