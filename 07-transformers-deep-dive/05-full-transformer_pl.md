# Pełny Transformer — Enkoder + Dekoder

> Attention jest gwiazdą. Wszystko inne — residua, normalizacja, feed-forward, cross-attention — to rusztowanie, które pozwala układać go głęboko.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Problem

Pojedyncza warstwa attention to ekstraktor cech, a nie model. Jedno matmul na warstwę to za mała pojemność dla języka. Potrzebujesz głębi — a głębia psuje się bez odpowiedniej infrastruktury.

Artykuł Vaswaniego z 2017 roku zawierał sześć decyzji projektowych, które zamieniły jedną warstwę attention w układalny w stos blok. Każdy transformer od tamtej pory — tylko enkoder (BERT), tylko dekoder (GPT), enkoder-dekoder (T5) — dziedziczy ten sam szkielet. W 2026 roku bloki zostały udoskonalone (RMSNorm, SwiGLU, pre-norm, RoPE), ale szkielet jest identyczny.

Ta lekcja to szkielet. Kolejne lekcje go specjalizują — 06 dla enkoderów, 07 dla dekoderów, 08 dla enkoder-dekoder.

## Koncepcja

![Wnętrza bloków enkodera i dekodera, połączone](../assets/full-transformer.svg)

### Sześć elementów

1. **Osadzanie (embedding) + sygnał pozycyjny.** Tokeny → wektory. Pozycja dodawana przez RoPE (nowoczesne) lub sinusoidalnie (klasyczne).
2. **Self-attention.** Każda pozycja zwraca uwagę na każdą inną. Maskowane w dekoderach.
3. **Sieć feed-forward (FFN).** Pozycyjny dwuwarstwowy MLP: `W_2 · activation(W_1 · x)`. Domyślny współczynnik ekspansji 4×.
4. **Połączenie residualne.** `x + sublayer(x)`. Bez niego gradienty znikają po ~6 warstwach.
5. **Normalizacja warstwowa.** `LayerNorm` lub `RMSNorm` (nowoczesne). Stabilizuje strumień residualny.
6. **Cross-attention (tylko dekoder).** Zapytania (queries) pochodzą z dekodera, klucze i wartości (keys i values) z wyjścia enkodera.

Obserwuj przepływ wektora przez jeden blok: attention miesza informacje między pozycjami, połączenie residualne przenosi je dalej, FFN transformuje, a normalizacja utrzymuje stabilność strumienia.

```figure
transformer-block
```

### Blok enkodera (używany przez BERT, enkoder T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Enkoder jest dwukierunkowy. Bez maskowania. Wszystkie pozycje widzą wszystkie pozycje.

### Blok dekodera (używany przez GPT, dekoder T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Dekoder ma trzy podwarstwy na blok. Środkowa — cross-attention — to jedyne miejsce, w którym informacja przepływa z enkodera do dekodera. W architekturze czysto dekoderowej (GPT) cross-attention jest pomijany i mamy tylko maskowany self-attention + FFN.

### Pre-norm vs post-norm

Oryginalny artykuł: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm stracił popularność około 2019 roku — trudniej go głęboko trenować bez starannej rozgrzewki (warmupu). Pre-norm (`LN` *przed* podwarstwą) jest domyślnym wyborem w 2026: używają go Llama, Qwen, GPT-3+, Mistral.

### Nowoczesny blok w 2026 roku

Vaswani 2017 dostarczył LayerNorm + ReLU. Nowoczesne stosy zastąpiły oba. Jak faktycznie wyglądają bloki produkcyjne:

| Komponent | 2017 | 2026 |
|-----------|------|------|
| Normalizacja | LayerNorm | RMSNorm |
| Aktywacja FFN | ReLU | SwiGLU |
| Ekspansja FFN | 4× | 2.6× (SwiGLU używa trzech macierzy, całkowita liczba parametrów taka sama) |
| Pozycja | Sinusoidalna absolutna | RoPE |
| Attention | Pełne MHA | GQA (lub MLA) |
| Wyrazy wolne (bias) | Tak | Nie |

RMSNorm usuwa odejmowanie średniej z LayerNorm (jedno odejmowanie mniej), co oszczędza obliczenia i jest empirycznie co najmniej tak samo stabilne. SwiGLU (`Swish(W1 x) ⊙ W3 x`) konsekwentnie osiąga około 0.5 punktu ppl lepiej niż FFN z ReLU/GELU w artykułach Llamy, PaLM i Qwen.

### Liczba parametrów

Dla jednego bloku z `d_model = d` i ekspansją FFN `r`:

- MHA: `4 · d²` (projekcje Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Normalizacje: pomijalne

Przy `d = 4096, r = 2.6, layers = 32` (mniej więcej Llama 3 8B), razem: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parametrów na warstwę × 32 ≈ 7B` (plus osadzenia i głowa). Zgodne z publikowanymi liczbami.

## Zbuduj to

### Krok 1: elementy składowe

Używając małej klasy `Matrix` z Lekcji 03 (skopiowanej do tego pliku dla niezależności):

- `layer_norm(x, eps=1e-5)` — odejmij średnią, podziel przez odchylenie standardowe.
- `rms_norm(x, eps=1e-6)` — podziel przez RMS. Bez odejmowania średniej.
- `gelu(x)` i `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` i `decoder_block(x, enc_out, params)`.

Zobacz `code/main.py` po pełne połączenia.

### Krok 2: połącz dwuwarstwowy enkoder i dwuwarstwowy dekoder

Ułóż je w stos. Przekaż wyjście enkodera do każdego cross-attention dekodera. Dodaj końcową LN przed projekcją wyjściową.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Krok 3: wykonaj przejście w przód na przykładowych danych

Przepuść 6-tokenowe źródło i 5-tokenowy cel. Sprawdź, czy kształt wyjścia to `(5, vocab)`. Bez trenowania — ta lekcja dotyczy architektury, a nie funkcji straty.

### Krok 4: zamień na RMSNorm + SwiGLU

Zastąp LayerNorm i FFN z ReLU na RMSNorm i SwiGLU. Potwierdź, że kształty są zgodne. To modernizacja z 2026 roku poprzez zamianę jednej funkcji.

## Użyj tego

Implementacje referencyjne w PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Ale większość produkcyjnego kodu z 2026 roku pisze własny blok, ponieważ:

- Flash Attention jest wywoływane wewnątrz attention, a nie przez `nn.MultiheadAttention`.
- GQA / MLA nie ma w standardowej bibliotece.
- RoPE, RMSNorm, SwiGLU nie są domyślnymi ustawieniami PyTorch.

HF `transformers` ma czyste referencyjne bloki, które warto przeczytać: `modeling_llama.py` to kanoniczny blok dekoderowy z 2026 roku. Ma około 500 linii i warto go przejrzeć raz.

**Enkoder vs dekoder vs enkoder-dekoder — kiedy wybrać:**

| Potrzeba | Wybór | Przykład |
|----------|-------|----------|
| Klasyfikacja, osadzenia, QA na tekście | Tylko enkoder | BERT, DeBERTa, ModernBERT |
| Generowanie tekstu, czat, kod, rozumowanie | Tylko dekoder | GPT, Llama, Claude, Qwen |
| Strukturalne wejście → strukturalne wyjście (tłumaczenie, streszczanie) | Enkoder-dekoder | T5, BART, Whisper |

Architektura tylko z dekoderem wygrała w języku, ponieważ skaluje się najczyściej i obsługuje zarówno rozumienie, jak i generowanie. Enkoder-dekoder jest wciąż najlepszy, gdy wejście ma wyraźną tożsamość "sekwencji źródłowej" (tłumaczenie, rozpoznawanie mowy, zadania strukturalne).

## Dostarcz to

Zobacz `outputs/skill-transformer-block-reviewer.md`. Umiejętność (skill) sprawdza nową implementację bloku transformera względem domyślnych ustawień z 2026 roku i wskazuje brakujące elementy (pre-norm, RoPE, RMSNorm, GQA, współczynnik ekspansji FFN).

## Ćwiczenia

1. **Łatwe.** Policz parametry w swoim `encoder_block` przy `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Zweryfikuj, implementując blok i używając `sum(p.numel() for p in block.parameters())`.
2. **Średnie.** Przełącz z post-norm na pre-norm. Zainicjuj oba i zmierz normę aktywacji po 12 ułożonych warstwach na losowym wejściu. Aktywacje post-norm powinny eksplodować; pre-norm powinny pozostać ograniczone.
3. **Trudne.** Zaimplementuj 4-warstwowy enkoder-dekoder na zabawkowym zadaniu kopiowania (kopiuj `x` odwrócone). Trenuj przez 100 kroków. Raportuj straty. Zamień na RMSNorm + SwiGLU + RoPE — czy straty spadną?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Blok | "Jedna warstwa transformera" | Stos normalizacji + attention + normalizacji + FFN, owinięty połączeniami residualnymi. |
| Residuum | "Połączenie pomijające (skip connection)" | Wyjście `x + f(x)`; umożliwia przepływ gradientu przez głębokie stosy. |
| Pre-norm | "Normalizuj przed, nie po" | Nowoczesne: `x + sublayer(LN(x))`. Trenuje głębiej bez akrobacji z rozgrzewką. |
| RMSNorm | "LayerNorm bez średniej" | Dziel przez RMS; jedna operacja mniej, ta sama empiryczna stabilność. |
| SwiGLU | "FFN, na który wszyscy przeszli" | `Swish(W1 x) ⊙ W3 x → W2`. Pokonuje ReLU/GELU na ppl w modelach językowych. |
| Cross-attention | "Jak dekoder widzi enkoder" | MHA z Q z dekodera, K/V z wyjść enkodera. |
| Ekspansja FFN | "Jak szeroki jest środkowy MLP" | Stosunek rozmiaru ukrytego do d_model, zwykle 4 (LayerNorm) lub 2.6 (SwiGLU). |
| Bez obciążenia | "Pomiń składniki +b" | Nowoczesne stosy pomijają obciążenia w warstwach liniowych; niewielka poprawa ppl, mniejszy model. |

## Dalsze czytanie

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — oryginalna specyfikacja bloku.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — dlaczego pre-norm pokonuje post-norm na głębokości.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — artykuł o SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — kanoniczny blok dekoderowy z 2026 roku.