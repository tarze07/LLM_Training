# Dekodowanie Spekulacyjne — Szkicuj, Weryfikuj, Powtarzaj

> Dekodowanie autoregresyjne jest szeregowe. Każdy token czeka na poprzedni. Dekodowanie spekulacyjne zrywa łańcuch: tani model szkicuje N tokenów, drogi model weryfikuje wszystkie N w jednym przebiegu w przód. Gdy szkic jest poprawny, zapłaciłeś jeden duży przebieg w przód za N generacji.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## Problem

Model 70B LLM próbkujący jeden token zajmuje ~30 ms na H100. Model 3B do szkicowania zajmuje ~3 ms. Jeśli pozwolimy modelowi 3B naszkicować 5 tokenów do przodu, a następnie uruchomimy model 70B *raz*, by zweryfikować wszystkie 5, całkowity czas wynosi `5×3 + 30 = 45 ms` za do 5 zaakceptowanych tokenów — w porównaniu do `5×30 = 150 ms` dla generacji liniowej. To jest pełny pitch dekodowania spekulacyjnego: zamień niewielką ilość dodatkowej pamięci GPU (model szkicujący) na 2–4× niższe opóźnienie dekodowania.

Sztuczka musi zachować rozkład. Próbkowanie spekulacyjne, wprowadzone przez Leviathan i in. (2023) oraz niezależnie przez Chen i in., gwarantuje, że sekwencja wyjściowa ma **identyczny rozkład** jak ta, którą duży model wyprodukowałby samodzielnie. Żadnego kompromisu jakościowego. Po prostu szybciej.

Cztery rodziny par szkicujący-weryfikator dominują w inferencji 2026:

1. **Klasyczne spekulacyjne (Leviathan 2023).** Oddzielny model szkicujący (np. Llama 3 1B) + weryfikator (np. Llama 3 70B).
2. **Medusa (Cai 2024).** Wielokrotne głowy dekodujące na weryfikatorze przewidują pozycje `t+1..t+k` równolegle. Żadnego osobnego modelu szkicującego.
3. **Rodzina EAGLE (Li 2024, 2025).** Lekki szkic, który wykorzystuje ukryte stany weryfikatora; wyższy wskaźnik akceptacji niż klasyczne; typowo 3–4×.
4. **Dekodowanie z wyprzedzeniem (Fu 2024).** Iteracja Jacobiego; żaden model szkicujący nie jest potrzebny. Samospekulacja. Niszowe, ale bez zależności.

Każdy produkcyjny stos inferencji w 2026 domyślnie dostarcza dekodowanie spekulacyjne. vLLM, TensorRT-LLM, SGLang i llama.cpp wszystkie obsługują co najmniej klasyczne + EAGLE-2.

## Koncepcja

### Główny algorytm

Dany weryfikator `M_q` i tańszy szkic `M_p`:

1. Niech `x_1..x_k` będzie prefiksem już zdekodowanym.
2. **Szkicuj**: użyj `M_p` autoregresyjnie, by zaproponować `d_{k+1}, d_{k+2}, ..., d_{k+N}` z prawdopodobieństwami szkicu `p_1..p_N`.
3. **Weryfikuj równolegle**: uruchom `M_q` raz na `x_1..x_k, d_{k+1}, ..., d_{k+N}`, otrzymując prawdopodobieństwa weryfikatora `q_1..q_{N+1}` dla pozycji `k+1..k+N+1`.
4. **Akceptuj/odrzucaj każdy token szkicu od lewej do prawej**: dla każdego `i`, zaakceptuj z prawdopodobieństwem `min(1, q_i(d_i) / p_i(d_i))`.
5. Przy pierwszym odrzuceniu na pozycji `j`: próbkuj `t_j` z "resztkowego" rozkładu `(q_j - p_j)_+` znormalizowanego. Wszystkie szkice po `j` są odrzucane.
6. Po zaakceptowaniu wszystkich `N`: próbkuj jeden dodatkowy token `t_{N+1}` z `q_{N+1}` (darmowy bonusowy token).

Sztuczka z rozkładem resztkowym to matematyczny wgląd, który utrzymuje rozkład wyjścia dokładnie taki, jak gdyby `M_q` próbkował od zera.

### Co determinuje przyspieszenie

Niech `α` = oczekiwany wskaźnik akceptacji na token szkicu. Niech `c` = stosunek kosztu szkicu do weryfikatora. Na krok:

- Naiwna generacja wykonuje 1 wywołanie dużego modelu na token.
- Spekulacyjna wykonuje 1 wywołanie dużego modelu na `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokenów, gdy `α` jest wysokie.

Typowa zasada kciuka przy `α = 0.75` i `N = 5`: 3× mniej wywołań dużego modelu. Koszt szkicu to 5× tani. Całkowity czas ścienny spada ~2.5×.

**α zależy od:**

- Jak dobrze szkic przybliża weryfikator. Ta sama rodzina / te same dane treningowe znacząco podnoszą α.
- Strategii dekodowania. Zachłanny szkic przeciw zachłannemu weryfikatorowi: wysokie α. Próbkowanie z temperaturą: trudniej dopasować; akceptacja spada.
- Typu zadania. Kod i strukturalne wyjścia akceptują więcej (przewidywalne); swobodne pisanie kreatywne akceptuje mniej.

### Medusa — szkice bez modelu szkicującego

Medusa zastępuje model szkicujący dodatkowymi głowami wyjściowymi na weryfikatorze. Na pozycji `t`:

```
wspólny pień → ukryty h_t
    ├── head_0: przewiduje token w t+1  (standardowa głowa LM)
    ├── head_1: przewiduje token w t+2
    ├── head_2: przewiduje token w t+3
    ├── head_3: przewiduje token w t+4
```

Każda głowa wyprowadza własne logity. W inferencji próbkujesz z każdej głowy, by uzyskać sekwencję kandydacką, a następnie weryfikujesz jednym przebiegiem w przód, używając schematu drzewiastej atencji, który rozważa wszystkie kontynuacje kandydackie naraz.

Zalety: żaden drugi model. Wady: dodaje trenowalne parametry; wymaga etapu nadzorowanego fine-tuningu (~1B tokenów); wskaźnik akceptacji jest nieco niższy niż w klasycznym spekulacyjnym z dobrym szkicem.

### EAGLE — lepszy szkic przez wykorzystanie ukrytych stanów

EAGLE-1/2/3 (Li i in., 2024–2025) sprawia, że model szkicujący jest malutkim transformerem (zazwyczaj 1 warstwa), który przyjmuje stany ukryte ostatniej warstwy weryfikatora. Ponieważ szkic widzi reprezentację cech weryfikatora, jego przewidywania silnie korelują z rozkładem wyjścia weryfikatora. Wskaźniki akceptacji rosną z ~0.6 (klasyczne) do 0.85+.

EAGLE-3 (2025) dodał przeszukiwanie drzewa kontynuacji kandydackich. vLLM i SGLang dostarczają EAGLE-2/3 jako domyślną ścieżkę spekulacyjną dla Llama 3/4 i Qwen 3.

### Taniec z cache KV

Weryfikacja podaje `N` tokenów szkicu do weryfikatora w jednym przebiegu w przód. To rozszerza cache KV weryfikatora o `N` wpisów. Jeśli niektóre szkice są odrzucane, musisz cofnąć cache do zaakceptowanej długości prefiksu.

Implementacje produkcyjne (`--speculative-model` vLLM, LookaheadDecoder TensorRT-LLM) obsługują to za pomocą buforów KV roboczych. Zapisz najpierw, zatwierdź po akceptacji. Nie jest to koncepcyjnie trudne, ale jest kłopotliwe.

## Zbuduj To

Zobacz `code/main.py`. Implementujemy główny algorytm próbkowania spekulacyjnego (krok odrzucenia + rozkład resztkowy) z:

- "Dużym modelem", który jest deterministycznym softmaxem nad ręcznie zakodowanym rozkładem (abyśmy mogli analitycznie zweryfikować matematykę akceptacji).
- "Modelem szkicującym", który jest perturbacją dużego modelu.
- Pętlą akceptacji / odrzucenia, która daje ten sam rozkład brzegowy co bezpośrednie próbkowanie.

### Krok 1: krok odrzucenia

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` to jednostajna liczba losowa. `q_prob` to prawdopodobieństwo weryfikatora dla szkicowanego tokena. `p_prob` to prawdopodobieństwo modelu szkicującego. Twierdzenie Leviathana mówi, że ta decyzja Bernoulliego, po której następuje próbkowanie z rozkładu resztkowego przy odrzuceniu, zachowuje rozkład weryfikatora dokładnie.

### Krok 2: rozkład resztkowy

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Odejmij `p` od `q` po współrzędnych, przytnij ujemne wartości do zera, renormalizuj. Próbkuj z tego przy każdym odrzuceniu.

### Krok 3: jeden krok spekulacyjny

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Pięć zaakceptowanych → jeden bonusowy → sześć tokenów wyprodukowanych w jednym przebiegu weryfikatora.

### Krok 4: zmierz wskaźnik akceptacji

Uruchom 10,000 kroków spekulacyjnych przy różnych poziomach jakości szkicu. Narysuj wykres wskaźnika akceptacji vs. dywergencji KL między rozkładami szkicu i weryfikatora. Powinieneś zobaczyć czystą monotoniczną zależność.

### Krok 5: zweryfikuj równoważność rozkładu

Empirycznie: histogram tokenów wyprodukowanych przez pętlę spekulacyjną powinien pasować do histogramu wyprodukowanego przez bezpośrednie próbkowanie z weryfikatora. To jest twierdzenie Leviathana w praktyce. Test chi-kwadrat potwierdza w granicach błędu próbkowania.

## Użyj Tego

Produkcja:

```bash
# vLLM z EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM z klasycznym modelem szkicującym
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM ma najszybszą ścieżkę Medusa od połowy 2026. `faster-whisper` opakowuje dekodowanie spekulacyjne dla Whisper-large z małym szkicem.

**Wybór szkicu:**

| Strategia | Kiedy wybrać | Przyspieszenie |
|-----------|--------------|----------------|
| Klasyczny szkic (rodzina 1B/3B Llama) | Szybki prototyp, bez trenowania | 1.8–2.3× |
| Głowy Medusa | Możesz fine-tunować weryfikator | 2–3× |
| EAGLE-2 / 3 | Produkcja, maksymalna prędkość | 3–4× |
| Dekodowanie z wyprzedzeniem | Bez szkicu, bez trenowania, bez dodatkowych parametrów | 1.3–1.6× |

**KIEDY NIE stosować dekodowania spekulacyjnego:**

- Generowanie pojedynczej sekwencji 1–5 tokenów. Narzut dominuje.
- Bardzo kreatywne / wysokotemperaturowe próbkowanie (α spada).
- Wdrożenia z ograniczoną pamięcią (model szkicujący dodaje VRAM).

## Dostarcz To

Zobacz `outputs/skill-spec-decode-picker.md`. Umiejętność dobiera strategię dekodowania spekulacyjnego (klasyczne / Medusa / EAGLE / z wyprzedzeniem) i parametry dostrajania (N, temperatura szkicu) dla nowego obciążenia inferencyjnego.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź, że rozkład tokenów spekulacyjnych pasuje do rozkładu bezpośredniego próbkowania weryfikatora na 50,000 tokenach z p > 0,05 w teście chi-kwadrat.
2. **Średnie.** Narysuj wykres przyspieszenia (tokenów na przebieg w przód dużego modelu) jako funkcję `N` dla `α = 0.5, 0.7, 0.85`. Zidentyfikuj optymalne `N` dla każdego α. (Wskazówka: oczekiwane tokeny na wywołanie weryfikacji = `(1 - α^{N+1}) / (1 - α)`.)
3. **Trudne.** Zaimplementuj małą Medusę: weź GPT z projektu końcowego z Lekcji 14, dodaj 3 dodatkowe głowy LM przewidujące pozycje t+2, t+3, t+4. Trenuj na tinyshakespeare z łączną stratą wielogłową. Porównaj wskaźniki akceptacji z klasycznym szkicem utworzonym przez przycięcie tego samego modelu.
4. **Trudne.** Zaimplementuj wycofywanie: zacznij z 10-tokenowym cache KV prefiksu, podaj 5 tokenów szkicu, symuluj odrzucenie na pozycji 3. Sprawdź, czy twój cache odczytuje poprawnie "prefiks + pierwsze 2 zaakceptowane szkice" przy następnej iteracji.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|--------|-----------------|-----------------------|
| Model szkicujący | "Ten tani" | Mniejszy model, który proponuje tokeny kandydackie; zwykle 10–50× tańszy od weryfikatora. |
| Weryfikator | "Ten duży" | Docelowy model, którego rozkład zachowujemy; uruchamiany raz na krok spekulacyjny. |
| Wskaźnik akceptacji (α) | "Jak często szkic ma rację" | Prawdopodobieństwo na token, że weryfikator zaakceptuje szkic. Typowo 0.7–0.9. |
| Rozkład resztkowy | "Awaryjne próbkowanie przy odrzuceniu" | `(q - p)_+` znormalizowane; próbkowanie z tego przy odrzuceniu zachowuje rozkład weryfikatora. |
| Bonusowy token | "Darmowy" | Gdy wszystkie N szkiców zaakceptowane, próbkuj jeden więcej z rozkładu następnego kroku weryfikatora. |
| Medusa | "Spekulacja bez szkicu" | Wielokrotne głowy LM na weryfikatorze przewidują pozycje t+1..t+k równolegle. |
| EAGLE | "Szkic na stanach ukrytych" | Malutki szkic transformera warunkowany stanami ukrytymi ostatniej warstwy weryfikatora. |
| Dekodowanie z wyprzedzeniem | "Iteracja Jacobiego" | Samospekulacja z użyciem iteracji punktu stałego; żaden model szkicujący. |
| Atencja drzewiasta | "Weryfikuj wielu kandydatów naraz" | Rozgałęziona weryfikacja rozważająca kilka kontynuacji szkicu jednocześnie. |
| Wycofywanie KV | "Cofnij odrzucone szkice" | Roboczy bufor KV; zatwierdź przy akceptacji, odrzuć przy odrzuceniu. |

## Dalsza lektura

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — główny algorytm i twierdzenie o równoważności.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — równoczesne wprowadzenie; czysty dowód odrzucenia Bernoulliego.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — artykuł Medusa; weryfikacja z atencją drzewiastą.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; szkic warunkowany stanami ukrytymi.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; dynamiczna głębokość drzewa.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — dekodowanie z wyprzedzeniem, podejście bez szkicu.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — kanoniczna referencja produkcyjna ze wszystkimi czterema strategiami.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — kod referencyjny dla EAGLE-1/2/3.