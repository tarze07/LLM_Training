# Flow Matching i Rectified Flows

> Modele dyfuzyjne potrzebują 20-50 kroków próbkowania, ponieważ podążają zakrzywioną ścieżką od szumu do danych. Flow matching (Lipman et al., 2023) i rectified flow (Liu et al., 2022) trenują proste ścieżki. Prostsze ścieżki oznaczają mniej kroków i szybszą inferencję. Stable Diffusion 3, Flux.1 i AudioCraft 2 wszystkie przeszły na flow matching w 2024 roku.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## Problem

Proces odwrotny DDPM to 1000-krokowy stochastyczny spacer od `N(0, I)` z powrotem do rozkładu danych. DDIM zredukował go do 20-50 deterministycznych kroków. Chcesz jeszcze mniej kroków — najlepiej jeden. Blokadą jest to, że ODE rozwiązujące proces odwrotny jest sztywne; ścieżka jest zakrzywiona.

Gdybyś mógł wytrenować model tak, aby ścieżka od szumu do danych była *linią prostą*, pojedynczy krok Eulera od `t=1` do `t=0` by zadziałał. Flow matching buduje to bezpośrednio: definiuje liniową interpolację od `x_1 ∼ N(0, I)` do `x_0 ∼ data`, trenuje pole wektorowe `v_θ(x, t)`, aby dopasować jego pochodną po czasie, a następnie całkuje podczas inferencji.

Rectified flow (Liu 2022) idzie dalej: iteracyjnie prostuje ścieżki za pomocą procedury reflow, która tworzy coraz bardziej liniowe ODE. Po dwóch iteracjach reflow, 2-krokowy sampler dorównuje jakości 50-krokowego DDPM.

## Koncepcja

![Flow matching: liniowa interpolacja między szumem a danymi](../assets/flow-matching.svg)

### Przepływ liniowy

Definiujemy:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

gdzie `x_0 ~ data` i `x_1 ~ N(0, I)`. Pochodna po czasie wzdłuż tej linii prostej jest stała:

```
dx_t / dt = x_1 - x_0
```

Definiujemy neuronowe pole wektorowe `v_θ(x_t, t)` i trenujemy je, aby dopasować tę pochodną:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Jest to strata **warunkowego flow matchingu** (Lipman 2023). Trenowanie jest symulacjo-wolne: nigdy nie rozwijasz ODE. Po prostu próbkuj `(x_0, x_1, t)` i regresuj.

### Próbkowanie

Podczas inferencji całkuj nauczone pole wektorowe *wstecz* w czasie:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Zacznij od `x_1 ~ N(0, I)`, wykonuj kroki Eulera w dół do `t=0`.

### Rectified flow (Liu 2022)

Flow matching działa, ale nauczone ścieżki *nie są faktycznie proste* — zakrzywiają się, ponieważ wiele `x_0` może mapować do tego samego `x_1`. Krok reflow rectified flow:

1. Trenuj model przepływu v_1 z losowym parowaniem.
2. Próbkuj N par `(x_1, x_0)` przez całkowanie v_1 od `x_1` do jego docelowego `x_0`.
3. Trenuj v_2 na tych sparowanych przykładach. Ponieważ pary są teraz "dopasowane ODE", liniowa interpolacja między nimi jest rzeczywiście bardziej płaska.
4. Powtórz.

W praktyce 2 iteracje reflow dają prawie liniowe ścieżki, umożliwiając 2-4 krokową inferencję. SDXL-Turbo, SD3-Turbo, LCM to wszystkie modele destylowane z flow matchingu.

### Dlaczego to wygrało dla obrazów w 2024

Trzy powody:

1. **Trenowanie wolne od symulacji** — brak rozwijania ODE podczas trenowania, trywialne do implementacji.
2. **Lepsza geometria straty** — proste ścieżki mają spójny stosunek sygnału do szumu, podczas gdy strata ε DDPM ma zły SNR na krawędziach harmonogramu.
3. **Szybsza inferencja** — 4-8 kroków przy jakości SDXL-Turbo; 1 krok z consistency distillation.

## Flow matching vs DDPM — dokładne połączenie

Flow matching ze ścieżką warunkową Gaussa to dyfuzja *z określonym harmonogramem szumu*. Wybierz harmonogram `x_t = α(t) x_0 + σ(t) x_1`, a flow matching odtwarza dyfuzję w reformulacji Stratonovicha z `v = α'·x_0 - σ'·x_1`. Te dwa podejścia są algebraicznie równoważne dla ścieżek Gaussowskich.

To, co dodał flow matching: *przejrzystość* celu (zwykła prędkość), czystszą stratę i swobodę eksperymentowania z niegaussowskimi interpolantami.

## Zbuduj to

`code/main.py` implementuje 1-wymiarowy flow matching na dwumodalnej mieszaninie Gaussa. Pole wektorowe `v_θ(x, t)` to małe MLP trenowane z liniowym celem. Podczas inferencji całkuj 1, 2, 4 i 20 kroków Eulera i porównaj jakość próbek.

### Krok 1: strata treningowa

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Krok 2: wielokrokowa inferencja

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Krok 3: porównaj liczby kroków

Spodziewaj się, że 4-krokowy sampler już dorówna jakości 20-krokowej — to duża sprawa dla opóźnienia.

## Pułapki

- **Parametryzacja czasu.** Flow matching używa `t ∈ [0, 1]` z `t=0` dla danych, `t=1` dla szumu. DDPM używa `t ∈ [0, T]` z `t=0` dla danych, `t=T` dla szumu. Ten sam kierunek, inna skala. Artykuły naukowe ciągle się w tym mylą.
- **Wybór harmonogramu.** Linia prosta rectified flow to "ten" harmonogram flow matchingu, ale możesz użyć cosinusa lub logit-normal t-samplingu (SD3 to robi) dla lepszego pokrycia skali.
- **Koszt reflow.** Generowanie sparowanego zestawu danych dla reflow to pełny przebieg inferencji na próbkę. Rób reflow tylko wtedy, gdy naprawdę potrzebujesz 1-2 kroków inferencji.
- **Classifieer-free guidance wciąż działa.** Po prostu zamień ε na v w kombinacji liniowej: `v_cfg = (1+w) v_cond - w v_uncond`.

## Użyj tego

| Zastosowanie | Stos 2026 |
|-------------|-----------|
| Tekst-na-obraz, najlepsza jakość | Flow matching: SD3, Flux.1-dev |
| Tekst-na-obraz, 1-4 kroki | Destylowany flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Inferencja w czasie rzeczywistym | Consistency distillation z bazy flow-matched (LCM, PCM) |
| Generowanie dźwięku | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Generowanie wideo | Flow matching zmieszany z dyfuzją (Sora, Veo, Stable Video) |
| Nauka/fizyka (trajektorie cząstek, cząsteczki) | Flow matching + ekwiwariantne pole wektorowe |

Zawsze, gdy artykuł mówi "szybszy niż dyfuzja" w 2025-2026, to prawie zawsze flow matching + destylacja.

## Dostarcz to

Zapisz `outputs/skill-fm-tuner.md`. Skill przyjmuje specyfikację modelu w stylu dyfuzji i konwertuje ją na konfigurację treningową flow matchingu: wybór harmonogramu, rozkład próbkowania czasu (uniform / logit-normal), optymalizator, plan reflow, docelowa liczba kroków, protokół ewaluacji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i porównaj MSE 1-krokowy vs 20-krokowy względem prawdziwego rozkładu danych.
2. **Średnie.** Przełącz z uniform `t` na logit-normal (koncentruje próbkowanie w środkowych `t`). Czy jakość modelu się poprawia?
3. **Trudne.** Zaimplementuj jedną iterację reflow: wygeneruj sparowane `(x_0, x_1)` przez całkowanie pierwszego modelu, wytrenuj drugi model na parach i porównaj jakość 1-krokowego próbkowania.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Flow matching | "Dyfuzja po linii prostej" | Trenuj `v_θ(x, t)`, aby dopasować `x_1 - x_0` wzdłuż interpolantu. |
| Rectified flow | "Reflow" | Iteracyjna procedura prostująca nauczone przepływy. |
| Pole prędkości | "v_θ" | Wyjście modelu — kierunek, w którym przesunąć `x_t`. |
| Interpolant liniowy | "Ścieżka" | `x_t = (1-t)·x_0 + t·x_1`; trywialna pochodna celu. |
| Sampler Eulera | "Rozwiązywacz ODE 1-go rzędu" | Najprostszy integrator; działa dobrze, gdy ścieżki są proste. |
| Logit-normal t | "Próbkowanie SD3" | Koncentruje próbkowanie `t` w kierunku wartości środkowych, gdzie gradienty są najsilniejsze. |
| Consistency distillation | "1-krokowy sampler" | Trenuj ucznia, aby mapował dowolne `x_t` bezpośrednio na `x_0`. |
| CFG z prędkością | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; ten sam trik, nowa zmienna. |

## Uwaga produkcyjna: Flux.1-schnell to flow matching w najszybszej postaci

Produkcyjnym zwycięstwem flow matchingu jest Flux.1-schnell — flow-matched DiT destylowany do 1-4 kroków inferencji przy zachowaniu jakości Flux-dev. Notebook "Uruchom Flux na 8GB maszynie" Nielsa to referencyjny przepis wdrożeniowy: kodowanie T5 + CLIP, skwantowane odszumianie MMDiT (w 4 krokach dla schnell vs 50 dla dev), dekodowanie VAE. Rachunek kosztów:

| Wariant | Kroki | Opóźnienie przy 1024² na L4 | Całkowite FLOPy (względne) |
|---------|-------|-----------------------------|---------------------------|
| Flux.1-dev (surowy) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× szybciej) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Zasada produkcyjna: **flow-matched baza + destylacja = domyślny wybór 2026 dla szybkiego tekstu-na-obraz.** Każdy większy dostawca dostarcza tę kombinację: SD3-Turbo (SD3 + flow + destylacja), Flux-schnell (Flux-dev + prostowanie rectified-flow), CogView-4-Flash. Czyste bazy dyfuzyjne istnieją tylko dla legacy checkpointów.

## Dalsza lektura

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow w skali.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — ogólna struktura obejmująca FM + dyfuzję.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 1-krokowa destylacja dyfuzji / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — wariant turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching w produkcji.