# Modele Dyfuzyjne — DDPM od Podstaw

> Ho, Jain, Abbeel (2020) dali polu przepis, od którego nie mogło ono odejść. Zniszcz dane szumem w tysiącu małych krokach. Wytrenuj jedną sieć neuronową do przewidywania szumu. Odwróć proces podczas wnioskowania. Dziś każdy główny model obrazu, wideo, 3D i muzyki działa na tej pętli, ewentualnie z flow matching lub consistency tricks na wierzchu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## Problem

Chcesz próbnik dla `p_data(x)`. GANy grają w grę minimax, która często rozbiega się. VAE produkują rozmyte próbki z gaussowskiego dekodera. To, czego naprawdę potrzebujesz, to funkcja celu, która jest (a) pojedynczą stabilną stratą (bez punktu siodłowego, bez minimax), (b) dolnym ograniczeniem `log p(x)` (więc masz wiarygodności) oraz (c) próbkami o jakości SOTA.

Sohl-Dickstein et al. (2015) mieli teoretyczną odpowiedź: zdefiniuj łańcuch Markowa `q(x_t | x_{t-1})`, który stopniowo dodaje szum gaussowski, i wytrenuj odwrotny łańcuch `p_θ(x_{t-1} | x_t)` do odszumiania. Ho, Jain, Abbeel (2020) pokazali, że stratę można uprościć do jednej linii — przewidywania szumu — i oczyścili matematykę. W 2020 było to ciekawostką. W 2021 dało próbki na poziomie SOTA. W 2022 stało się Stable Diffusion. W 2026 jest podstawą.

## Koncepcja

![DDPM: szum w przód, odszumianie w tył](../assets/ddpm.svg)

**Proces w przód `q`.** Dodawaj szum gaussowski w `T` małych krokach. Postać zamknięta — powód, dla którego matematyka jest łatwa do obliczenia — polega na tym, że skumulowany krok również jest gaussowski:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

gdzie `α̅_t = ∏_{s=1..t} (1 - β_s)` dla harmonogramu `β_t`. Wybierz `β_t` od 1e-4 do 0.02 liniowo przez T=1000 kroków, a `x_T` będzie w przybliżeniu `N(0, I)`.

**Proces wsteczny `p_θ`.** Naucz sieć neuronową `ε_θ(x_t, t)`, która przewiduje dodany szum. Mając `x_t`, odszum przez:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

gdzie `σ_t` to albo `sqrt(β_t)`, albo wyuczona wariancja. Wyrażenie jest brzydkie, ale to tylko algebra — rozwiązanie dla `x_{t-1}` na podstawie rozkładu a posteriori `q(x_{t-1} | x_t, x_0)` i zastąpienie `x_0` jego estymatą z przewidywanego szumu.

**Funkcja straty podczas treningu.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Próbkuj `x_0` z danych, wybierz losowe `t`, próbkuj `ε ~ N(0, I)`, oblicz zaszumione `x_t` w jednym kroku przez postać zamkniętą i regresuj względem szumu. Jedna strata, bez minimax, bez KL, bez trików z reparametryzacją.

**Próbkowanie.** Zacznij od `x_T ~ N(0, I)`. Iteruj odwrotny krok od `t = T` do `1`. Gotowe.

## Dlaczego to działa

Trzy intuicje:

1. **Odszumianie jest łatwe; generowanie jest trudne.** Przy `t=T` dane są czystym szumem — sieć musi rozwiązać trywialny problem. Przy `t=0` sieć musi tylko oczyścić kilka pikseli. Przy pośrednim `t` problem jest trudny, ale sieć ma wiele gradientów przepływających przez te same wagi z każdego poziomu szumu.

2. **Score matching w przebraniu.** Vincent (2011) udowodnił, że przewidywanie szumu jest równoważne estymacji `∇_x log q(x_t | x_0)`, czyli *score*. Odwrotne SDE używa tego score do wchodzenia pod górę gradientu gęstości — spacer z losowym ukierunkowaniem w stronę regionów o wysokim prawdopodobieństwie.

3. **ELBO sprowadza się do prostego MSE.** Pełna wariacyjna dolna granica zawiera składnik KL dla każdego kroku czasowego. Dzięki parametryzacji DDPM te składniki KL upraszczają się do MSE na przewidywaniu szumu z określonymi współczynnikami; Ho pominął współczynniki (nazywając to stratą "prostą"), a jakość *wzrosła*.

```figure
diffusion-denoise
```

## Zbuduj To

`code/main.py` implementuje 1-wymiarowy DDPM. Dane to mieszanina dwóch modów. "Sieć" to mały MLP, który przyjmuje `(x_t, t)` i zwraca przewidywany szum. Trening to jedno-liniowa strata. Próbkowanie iteruje odwrotny łańcuch.

### Krok 1: harmonogram w przód (postać zamknięta)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Krok 2: próbkuj `x_t` w jednym kroku

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Krok 3: jeden krok treningowy

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Krok 4: odwrotne próbkowanie

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Dla problemu 1-W z 40 krokami czasowymi i 24-jednostkowym MLP, uczy się mieszaniny dwóch modów w ~200 epokach.

## Warunek czasowy

Sieć musi wiedzieć, który krok czasowy odszumia. Dwa standardowe rozwiązania:

- **Osadzenie sinusoidalne.** Jak pozycyjne kodowanie w Transformerze. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Przepuść przez MLP, rozgłoś do sieci.
- **FiLM / warunek na normalizacji grupowej.** Rzutuj osadzenie na przesunięcie/skalę na kanał (FiLM) w każdym bloku.

Nasz zabawkowy kod używa sinusoidalnego → konkatenacji. Produkcyjne U-Nety używają FiLM.

## Pułapki

- **Harmonogram ma ogromne znaczenie.** Liniowe `β` to domyślne ustawienie DDPM, ale harmonogram cosinusowy (Nichol & Dhariwal, 2021) daje lepsze FID przy tych samych zasobach. Zmień harmonogram, jeśli jakość stoi w miejscu.
- **Osadzenie kroku czasowego jest delikatne.** Przekazywanie surowego `t` jako liczby zmiennoprzecinkowej działa dla zabawkowych 1-W, ale zawodzi dla obrazów; zawsze używaj właściwego osadzenia.
- **V-prediction vs ε-prediction.** W wąskich reżimach (bardzo małe lub bardzo duże t), `ε` ma słaby stosunek sygnału do szumu. V-prediction (`v = α·ε - σ·x`) jest stabilniejsze; używają go SDXL, SD3 i Flux.
- **Klasyfikator-bezprzewodowe sterowanie (CFG).** Podczas wnioskowania oblicz zarówno warunkowe, jak i bezwarunkowe `ε`, a następnie `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` z `w ≈ 3-7`. Omówione w Lekcji 08.
- **1000 kroków to dużo.** Produkcja używa DDIM (20-50 kroków), DPM-Solver (10-20 kroków) lub destylacji (1-4 kroki). Zobacz Lekcję 12.

## Użyj Tego

| Rola | Typowy stack w 2026 |
|------|---------------------|
| Obrazowa dyfuzja w przestrzeni pikseli (mała, zabawkowa) | DDPM + U-Net |
| Latentna dyfuzja obrazów | VAE enkoder + U-Net lub DiT (Lekcja 07) |
| Latentna dyfuzja wideo | Spatiotemporalny DiT (Sora, Veo, WAN) |
| Latentna dyfuzja audio | Encodec + dyfuzyjny transformer |
| Nauka (cząsteczki, białka, fizyka) | Równorzędna dyfuzja (EDM, RFdiffusion, AlphaFold3) |

Dyfuzja jest uniwersalnym generatywnym kręgosłupem. Flow matching (Lekcja 13) to konkurent z lat 2024-2026, który zwykle wygrywa na szybkości wnioskowania przy tej samej jakości.

## Wyślij To

Zapisz `outputs/skill-diffusion-trainer.md`. Umiejętność przyjmuje zbiór danych + budżet obliczeniowy i zwraca: harmonogram (liniowy/cosinusowy/sigmoidalny), cel predykcji (ε/v/x), liczbę kroków, skalę sterowania, rodzinę próbników oraz protokół ewaluacji.

## Ćwiczenia

1. **Łatwe.** Zmień T z 40 na 10 w `code/main.py`. Jak pogarsza się jakość próbek (wizualny histogram wyników)? Przy jakim T załamuje się struktura dwóch modów?
2. **Średnie.** Przełącz z ε-prediction na v-prediction. Wyprowadź ponownie odwrotny krok. Porównaj końcową jakość próbek.
3. **Trudne.** Dodaj klasyfikator-bezprzewodowe sterowanie. Warunkuj na etykiecie klasy `c ∈ {0, 1}`, pomiń ją w 10% przypadków podczas treningu, a w czasie próbkowania użyj `ε = (1+w)·ε_cond - w·ε_uncond`. Zmierz trafność warunkowego modu przy `w = 0, 1, 3, 7`.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| Proces w przód | "Dodawanie szumu" | Stały łańcuch Markowa `q(x_t \| x_{t-1})`, który niszczy dane. |
| Proces wsteczny | "Odszumianie" | Uczony łańcuch `p_θ(x_{t-1} \| x_t)`, który rekonstruuje dane. |
| Harmonogram β | "Drabina szumu" | Wariancja na krok; liniowa, cosinusowa lub sigmoidalna. |
| α̅ | "Alfa bar" | Iloczyn skumulowany `∏(1 - β)`; daje postać zamkniętą `x_t` z `x_0`. |
| Prosta strata | "MSE na szumie" | `\|\|ε - ε_θ(x_t, t)\|\|²`; wszystkie wyprowadzenia wariacyjne sprowadzają się do tego. |
| ε-prediction | "Przewidywanie szumu" | Wynik to dodany szum; standardowy DDPM. |
| V-prediction | "Przewidywanie prędkości" | Wynik to `α·ε - σ·x`; lepsze uwarunkowanie w poprzek t. |
| DDPM | "Ta praca" | Ho et al. 2020; liniowe β, 1000 kroków, U-Net. |
| DDIM | "Deterministyczny próbnik" | Próbnik nie-markowowski, 20-50 kroków, ten sam cel treningowy. |
| Klasyfikator-bezprzewodowe sterowanie | "CFG" | Mieszanie warunkowych i bezwarunkowych predykcji szumu w celu wzmocnienia warunku. |

## Uwaga produkcyjna: wnioskowanie dyfuzyjne to problem liczby kroków

Praca DDPM wykonuje T=1000 kroków wstecznych. Nikt nie dostarcza tego w produkcji. Każdy prawdziwy stos wnioskowania wybiera jedną z trzech strategii — a każda z nich dobrze wpisuje się w produkcyjne ramy "skąd pochodzi opóźnienie":

1. **Szybszy próbnik, ten sam model.** DDIM (20-50 kroków), DPM-Solver++ (10-20), UniPC (8-16). Zamienna wstawka pętli odwrotnej; wytrenowane wagi `ε_θ` pozostają nietknięte. Zmniejsza opóźnienie 20-50×.
2. **Destylacja.** Trenuj ucznia, aby dopasował nauczyciela w mniejszej liczbie kroków: Progressive Distillation (2 → 1), Consistency Models (dowolne → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Zmniejsza opóźnienie kolejne 5-10×, wymaga ponownego treningu.
3. **Buforowanie i kompilacja.** `torch.compile(unet, mode="reduce-overhead")`, backendy dyfuzyjne TensorRT-LLM, `xformers`/SDPA attention, wagi bf16. Zmniejsza opóźnienie na krok ~2×. Łączy się z (1) i (2).

Dla produkcyjnego serwera dyfuzyjnego rozmowa o budżecie jest taka sama, jak opisuje literatura produkcyjna dla LLM: opóźnienie to `num_steps × step_cost + VAE_decode`, przepustowość to `batch_size × (num_steps × step_cost)^-1`. TTFT jest małe (jeden krok); odpowiednik TPOT to pełny czas odpowiedzi, ponieważ generowanie obrazu jest "wszystko-na-raz" z perspektywy użytkownika.

## Dalsza Literatura

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — praca o dyfuzji, wyprzedzająca swoje czasy.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, mniej kroków.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — harmonogram cosinusowy, wyuczona wariancja.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — sterowanie klasyfikatorem.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — ujednolicona notacja, najczystszy przepis.