# Proksymalna Optymalizacja Polityki (PPO)

> A2C wyrzuca każde wykonanie po jednej aktualizacji. PPO owija gradient polityki w przycięty współczynnik ważności, abyś mógł zrobić 10+ epok na tych samych danych bez eksplozji polityki. Schulman i in. (2017). Wciąż domyślny algorytm gradientu polityki w 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## Problem

A2C (Lekcja 07) działa w ramach polityki: gradient `E_{π_θ}[A · ∇ log π_θ]` wymaga danych próbkowanych z *bieżącej* `π_θ`. Wykonaj jedną aktualizację, a `π_θ` się zmieni; dane, których użyłeś, są teraz poza polityką. Użyj ich ponownie, a twój gradient będzie obciążony.

Wykonania są kosztowne. Na Atari, jedno wykonanie na 8 środowisk × 128 kroków = 1024 tranzycje i kilkanaście sekund czasu środowiska. Wyrzucanie ich po jednym kroku gradientowym jest marnotrawstwem.

Optymalizacja polityki z regionem zaufania (TRPO, Schulman 2015) była pierwszą poprawką: ogranicz każdą aktualizację, aby dywergencja KL między starą a nową polityką pozostawała poniżej `δ`. Czyste teoretycznie, ale wymaga rozwiązania gradientów sprzężonych na aktualizację. Nikt nie uruchamia TRPO w 2026.

PPO (Schulman i in. 2017) zastępuje twarde ograniczenie regionu zaufania prostym przyciętym celem. Jedna dodatkowa linia kodu. Dziesięć epok na wykonanie. Żadnych gradientów sprzężonych. Wystarczająco dobre gwarancje teoretyczne. Dziewięć lat później wciąż jest domyślnym algorytmem gradientu polityki dla wszystkiego od MuJoCo po RLHF.

## Koncepcja

![Przycięty cel zastępczy PPO: przycięcie współczynnika przy 1 ± ε](../assets/ppo.svg)

**Współczynnik ważności.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

To iloraz wiarygodności nowej polityki względem polityki, która zebrała dane. `r_t = 1` oznacza brak zmiany. `r_t = 2` oznacza, że nowa polityka jest dwa razy bardziej skłonna wykonać `a_t` niż stara.

**Przycięty cel zastępczy.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Dwa wyrazy:

- Jeśli przewaga `A_t > 0`, a współczynnik próbuje urosnąć powyżej `1 + ε`, przycięcie spłaszcza gradient — nie pchaj dobrej akcji dalej niż `+ε` powyżej starego prawdopodobieństwa.
- Jeśli przewaga `A_t < 0`, a współczynnik próbuje urosnąć powyżej `1 - ε` (co oznaczałoby, że czynimy złą akcję bardziej prawdopodobną w porównaniu do jej przyciętej redukcji), przycięcie ogranicza gradient — nie pchaj złej akcji poniżej `-ε`.

`min` obsługuje drugi kierunek: jeśli współczynnik przesunął się w *korzystnym* kierunku, wciąż dostajesz gradient (brak przycięcia po stronie, która by cię zranić).

Typowe `ε = 0.2`. Narysuj cel jako funkcję `r_t`: funkcję odcinkowo liniową z płaskim dachem po "dobrej stronie" i płaską podłogą po "złej stronie."

**Pełna strata PPO.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Ta sama struktura aktor-krytyk co A2C. Trzy współczynniki, zwykle `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Pętla treningowa.**

1. Zbierz `N × T` tranzycji z `N` równoległych środowisk po `T` kroków każde.
2. Oblicz przewagi (GAE), zamroź je jako stałe.
3. Zamroź `π_{θ_old}` jako migawkę bieżącej `π_θ`.
4. Przez `K` epok, dla każdej minipartii `(s, a, A, V_target, log π_old(a|s))`:
   - Oblicz `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Zastosuj `L^{CLIP}` + strata wartości + entropia.
   - Krok gradientowy.
5. Odrzuć wykonanie. Wróć do kroku 1.

`K = 10` i minipartie po 64 to standardowy zestaw hiperparametrów. PPO jest wytrzymałe: dokładne liczby rzadko mają znaczenie w granicach ±50%.

**Wariant z karą KL.** Oryginalna praca zaproponowała alternatywę używającą adaptacyjnej kary KL: `L = L^{PG} - β · KL(π_θ || π_old)` z `β` dostosowywanym na podstawie obserwowanego KL. Wersja z przycięciem stała się dominująca; wariant KL przetrwa w RLHF (gdzie KL do polityki referencyjnej jest osobnym ograniczeniem, które i tak zawsze chcesz).

## Zbuduj To

### Krok 1: przechwyć `log π_old(a | s)` w czasie wykonania

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Migawka jest robiona raz, w czasie wykonania. Nie zmienia się podczas epok aktualizacji.

### Krok 2: oblicz przewagi GAE (Lekcja 07)

Tak samo jak w A2C. Normalizuj w całej partii.

### Krok 3: aktualizacja przyciętego celu zastępczego

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Wzorzec "przycięte → zerowy gradient" jest sercem PPO. Jeśli nowa polityka już zbytnio odbiegła w korzystnym kierunku, aktualizacja się zatrzymuje.

### Krok 4: wartość i entropia

Dodaj standardowe MSE do celu krytyka i bonus entropii dla aktora, tak samo jak w A2C.

### Krok 5: diagnostyka

Trzy rzeczy do obserwowania przy każdej aktualizacji:

- **Średnie KL** `E[log π_old - log π_θ]`. Powinno pozostać w `[0, 0.02]`. Jeśli przekroczy `0.1`, zmniejsz `K_EPOCHS` lub `LR`.
- **Frakcja przycięcia** — frakcja próbek, których współczynnik znajduje się poza `[1-ε, 1+ε]`. Powinna wynosić `~0.1-0.3`. Jeśli `~0`, przycięcie nigdy nie działa → podnieś `LR` lub `K_EPOCHS`. Jeśli `~0.5+`, przeuczasz się na wykonaniu → obniż je.
- **Wyjaśniona wariancja** `1 - Var(V_target - V_pred) / Var(V_target)`. Metryka jakości krytyka. Powinna rosnąć w kierunku 1, gdy krytyk się uczy.

## Pułapki

- **Nieprawidłowo dobrany współczynnik przycięcia.** `ε = 0.2` to de facto standard. Przejście na `0.1` czyni aktualizacje zbyt nieśmiałymi; `0.3+` zaprasza niestabilność.
- **Zbyt wiele epok.** `K > 20` rutynowo destabilizuje, ponieważ polityka odbiega daleko od `π_old`. Ogranicz epoki, szczególnie dla dużych sieci.
- **Brak normalizacji nagrody.** Duże skale nagród wchodzą w zakres przycięcia. Normalizuj nagrody (bieżące odchylenie standardowe) przed obliczeniem przewag.
- **Zapomnienie normalizacji przewagi.** Normalizacja na partię do zerowej średniej/jednostkowego odchylenia standardowego jest standardem. Pominięcie jej rujnuje PPO na większości benchmarków.
- **Współczynnik uczenia nie zanika.** PPO korzysta z liniowego zaniku LR do zera. Stałe LR jest często gorsze.
- **Błędy matematyczne współczynnika ważności.** Zawsze `exp(log_new - log_old)` dla stabilności numerycznej, nie `new / old`.
- **Zły znak gradientu.** Maksymalizuj cel zastępczy = *minimalizuj* `-L^{CLIP}`. Odwrócony znak to najczęstszy błąd PPO.

## Użyj Tego

PPO jest domyślnym algorytmem RL w 2026 w zaskakująco wielu domenach:

| Zastosowanie | Wariant PPO |
|----------|-------------|
| MuJoCo / sterowanie robotyczne | PPO z polityką Gaussa, GAE(0.95) |
| Atari / gry dyskretne | PPO z polityką kategorialną, 128-krokowe wykonania |
| RLHF dla LLM | PPO z karą KL do modelu referencyjnego, nagroda z RM na końcu odpowiedzi |
| Agenci gier na dużą skalę | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Rozumujące LLM | GRPO (Lekcja 12) — wariant PPO bez krytyka |
| Tylko dane preferencyjne | DPO — złożenie PPO+KL w formie zamkniętej, bez próbkowania online |

*Kształt straty* PPO — przycięty cel zastępczy + wartość + entropia — jest rusztowaniem dla DPO, GRPO i prawie każdego pipeline'u RLHF.

## Dostarcz To

Zapisz jako `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Ćwiczenia

1. **Łatwe.** Uruchom PPO na GridWorld 4×4 z `ε=0.2, K=4`. Porównaj efektywność próbkową z A2C (jedna epoka na wykonanie) przy dopasowanej liczbie kroków środowiska.
2. **Średnie.** Przeskanuj `K ∈ {1, 4, 10, 30}`. Narysuj zwrot vs kroki środowiska i śledź średnie KL na aktualizację. Przy jakim `K` KL eksploduje w tym zadaniu?
3. **Trudne.** Zastąp przycięty cel zastępczy adaptacyjną karą KL (`β` podwajane, jeśli `KL > 2·target`, zmniejszane o połowę, jeśli `KL < target/2`). Porównaj końcowy zwrot, stabilność i brak przycięcia.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Współczynnik ważności | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; odchylenie od polityki, która zebrała dane. |
| Przycięty cel zastępczy | "Główna sztuczka PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; spłaszczony gradient po przycięciu po korzystnej stronie. |
| Region zaufania | "Intencja TRPO / PPO" | Ogranicz każdą aktualizację KL, aby zagwarantować monotoniczną poprawę. |
| Kara KL | "Miękki region zaufania" | Alternatywne PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptacyjne `β`. |
| Frakcja przycięcia | "Jak często przycięcie działa" | Diagnostyka — powinna wynosić 0.1-0.3; poza tym źle dostrojone. |
| Trening wieloepokowy | "Wielokrotne użycie danych" | K epok na każdym wykonaniu; koszt wariancji wymieniony na efektywność próbkową. |
| Mniej-więcej w ramach polityki | "Głównie w ramach polityki" | PPO nominalnie w ramach polityki, ale K>1 epok bezpiecznie używa lekko poza-politycznych danych. |
| PPO-KL | "To drugie PPO" | Wariant z karą KL; używane w RLHF, gdzie KL-do-referencji jest już ograniczeniem. |

## Dalsza Lektura

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — praca.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, poprzednik PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — każdy hiperparametr PPO poddany ablacji.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; przepis PPO-w-RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — czysty nowoczesny wykład z PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — referencyjny jednowątkowy PPO używany przez wiele artykułów.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — produkcyjny przepis na PPO na modelach językowych; czytaj obok Lekcji 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — praca o "37 optymalizacjach na poziomie kodu"; które sztuczki PPO są nośne, a które folklorem.