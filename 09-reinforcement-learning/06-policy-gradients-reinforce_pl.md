# Gradient Polityki — REINFORCE od Zera

> Przestań estymować wartość. Sparametryzuj politykę bezpośrednio, oblicz gradient oczekiwanego zwrotu, idź pod górkę. Williams (1992) napisał to w jednym twierdzeniu. To dlatego istnieją PPO, GRPO i każda pętla RL dla LLM.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## Problem

Q-learning i DQN parametryzują funkcję *wartości*. Wybierasz akcje przez `argmax Q`. To działa dla dyskretnych akcji i stanów. Załamuje się, gdy akcje są ciągłe (jaki `argmax` nad 10-wymiarowym momentem obrotowym?) lub gdy chcesz stochastycznej polityki (`argmax` jest z definicji deterministyczny).

Gradienty polityki parametryzują *politykę* zamiast tego. `π_θ(a | s)` to sieć neuronowa, która wypisuje rozkład nad akcjami. Próbkuj z niej, aby działać. Oblicz gradient oczekiwanego zwrotu względem `θ`. Idź pod górkę. Bez `argmax`. Bez rekurencji Bellmana. Tylko gradient wstępujący na `J(θ) = E_{π_θ}[G]`.

Twierdzenie REINFORCE (Williams 1992) mówi, że ten gradient jest obliczalny: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Uruchom epizod. Oblicz zwrot. Pomnóż przez `∇ log π_θ(a | s)` na każdym kroku. Uśrednij. Gradient wstępujący. Gotowe.

Każdy algorytm RL dla LLM w 2026 — PPO, DPO, GRPO — jest udoskonaleniem REINFORCE. Zrozumienie go na wyczucie jest warunkiem wstępnym dla reszty tej fazy oraz dla Fazy 10 · 07 (implementacja RLHF) i Fazy 10 · 08 (DPO).

## Koncepcja

![Gradient polityki: polityka softmax, gradient log-π, aktualizacja ważona zwrotem](../assets/policy-gradient.svg)

**Twierdzenie o gradiencie polityki.** Dla dowolnej polityki `π_θ` parametryzowanej przez `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

gdzie `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` to zdyskontowany zwrot od kroku `t`. Wartość oczekiwana jest po pełnych trajektoriach `τ` próbkowanych z `π_θ`.

**Dowód jest krótki.** Zróżniczkuj `J(θ) = Σ_τ P(τ; θ) G(τ)` pod wartością oczekiwaną. Użyj `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (sztuczka log-pochodnej). Rozłóż `log P(τ; θ) = Σ log π_θ(a_t | s_t) + wyrazy środowiskowe niezależne od θ`. Wyrazy środowiskowe znikają. Dwie linie algebry dają twierdzenie.

**Sztuczki redukcji wariancji.** Czysty REINFORCE ma morderczą wariancję — zwroty są zaszumione, `∇ log π` jest zaszumione, ich iloczyn jest bardzo zaszumiony. Dwie standardowe poprawki:

1. **Odejmowanie wartości bazowej.** Zastąp `G_t` przez `G_t - b(s_t)` dla dowolnej wartości bazowej `b(s_t)` niezależnej od `a_t`. Nieobciążone, ponieważ `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Typowy wybór: `b(s_t) = V̂(s_t)` wyuczone przez krytyka → aktor-krytyk (Lekcja 07).
2. **Zwrot od teraz.** Zastąp `Σ_t G_t · ∇ log π_θ(a_t | s_t)` przez `Σ_t G_t^{od t} · ∇ log π_θ(a_t | s_t)`. Tylko przyszłe zwroty mają znaczenie dla danej akcji — przeszłe nagrody wnoszą szum o zerowej średniej.

Połączone dają:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

czyli REINFORCE z wartością bazową — bezpośredni przodek A2C (Lekcja 07) i PPO (Lekcja 08).

**Parametryzacja polityki softmax.** Dla dyskretnych akcji, standardowy wybór:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

gdzie `f_θ` to dowolna sieć neuronowa wypisująca wynik na akcję. Gradient ma czystą postać:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

tzn. wynik wykonanej akcji minus jego wartość oczekiwana przy polityce.

**Polityka Gaussa dla akcji ciągłych.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` ma postać zamkniętą. To wszystko, czego potrzebuje SAC z Fazy 9 · 07.

```figure
policy-gradient-landscape
```

## Zbuduj To

### Krok 1: sieć polityki softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Użyj liniowej polityki (jeden wektor wag na akcję) dla środowiska tabelarycznego. Dla Atari, wstaw CNN i zachowaj głowę softmax.

### Krok 2: próbkowanie i log-prawdopodobieństwo

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Krok 3: wykonanie z przechwyconymi log-prawdopodobieństwami

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Krok 4: aktualizacja REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (one-hot `a` minus prawdopodobieństwa) jest sercem gradientów polityki softmax. Wpisz go w pamięć mięśniową.

### Krok 5: wartości bazowe

Średnia krocząca `G` z ostatnich epizodów to wystarczająca redukcja wariancji, aby uruchomić GridWorld 4×4; potrzeba ~500 epizodów do zbieżności. Ulepsz wartość bazową do wyuczonego `V̂(s)`, a otrzymasz aktora-krytyka.

## Pułapki

- **Eksplodujące gradienty.** Zwroty mogą być ogromne. Zawsze normalizuj `G` do `~N(0, 1)` w całej partii przed mnożeniem przez `∇ log π`.
- **Zapadnięcie entropii.** Polityka zbiega do prawie deterministycznej akcji zbyt wcześnie, przestaje eksplorować i utyka. Poprawka: dodaj bonus entropii `β · H(π(·|s))` do celu.
- **Wysoka wariancja.** Czysty REINFORCE potrzebuje tysięcy epizodów. Wartość bazowa krytyka (Lekcja 07) lub region zaufania TRPO/PPO (Lekcja 08) to standardowa poprawka.
- **Nieefektywność próbkowa.** W ramach polityki oznacza, że wyrzucasz każdą tranzycję po jednej aktualizacji. Korekty poza polityką przez próbkowanie ważności przywracają dane, kosztem wariancji (współczynnik PPO to przycięta waga IS).
- **Niestacjonarne gradienty.** Ten sam gradient sprzed 100 epizodów używa starego `π`. Metody w ramach polityki aktualizują co kilka wykonań z tego powodu.
- **Przypisanie odpowiedzialności.** Bez zwrotu od teraz, przeszłe nagrody wnoszą szum. Zawsze używaj zwrotu od teraz.

## Użyj Tego

W 2026 REINFORCE rzadko jest uruchamiane bezpośrednio, ale jego wzór gradientu jest wszędzie:

| Zastosowanie | Metoda pochodna |
|----------|---------------|
| Sterowanie ciągłe | PPO / SAC z polityką Gaussa |
| RLHF dla LLM | PPO z karą KL, działające na poziomie tokena |
| Rozumowanie LLM (DeepSeek) | GRPO — REINFORCE z wartością bazową względną grupowo, bez krytyka |
| Wieloagentowe | Scentralizowany-krytyk REINFORCE (MADDPG, COMA) |
| Robotyka z dyskretnymi akcjami | A2C, A3C, PPO |
| Tylko preferencje | DPO — REINFORCE przepisany jako strata preferencyjna, bez próbkowania |

Gdy czytasz `loss = -advantage * log_prob` w skrypcie treningowym z 2026, to jest REINFORCE z wartością bazową. Całe artykuły (DPO, GRPO, RLOO) to sztuczki redukcji wariancji na tej jednej linii.

## Dostarcz To

Zapisz jako `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj REINFORCE na GridWorld 4×4 z liniową polityką softmax. Trenuj przez 1000 epizodów bez wartości bazowej. Narysuj krzywą uczenia; zmierz wariancję (odchylenie standardowe zwrotów).
2. **Średnie.** Dodaj wartość bazową jako średnią kroczącą. Trenuj ponownie. Porównaj efektywność próbkową i wariancję z czystym uruchomieniem. O ile wartość bazowa redukuje liczbę kroków do zbieżności?
3. **Trudne.** Dodaj bonus entropii `β · H(π)`. Przeskanuj `β ∈ {0, 0.01, 0.1, 1.0}`. Narysuj końcowy zwrot i entropię polityki. Gdzie jest optymalny punkt dla tego zadania?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Gradient polityki | "Trenuj politykę bezpośrednio" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; wyprowadzone ze sztuczki log-pochodnej. |
| REINFORCE | "Oryginalny algorytm PG" | Williams (1992); zwroty Monte Carlo pomnożone przez gradient log-polityki. |
| Sztuczka log-pochodnej | "Estymator funkcji wyniku" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; czyni gradienty wartości oczekiwanych możliwymi do obliczenia. |
| Wartość bazowa | "Redukcja wariancji" | Dowolne `b(s)` odjęte od `G`; nieobciążone, ponieważ `E[b · ∇ log π] = 0`. |
| Zwrot od teraz | "Tylko przyszłe zwroty się liczą" | `G_t^{od t}` zamiast pełnego `G_0`; poprawne i o niższej wariancji. |
| Bonus entropii | "Zachęcaj do eksploracji" | Wyraz `+β · H(π(·\|s))` zapobiega zapadnięciu się polityki. |
| W ramach polityki | "Trenuj na tym, co właśnie widziałeś" | Wartość oczekiwana gradientu jest względem bieżącej polityki — nie można bezpośrednio używać starych danych. |
| Przewaga | "O ile lepiej niż średnio" | `A(s, a) = G(s, a) - V(s)`; wielkość ze znakiem, którą mnoży REINFORCE z wartością bazową. |

## Dalsza Lektura

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — oryginalna praca REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — nowoczesne twierdzenie o gradiencie polityki z aproksymacją funkcji.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — prezentacja podręcznikowa.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — jasny pedagogiczny wykład z kodem PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — redukcja wariancji i widok naturalnego gradientu, który łączy REINFORCE z rodziną regionu zaufania (TRPO, PPO).