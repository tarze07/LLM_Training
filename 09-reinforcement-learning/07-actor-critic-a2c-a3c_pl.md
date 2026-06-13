# Aktor-Krytyk — A2C i A3C

> REINFORCE jest zaszumione. Dodaj krytyka, który uczy się `V̂(s)`, odejmij go od zwrotu, a otrzymasz przewagę o tej samej wartości oczekiwanej, ale znacznie niższej wariancji. To jest aktor-krytyk. A2C działa synchronicznie; A3C działa wielowątkowo. Oba są modelem mentalnym dla każdej nowoczesnej metody głębokiego RL.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## Problem

Czysty REINFORCE działa, ale jego wariancja jest straszna. Zwroty Monte Carlo `G_t` mogą wahać się o rząd wielkości między epizodami. Mnożenie tego szumu przez `∇ log π` i uśrednianie daje estymator gradientu, który potrzebuje tysięcy epizodów, aby przesunąć politykę na tę samą odległość, co znacznie mniej aktualizacji DQN.

Wariancja pochodzi z używania surowych zwrotów. Jeśli odejmiesz wartość bazową `b(s_t)` — dowolną funkcję stanu, w tym wyuczoną wartość — wartość oczekiwana jest niezmieniona, a wariancja spada. Najlepszą możliwą do obliczenia wartością bazową jest `V̂(s_t)`. Teraz wielkość mnożąca `∇ log π` to *przewaga*:

`A(s, a) = G - V̂(s)`

Akcja jest dobra, jeśli dała ponadprzeciętny zwrot; zła, jeśli poniżej. REINFORCE z wyuczonym krytykiem to *aktor-krytyk*. Krytyk daje aktorowi nauczyciela o niskiej wariancji. To jest każda metoda głębokiej polityki po 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Koncepcja

![Aktor-krytyk: sieć polityki plus sieć wartości, reszta TD jako przewaga](../assets/actor-critic.svg)

**Dwie sieci, jedna wspólna strata:**

- **Aktor** `π_θ(a | s)`: polityka. Próbkowana, aby działać. Trenowana gradientem polityki.
- **Krytyk** `V_φ(s)`: estymuje oczekiwany zwrot ze stanu. Trenowany, aby minimalizować `(V_φ(s) - target)²`.

**Przewaga.** Dwie standardowe formy:

- *Przewaga MC:* `A_t = G_t - V_φ(s_t)`. Nieobciążona, wyższa wariancja.
- *Przewaga TD:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Obciążona (używa `V_φ`), znacznie niższa wariancja. Nazywana też *resztą TD* `δ_t`.

**Przewaga n-krokowa.** Interpoluj między tymi dwoma:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` to czyste TD. `n = ∞` to MC. Większość implementacji używa `n = 5` dla Atari, `n = 2048` dla PPO na MuJoCo.

**Uogólniona estymacja przewagi (GAE).** Schulman i in. (2016) zaproponowali wykładniczo ważoną średnią po wszystkich n-krokowych przewagach:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

z `λ ∈ [0, 1]`. `λ = 0` to TD (niska wariancja, wysokie obciążenie). `λ = 1` to MC (wysoka wariancja, nieobciążone). `λ = 0.95` to domyślne ustawienie w 2026 — dostrajaj, aż pokrętło obciążenie/wariancja będzie tam, gdzie chcesz.

**A2C: synchroniczny aktor-krytyk przewagi.** Zbierz `T` kroków z `N` równoległych środowisk. Oblicz przewagi dla każdego kroku. Zaktualizuj aktora i krytyka na połączonej partii. Powtórz. Prostsze, bardziej skalowalne rodzeństwo A3C.

**A3C: asynchroniczny aktor-krytyk przewagi.** Mnih i in. (2016). Utwórz `N` wątków roboczych, każdy uruchamiający środowisko. Każdy robotnik oblicza gradienty lokalnie na swoim własnym wykonaniu, a następnie asynchronicznie stosuje je do współdzielonego serwera parametrów. Żaden bufor replay nie jest potrzebny — robotnicy dekorelują przez uruchamianie różnych trajektorii. A3C udowodniło, że można trenować na CPU w skali. W 2026, A2C oparty na GPU (środowiska równoległe wsadowo) dominuje, ponieważ GPU chcą dużych partii.

**Połączona strata.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Trzy wyrazy: strata gradientu polityki, regresja wartości, bonus entropii. `c_v ~ 0.5`, `c_e ~ 0.01` to kanoniczne punkty startowe.

## Zbuduj To

### Krok 1: krytyk

Liniowy krytyk `V_φ(s) = w · features(s)` aktualizowany MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

W środowisku tabelarycznym krytyk zbiega w kilkuset epizodach. Na Atari, zastąp liniowego krytyka współdzielonym pniem CNN + głową wartości.

### Krok 2: przewaga n-krokowa

Dla wykonania długości `T` i uruchomionego końcowego `V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` to cel krytyka. `advantages` jest tym, co mnoży `∇ log π`.

### Krok 3: połączona aktualizacja

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

W ramach polityki, jedno wykonanie na aktualizację, oddzielne współczynniki uczenia dla aktora i krytyka.

### Krok 4: równoległość (A3C vs A2C)

- **A3C:** uruchom `N` wątków. Każdy ma własne środowisko i własny przebieg w przód. Okresowo wysyłaj aktualizacje gradientu do współdzielonego mastera. Żadnych blokad na masterze — wyścigi są w porządku, po prostu dodają szum.
- **A2C:** uruchom `N` instancji środowiska w jednym procesie, ułóż obserwacje w partię `[N, obs_dim]`, wsadowy przebieg w przód, wsadowy przebieg wstecz. Wyższe wykorzystanie GPU, deterministyczne, łatwiejsze do rozumowania. Domyślne w 2026.

Nasz zabawkowy kod jest jednowątkowy dla przejrzystości; przepisanie na wsadowy A2C to trzy linie numpy.

## Pułapki

- **Obciążenie krytyka przed gradientem aktora.** Jeśli krytyk jest losowy, jego wartość bazowa jest nieinformacyjna i trenujesz na czystym szumie. Rozgrzej krytyka przez kilkaset kroków przed włączeniem gradientu polityki lub użyj wolnego współczynnika uczenia aktora.
- **Normalizacja przewagi.** Normalizuj przewagi do zerowej średniej/jednostkowego odchylenia standardowego na partię. Stabilizuje trening ogromnie przy prawie zerowym koszcie.
- **Współdzielony pień.** Użyj współdzielonego ekstraktora cech dla aktora i krytyka na wejściach obrazowych. Oddzielne głowy. Współdzielone cechy korzystają z obu strat.
- **Kontrakt w ramach polityki.** A2C używa danych dokładnie raz. Więcej, a twój gradient jest obciążony (korekta próbkowania ważności to właśnie to, co dodaje PPO).
- **Zapadnięcie entropii.** Bez `c_e > 0`, polityka staje się prawie deterministyczna w kilkuset aktualizacjach i przestaje eksplorować.
- **Skala nagrody.** Wielkości przewagi zależą od skali nagrody. Normalizuj nagrody (np. dzielenie przez bieżące odchylenie standardowe) dla spójnych wielkości gradientów w różnych zadaniach.

## Użyj Tego

A2C/A3C rzadko są ostatecznym wyborem w 2026, ale są architekturą, którą wszystko później udoskonala:

| Metoda | Relacja do A2C |
|--------|----------------|
| PPO | A2C + przycięty współczynnik ważności dla wieloepokowych aktualizacji |
| IMPALA | A3C + korekta poza polityką V-trace |
| SAC (Faza 9 · 07) | A2C poza polityką z krytykiem miękkiej wartości (następna lekcja) |
| GRPO (Faza 9 · 12) | A2C bez krytyka — przewaga względna grupowo |
| DPO | A2C zwinięte w stratę rankingu preferencji, bez próbkowania |
| AlphaStar / OpenAI Five | A2C z treningiem ligowym + wstępnym treningiem imitacyjnym |

Jeśli widzisz "przewaga" w artykule z 2026, myśl aktor-krytyk.

## Dostarcz To

Zapisz jako `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Ćwiczenia

1. **Łatwe.** Trenuj aktora-krytyka z przewagą MC (`G_t - V(s_t)`) na GridWorld 4×4. Porównaj efektywność próbkową z REINFORCE z wartością bazową średniej kroczącej z Lekcji 06.
2. **Średnie.** Przełącz się na przewagę reszty TD (`r + γ V(s') - V(s)`). Zmierz wariancję partii przewag. O ile spada?
3. **Trudne.** Zaimplementuj GAE(λ). Przeskanuj `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Narysuj końcowy zwrot vs efektywność próbkową. Gdzie jest optymalny punkt obciążenie/wariancja dla tego zadania?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Aktor | "Sieć polityki" | `π_θ(a\|s)`, aktualizowana przez gradient polityki. |
| Krytyk | "Sieć wartości" | `V_φ(s)`, aktualizowana przez regresję MSE do zwrotów / celów TD. |
| Przewaga | "O ile lepiej niż średnio" | `A(s, a) = Q(s, a) - V(s)` lub jej estymatory. Mnożnik dla `∇ log π`. |
| Reszta TD | "δ" | `δ_t = r + γ V(s') - V(s)`; jednokrokowy estymator przewagi. |
| GAE | "Pokrętło interpolacji" | Wykładniczo ważona suma n-krokowych przewag, parametryzowana przez `λ`. |
| A2C | "Synchroniczny aktor-krytyk" | Wsadowe przez środowiska; jeden krok gradientu na wykonanie. |
| A3C | "Asynchroniczny aktor-krytyk" | Wątki robotnicze wysyłają gradienty do współdzielonego serwera parametrów. Oryginalna praca; mniej powszechne w 2026. |
| Uruchamianie | "Użyj V na horyzoncie" | Skróć wykonanie, dodaj `γ^n V(s_{t+n})`, aby zamknąć sumę. |

## Dalsza Lektura

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, oryginalna praca o asynchronicznym aktorze-krytyku.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — podstawy; połącz z Rozdz. 9 o aproksymacji funkcji, gdy krytyk jest siecią neuronową.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — skalowalny rozproszony aktor-krytyk z korektą poza polityką V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — produkcyjne implementacje A2C/PPO warte przeczytania.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — fundamentalny wynik zbieżności dla dekompozycji aktor-krytyk w dwóch skalach czasu.