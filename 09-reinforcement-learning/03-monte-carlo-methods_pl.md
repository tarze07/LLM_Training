# Metody Monte Carlo — Uczenie się z Kompletnych Epizodów

> Programowanie dynamiczne potrzebuje modelu. Monte Carlo nie potrzebuje niczego poza epizodami. Uruchom politykę, obserwuj zwroty, uśrednij je. Najprostszy pomysł w RL — i ten, który odblokowuje wszystko, co następuje.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## Problem

Programowanie dynamiczne jest eleganckie, ale zakłada, że możesz zapytać `P(s' | s, a)` dla każdego stanu i akcji. Prawie nic w rzeczywistym świecie tak nie działa. Robot nie może analitycznie obliczyć rozkładu pikseli kamery po momencie obrotowym w stawie. Algorytm wyceny nie może scałkować po każdej możliwej reakcji klienta. LLM nie może wyliczyć wszystkich możliwych kontynuacji po tokenie.

Potrzebujesz metody, która wymaga jedynie umiejętności *próbkowania* ze środowiska. Uruchom politykę. Uzyskaj trajektorię `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Użyj jej do oszacowania wartości. To jest Monte Carlo.

Przejście od PD do MC jest filozoficznie ważne: przechodzimy od *znanego modelu + dokładnego kroku wstecz* do *próbkowanych przebiegów + uśrednionego zwrotu*. Wariancja rośnie, ale zastosowalność eksploduje. Każdy algorytm RL po tej lekcji — TD, Q-learning, REINFORCE, PPO, GRPO — jest w swej istocie estymatorem Monte Carlo, czasem z dodatkowym uruchamianiem na wyższych poziomach.

## Koncepcja

![Monte Carlo: wykonanie, obliczenie zwrotów, uśrednienie; pierwsze-odwiedziny vs każde-odwiedziny](../assets/monte-carlo.svg)

**Główna idea w jednej linii:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` gdzie `G^{(i)}(s)` to zaobserwowane zwroty po odwiedzinach `s` przy polityce `π`.

**MC pierwsze-odwiedziny vs każde-odwiedziny.** Dla epizodu odwiedzającego stan `s` wielokrotnie, MC pierwsze-odwiedziny liczy tylko zwrot z pierwszej wizyty; MC każde-odwiedziny liczy wszystkie. Oba są nieobciążone w granicy. Pierwsze-odwiedziny jest prostsze do analizy (próbki iid). Każde-odwiedziny wykorzystuje więcej danych na epizod i zazwyczaj zbiega szybciej w praktyce.

**Średnia inkrementalna.** Zamiast przechowywać wszystkie zwroty, aktualizuj średnią kroczącą:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Przeorganizuj: `V_new = V_old + α · (target - V_old)` z `α = 1/n`. Zamień `1/n` na stały rozmiar kroku `α ∈ (0, 1)`, a otrzymasz niestacjonarny estymator MC, który śledzi zmiany w `π`. Ten ruch to cały skok od MC do TD do każdego nowoczesnego algorytmu RL.

**Eksploracja jest teraz problemem.** PD dotykał każdego stanu przez wyliczenie. MC widzi tylko stany, które polityka odwiedza. Jeśli `π` jest deterministyczna, całe regiony przestrzeni stanów nigdy nie są próbkowane, a ich oszacowania wartości pozostają na zero. Trzy poprawki, w porządku historycznym:

1. **Starty eksploracyjne.** Rozpocznij każdy epizod od losowej pary (s, a). Gwarantuje pokrycie; nierealistyczne w praktyce (nie możesz "zresetować" robota do dowolnego stanu).
2. **ε-zachłanna.** Działaj zachłannie względem bieżącego Q, ale z prawdopodobieństwem `ε` wybierz losową akcję. Wszystkie pary stan-akcja są asymptotycznie próbkowane.
3. **MC poza polityką.** Zbieraj dane przy polityce behawioralnej `μ`, ucz się o polityce docelowej `π` przez próbkowanie ważności. Wysoka wariancja, ale jest to most do metod z buforem replay, takich jak DQN.

**Sterowanie Monte Carlo.** Oceń → popraw → oceń, tak jak w iteracji polityki, ale ewaluacja opiera się na próbkowaniu:

1. Uruchom `π`, uzyskaj epizod.
2. Zaktualizuj `Q(s, a)` z zaobserwowanych zwrotów.
3. Uczyń `π` ε-zachłanną względem `Q`.
4. Powtórz.

Zbiega do `Q*` i `π*` z prawdopodobieństwem 1 przy łagodnych warunkach (każda para odwiedzana nieskończenie często, `α` spełnia Robbins-Monro).

```figure
epsilon-greedy
```

## Zbuduj To

### Krok 1: wykonanie → lista (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Bez modelu, tylko `env.reset()` i `env.step(s, a)`. Ten sam interfejs co środowisko gym, ale uproszczony.

### Krok 2: oblicz zwroty (przebieg wsteczny)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Jeden przebieg, `O(T)`. Rekurencja wsteczna `G_t = r_{t+1} + γ G_{t+1}` unika ponownego sumowania.

### Krok 3: ewaluacja MC pierwsze-odwiedziny

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Trzy linie robią robotę: oznacz stan jako widziany przy pierwszej wizycie, zwiększ licznik, zaktualizuj średnią kroczącą.

### Krok 4: sterowanie MC ε-zachłanne (w ramach polityki)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Krok 5: porównaj ze złotym standardem PD

Twoje oszacowanie MC `V^π` powinno zgadzać się z wynikiem PD z Lekcji 02, gdy epizody → ∞. W praktyce: 50 000 epizodów na GridWorld 4×4 daje wynik w granicach `~0.1` od odpowiedzi PD.

## Pułapki

- **Nieskończone epizody.** MC wymaga, aby epizody się *kończyły*. Jeśli twoja polityka może zapętlić się w nieskończoność, ogranicz `max_steps` i traktuj to jako domyślną porażkę. GridWorld z losową polityką rutynowo przekracza limit czasu — to normalne, po prostu upewnij się, że liczysz to poprawnie.
- **Wariancja.** MC używa pełnych zwrotów. Przy długich epizodach wariancja jest ogromna — jedna pechowa nagroda na końcu przesuwa `V(s_0)` o tę samą wartość. Metody TD (Lekcja 04) redukują to przez uruchamianie (bootstrapping).
- **Pokrycie stanów.** Zachłanne MC na świeżym Q z remisami wypróbuje tylko jedną akcję. *Musisz* eksplorować (ε-zachłanne, starty eksploracyjne, UCB).
- **Niestacjonarne polityki.** Jeśli `π` zmienia się (jak w sterowaniu MC), stare zwroty pochodzą z innej polityki. MC ze stałym α sobie z tym radzi; MC ze średnią próbkową nie.
- **Próbkowanie ważności poza polityką.** Wagi `π(a|s)/μ(a|s)` mnożą się wzdłuż trajektorii. Wariancja eksploduje z horyzontem. Ogranicz przez ważone IS na decyzję lub przełącz się na TD.

## Użyj Tego

Rola metod Monte Carlo w 2026:

| Zastosowanie | Dlaczego MC |
|----------|--------|
| Gry o krótkim horyzoncie (blackjack, poker) | Epizody kończą się naturalnie; zwroty są czyste. |
| Offline'owa ocena zarejestrowanej polityki | Średnia zdyskontowanych zwrotów z zapisanych trajektorii. |
| Monte Carlo Tree Search (AlphaZero) | Przebiegi MC z liści drzewa prowadzą selekcję. |
| Ocena RL dla LLM | Oblicz średnią nagrodę z próbkowanych kompletacji dla danej polityki. |
| Estymacja wartości bazowej w PPO | Cel przewagi `A_t = G_t - V(s_t)` używa MC `G_t`. |
| Nauczanie RL | Najprostszy algorytm, który faktycznie działa — usuń uruchamianie, aby zobaczyć rdzeń. |

Nowoczesne algorytmy głębokiego RL (PPO, SAC) interpolują między czystym MC (pełne zwroty) a czystym TD (jednokrokowe uruchomienie) przez zwroty `n`-krokowe lub GAE. Oba końce są instancjami tego samego estymatora.

## Dostarcz To

Zapisz jako `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj ewaluację MC pierwsze-odwiedziny dla jednostajnej losowej polityki na GridWorld 4×4. Uruchom 10 000 epizodów. Narysuj wykres `V(0,0)` w funkcji liczby epizodów wraz z odpowiedzią PD.
2. **Średnie.** Zaimplementuj ε-zachłanne sterowanie MC z `ε ∈ {0.01, 0.1, 0.3}`. Porównaj średni zwrot po 20 000 epizodów. Jak wygląda krzywa? Gdzie leży kompromis błąd-wariancja?
3. **Trudne.** Zaimplementuj MC *poza polityką* z próbkowaniem ważności: zbieraj dane przy jednostajnej losowej polityce `μ`, oszacuj `V^π` dla deterministycznej optymalnej polityki `π`. Porównaj zwykłe IS, IS na decyzję i ważone IS. Które ma najniższą wariancję?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Monte Carlo | "Losowe próbkowanie" | Estymuj wartości oczekiwane przez uśrednianie po próbkach iid z rozkładu. |
| Zwrot `G_t` | "Przyszła nagroda" | Suma zdyskontowanych nagród od kroku `t` do końca epizodu: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| MC pierwsze-odwiedziny | "Policz każdy stan raz" | Tylko pierwsza wizyta w epizodzie wnosi wkład do oszacowania wartości. |
| MC każde-odwiedziny | "Użyj wszystkich wizyt" | Każda wizyta wnosi wkład; lekko obciążone, ale bardziej efektywne próbkowo. |
| ε-zachłanne | "Szum eksploracyjny" | Wybierz zachłanną akcję z prob `1-ε`; losową akcję z prob `ε`. |
| Próbkowanie ważności | "Korekta za próbkowanie z niewłaściwego rozkładu" | Przeważ zwroty przez iloczyny `π(a\|s)/μ(a\|s)`, aby oszacować `V^π` z danych `μ`. |
| W ramach polityki | "Ucz się z własnych danych" | Polityka docelowa = polityka behawioralna. Zwykłe MC, PPO, SARSA. |
| Poza polityką | "Ucz się z danych kogoś innego" | Polityka docelowa ≠ polityka behawioralna. MC z próbkowaniem ważności, Q-learning, DQN. |

## Dalsza Lektura

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — kanoniczne opracowanie.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — analiza pierwsze-odwiedziny vs każde-odwiedziny.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — MC poza polityką i kontrola wariancji.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — nowoczesne estymatory IS o niskiej wariancji.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — pierwsza duża empiryczna demonstracja samodzielnej gry MC/TD osiągającej nadludzki poziom; koncepcyjny prekursor każdej lekcji w drugiej połowie tej fazy.