# MDP, Stany, Akcje i Nagrody

> Proces decyzji Markowa to pięć rzeczy: stany, akcje, przejścia, nagrody, dyskonto. Wszystko w RL — Q-learning, PPO, DPO, GRPO — optymalizuje względem tego kształtu. Naucz się tego raz, a resztę uczenia przez wzmacnianie przeczytasz za darmo.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## Problem

Piszesz bota szachowego. Albo planistę zapasów. Albo agenta transakcyjnego. Albo pętlę PPO trenującą model rozumowania. Cztery różne domeny, jeden zaskakujący fakt: wszystkie sprowadzają się do tego samego matematycznego obiektu.

Uczenie nadzorowane daje pary `(x, y)` i każe dopasować funkcję. Uczenie przez wzmacnianie nie daje etykiet — tylko strumień stanów, wykonanych akcji i skalarną nagrodę. Czy ruch wygrał grę? Czy decyzja o uzupełnieniu zapasów zaoszczędziła pieniądze? Czy transakcja przyniosła zysk? Czy token wyprodukowany przez LLM doprowadził do wyższej nagrody od sędziego?

Nie możesz uczyć się z tego strumienia, dopóki go nie sformalizujesz. "Co widziałem", "co zrobiłem", "co się stało potem", "jak dobre to było" — każde z tych musi stać się obiektem, o którym możesz rozumować. Tą formalizacją jest proces decyzji Markowa. Każdy algorytm RL w tej fazie, włączając pętle RLHF i GRPO na końcu, optymalizuje względem tego kształtu.

## Koncepcja

![Proces decyzji Markowa: stany, akcje, przejścia, nagrody, dyskonto](../assets/mdp.svg)

**Pięć obiektów.**

- **Stany** `S`. Wszystko, czego agent potrzebuje do podjęcia decyzji. W GridWorld — komórka. W szachach — szachownica. W LLM — okno kontekstu plus ewentualna pamięć.
- **Akcje** `A`. Wybory. Ruch góra/dół/lewo/prawo. Wykonanie ruchu. Wyemitowanie tokena.
- **Przejścia** `P(s' | s, a)`. Dla danego stanu `s` i akcji `a`, rozkład następnego stanu. Deterministyczne w szachach, stochastyczne w zapasach, prawie-deterministyczne w dekodowaniu LLM.
- **Nagrody** `R(s, a, s')`. Skalarny sygnał. Wygrana = +1, przegrana = -1. Przychód minus koszt. Współczynnik ilorazu wiarygodności w GRPO.
- **Dyskonto** `γ ∈ [0, 1)`. Jak bardzo przyszła nagroda liczy się względem obecnej. `γ = 0.99` kupuje horyzont ~100 kroków; `γ = 0.9` kupuje ~10.

**Własność Markowa** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Przyszłość zależy tylko od obecnego stanu. Jeśli nie zależy — reprezentacja stanu jest niekompletna, co nie jest błędem metody, lecz błędem stanu.

**Polityki i zwroty.** Polityka `π(a | s)` odwzorowuje stany na rozkłady akcji. Zwrot `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` to zdyskontowana suma przyszłych nagród. Wartość `V^π(s) = E[G_t | s_t = s]` to oczekiwany zwrot startując z `s` według polityki `π`. Wartość Q `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` to oczekiwany zwrot startując z konkretną akcją. Każdy algorytm RL estymuje jedną z tych dwóch, a następnie odpowiednio poprawia `π`.

**Równania Bellmana.** Równania punktu stałego, z których korzysta wszystko w tej fazie:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Dzielą one oczekiwany zwrot na "nagrodę z tego kroku" plus "zdyskontowaną wartość miejsca, w którym wylądujesz". Rekurencyjne. Każdy algorytm w Fazie 9 albo iteruje to równanie do zbieżności (programowanie dynamiczne), albo próbkuje z niego (Monte Carlo), albo uruchamia je o jeden krok (różnica temporalna).

```figure
discount-horizon
```

## Zbuduj To

### Krok 1: mały deterministyczny MDP

GridWorld 4×4. Agent startuje w lewym górnym rogu, terminalny w prawym dolnym, nagroda -1 za krok, akcje `{góra, dół, lewo, prawo}`. Zobacz `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Pięć linii. To całe środowisko. Deterministyczne przejścia, stała kara za krok, absorbujący stan terminalny.

### Krok 2: wykonaj politykę

Polityka to funkcja ze stanu do rozkładu akcji. Najprostsza: jednostajna losowa.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Uruchom losową politykę 1000 razy. Średni zwrot to około -60 do -80 dla tej planszy 4×4. Optymalny zwrot to -6 (ścieżka prosta w dół i w prawo). Zamknięcie tej luki to wszystko w Fazie 9.

### Krok 3: oblicz `V^π` dokładnie przez równanie Bellmana

Dla małych MDP równanie Bellmana to układ liniowy. Wylicz stany, zastosuj wartość oczekiwaną, iteruj, aż wartości przestaną się zmieniać.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

To jest iteracyjna ewaluacja polityki. Jest pierwszym algorytmem w Sutton & Barto i teoretyczną podstawą każdej metody RL, która po nim następuje.

### Krok 4: `γ` jest hiperparametrem o fizycznym znaczeniu

Efektywny horyzont to w przybliżeniu `1 / (1 - γ)`. `γ = 0.9` → 10 kroków. `γ = 0.99` → 100 kroków. `γ = 0.999` → 1000 kroków.

Zbyt niski — agent działa krótkowzrocznie. Zbyt wysoki — przypisanie odpowiedzialności staje się zaszumione, ponieważ wiele wczesnych kroków dzieli odpowiedzialność za odległą przyszłą nagrodę. RLHF dla LLM zwykle używa `γ = 1`, ponieważ epizody są krótkie i ograniczone. Zadania sterowania używają `0.95–0.99`. Gry strategiczne o długim horyzoncie używają `0.999`.

## Pułapki

- **Stan nie-Markowa.** Jeśli potrzebujesz ostatnich trzech obserwacji do podjęcia decyzji, "stan" to nie tylko bieżąca obserwacja. Poprawka: układaj ramki (DQN w Atari układa 4) lub użyj stanu rekurencyjnego (LSTM/GRU na obserwacjach).
- **Rzadkie nagrody.** Nagrody tylko za wygraną sprawiają, że uczenie jest prawie niemożliwe w dużych przestrzeniach stanów. Kształtuj nagrody (sygnał pośredni) lub uruchom z imitacji (Faza 9 · 09).
- **Hakowanie nagrody.** Optymalizacja zastępczej nagrody często prowadzi do patologicznego zachowania. Agent łodzi OpenAI krążył w kółko zbierając power-upy zamiast ukończyć wyścig. Zawsze definiuj nagrodę na podstawie docelowego wyniku, a nie zastępczej funkcji.
- **Błędne dyskonto.** `γ = 1` w zadaniu o nieskończonym horyzoncie sprawia, że każda wartość jest nieskończona. Zawsze ograniczaj skończonym horyzontem lub `γ < 1`.
- **Skala nagrody.** Nagrody {+100, -100} vs {+1, -1} dają identyczne optymalne polityki, ale radykalnie różne wielkości gradientów. Normalizuj do przedziału około `[-1, 1]` przed użyciem w PPO/DQN.

## Użyj Tego

Stos z 2026 roku sprowadza każdy pipeline RL do MDP przed dotknięciem kodu:

| Sytuacja | Stan | Akcja | Nagroda | γ |
|-----------|-------|--------|--------|---|
| Sterowanie (lokomocja, manipulacja) | Kąty stawów + prędkości | Ciągłe momenty obrotowe | Specyficzne dla zadania | 0.99 |
| Gry (szachy, Go, poker) | Plansza + historia | Legalny ruch | Wygrana=+1 / przegrana=-1 | 1.0 (skończony) |
| Zapasy / wycena | Stan magazynu + popyt | Zamówienie | Przychód - koszt | 0.95 |
| RLHF dla LLM | Tokeny kontekstu | Następny token | Wynik modelu nagrody na końcu | 1.0 (epizod ~200 tokenów) |
| GRPO dla rozumowania | Prompt + częściowa odpowiedź | Następny token | Weryfikator 0/1 na końcu | 1.0 |

Napisz pięć krotek przed napisaniem jakiejkolwiek pętli treningowej. Większość raportów "RL nie działa" sprowadza się do sformułowania MDP, które było błędne na papierze.

## Dostarcz To

Zapisz jako `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj GridWorld 4×4 i wykonanie losowej polityki w `code/main.py`. Uruchom 10 000 epizodów. Podaj średnią i odchylenie standardowe zwrotu. Porównaj z optymalnym zwrotem (-6).
2. **Średnie.** Uruchom `policy_evaluation` z `γ ∈ {0.5, 0.9, 0.99}` dla jednostajnej losowej polityki. Wypisz `V` jako siatkę 4×4 dla każdego. Wyjaśnij, dlaczego wartości stanów w pobliżu terminala rosną szybciej dla większych `γ`.
3. **Trudne.** Uczyń GridWorld stochastycznym: każda akcja ześlizguje się w sąsiednim kierunku z prawdopodobieństwem `p = 0.1`. Ponownie oceń jednostajną politykę. Czy `V[start]` poprawia się, czy pogarsza? Dlaczego?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| MDP | "Ustawienie uczenia przez wzmacnianie" | Krotka `(S, A, P, R, γ)` spełniająca własność Markowa. |
| Stan | "Co agent widzi" | Wystarczająca statystyka dla przyszłej dynamiki przy danej klasie polityk. |
| Polityka | "Zachowanie agenta" | Rozkład warunkowy `π(a \| s)` lub deterministyczne odwzorowanie `s → a`. |
| Zwrot | "Całkowita nagroda" | Zdyskontowana suma `Σ γ^t r_t` od bieżącego kroku. |
| Wartość | "Jak dobry jest stan" | Oczekiwany zwrot przy `π` startując z `s`. |
| Wartość Q | "Jak dobra jest akcja" | Oczekiwany zwrot przy `π` startując z `s` z pierwszą akcją `a`. |
| Równanie Bellmana | "Rekurencja programowania dynamicznego" | Dekompozycja punktu stałego wartości/Q na nagrodę jednego kroku plus zdyskontowaną wartość następnika. |
| Dyskonto `γ` | "Przyszłość vs teraźniejszość" | Geometryczna waga na odległą przyszłą nagrodę; efektywny horyzont `~1/(1-γ)`. |

## Dalsza Lektura

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznik. Rozdz. 3 obejmuje MDP i równania Bellmana; Rozdz. 1 motywuje hipotezę nagrody leżącą u podstaw każdej kolejnej lekcji.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — źródło równania Bellmana.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — zwięzły wstęp do MDP z perspektywy głębokiego RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — referencja z badań operacyjnych na temat MDP i dokładnych metod rozwiązywania.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — najczystsze wyprowadzenie MDP jako specjalizacji programowania dynamicznego.