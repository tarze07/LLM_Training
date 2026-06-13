# Programowanie Dynamiczne — Iteracja Polityki i Iteracja Wartości

> Programowanie dynamiczne to RL z oszukiwaniem. Znasz już funkcje przejścia i nagrody; po prostu iterujesz równanie Bellmana, aż `V` lub `π` przestanie się zmieniać. Jest to benchmark, do którego każda metoda oparta na próbkowaniu stara się zbliżyć.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## Problem

Masz MDP ze znanym modelem: możesz zapytać `P(s' | s, a)` i `R(s, a, s')` dla dowolnej pary stan-akcja. Menedżer zapasów zna rozkład popytu. Gra planszowa ma deterministyczne przejścia. GridWorld to cztery linie Pythona. Masz *model*.

RL bez modelu (Q-learning, PPO, REINFORCE) został wymyślony na wypadek, gdy model nie jest dostępny — możesz jedynie próbkować ze środowiska. Ale gdy model masz, istnieją szybsze, lepsze metody: programowanie dynamiczne. Bellman zaprojektował je w 1957 roku. Wciąż definiują poprawność: gdy ludzie mówią "optymalna polityka dla tego MDP", mają na myśli politykę, którą zwróciłoby PD.

Potrzebujesz ich w 2026 z trzech powodów. Po pierwsze, każde środowisko tabelaryczne w badaniach RL (GridWorld, FrozenLake, CliffWalking) jest rozwiązywane za pomocą PD, aby uzyskać złotą politykę. Po drugie, dokładne wartości pozwalają *debugować* metody próbkowania: jeśli oszacowanie Q-learningu dla `V*(s_0)` różni się od odpowiedzi PD o 30%, twój Q-learning ma błąd. Po trzecie, nowoczesny RL offline i metody planowania (MCTS, wyszukiwanie AlphaZero, model-based RL w Fazie 9 · 10) wszystkie iterują krok wstecz Bellmana na wyuczonym lub danym modelu.

## Koncepcja

![Iteracja polityki i iteracja wartości, obok siebie](../assets/dp.svg)

**Dwa algorytmy, oba iteracje punktu stałego na Bellmanie.**

**Iteracja polityki.** Naprzemiennie wykonuje dwa kroki, aż polityka przestanie się zmieniać.

1. *Ewaluacja:* dla danej polityki `π`, oblicz `V^π`, wielokrotnie stosując `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` aż do zbieżności.
2. *Poprawa:* mając `V^π`, uczyń `π` zachłanną względem `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Zbieżność jest gwarantowana, ponieważ (a) każdy krok poprawy albo pozostawia `π` bez zmian, albo ściśle zwiększa `V^π` dla jakiegoś stanu, (b) przestrzeń deterministycznych polityk jest skończona. Zwykle zbiega się w ~5–20 zewnętrznych iteracji nawet dla dużych przestrzeni stanów.

**Iteracja wartości.** Łączy ewaluację i poprawę w jednym przebiegu. Zastosuj równanie Bellmana *optymalności*:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Powtarzaj, aż `max_s |V_{new}(s) - V(s)| < ε`. Wyodrębnij politykę na końcu, biorąc zachłanną akcję. Ściśle szybsza na iterację — brak wewnętrznej pętli ewaluacji — ale zazwyczaj potrzebuje więcej iteracji do zbieżności.

**Uogólniona iteracja polityki (GPI).** Jednoczące ujęcie. Funkcja wartości i polityka są zablokowane w dwukierunkowej pętli poprawy; każda metoda, która napędza obie w kierunku wzajemnej spójności (asynchroniczna iteracja wartości, zmodyfikowana iteracja polityki, Q-learning, aktor-krytyk, PPO) jest instancją GPI.

**Dlaczego `γ < 1` jest ważne.** Operator Bellmana jest `γ`-kontrakcją w normie supremum: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Kontrakcja implikuje unikalny punkt stały i geometryczną zbieżność. Usuń `γ < 1`, a stracisz gwarancję — potrzebujesz skończonego horyzontu lub absorbującego stanu terminalnego.

```figure
value-iteration-gamma
```

## Zbuduj To

### Krok 1: zbuduj model MDP GridWorld

Użyj tego samego GridWorld 4×4 z Lekcji 01. Dodajemy wariant stochastyczny: z prawdopodobieństwem `0.1` agent ześlizguje się w losowym prostopadłym kierunku.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` zwraca listę `(s', r, p)`. To cały model.

### Krok 2: ewaluacja polityki

Dla danej polityki `π(s) = {action: prob}`, iteruj równanie Bellmana, aż `V` przestanie się zmieniać:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Krok 3: poprawa polityki

Zastąp `π` zachłanną polityką względem `V`. Jeśli `π` się nie zmieniła, zwróć — jesteśmy w optimum.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Krok 4: połącz je razem

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitralny start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Typowa zbieżność na 4×4: 4–6 zewnętrznych iteracji. Wyniki: `V*(0,0) ≈ -6` i polityka ściśle zmniejszająca liczbę kroków.

### Krok 5: iteracja wartości (wersja z jedną pętlą)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Ten sam punkt stały, mniej linii kodu.

## Pułapki

- **Zapominanie o stanach terminalnych.** Jeśli zastosujesz Bellmana do stanu absorbującego, wciąż podnosi on "najlepszą akcję", która niczego nie zmienia. Zabezpiecz przez `if s == terminal: V[s] = 0`.
- **Zbieżność w normie supremum vs L2.** Używaj `max |V_new - V|`, nie średniej. Gwarancja teoretyczna dotyczy normy supremum.
- **Aktualizacje lokalne vs synchroniczne.** Aktualizowanie `V[s]` lokalnie (Gauss-Seidel) zbiega szybciej niż osobny słownik `V_new` (Jacobi). Kod produkcyjny używa lokalnych.
- **Remisy w polityce.** Jeśli dwie akcje mają równą wartość Q, `argmax` może rozstrzygać remisy inaczej w każdej iteracji, powodując oscylacje w sprawdzeniu "polityka stabilna". Użyj stabilnego rozstrzygania remisów (pierwsza akcja w ustalonej kolejności).
- **Eksplozja przestrzeni stanów.** PD to `O(|S| · |A|)` na przebieg. Działa do ~10⁷ stanów. Powyżej potrzebujesz aproksymacji funkcji (Faza 9 · 05 i dalej).

## Użyj Tego

W 2026 roku PD jest bazą poprawności i wewnętrzną pętlą planistów:

| Zastosowanie | Metoda |
|----------|--------|
| Rozwiązanie małego tabelarycznego MDP dokładnie | Iteracja wartości (prostsza) lub iteracja polityki (mniej kroków zewnętrznych) |
| Weryfikacja implementacji Q-learning / PPO | Porównaj z optymalnym DP-V* w prostym środowisku |
| RL oparty na modelu (Faza 9 · 10) | Krok wstecz Bellmana na wyuczonym modelu przejścia |
| Planowanie w AlphaZero / MuZero | Monte Carlo Tree Search = asynchroniczny krok wstecz Bellmana |
| RL offline (CQL, IQL) | Konserwatywna iteracja Q — PD z karą za działania OOD |

Za każdym razem, gdy ktoś mówi "optymalna funkcja wartości", ma na myśli "punkt stały PD". Gdy widzisz `V*` lub `Q*` w artykule, wyobraź sobie tę pętlę.

## Dostarcz To

Zapisz jako `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Ćwiczenia

1. **Łatwe.** Uruchom iterację wartości na GridWorld 4×4 z `γ ∈ {0.9, 0.99}`. Ile przebiegów do `max |ΔV| < 1e-6`? Wypisz `V*` jako siatkę 4×4.
2. **Średnie.** Porównaj iterację polityki vs iterację wartości na *stochastycznym* GridWorld (prawdopodobieństwo ześlizgu `0.1`). Policz: przebiegi, czas ścienny, końcowe `V*(0,0)`. Która zbiega szybciej w iteracjach? W czasie ściennym?
3. **Trudne.** Zbuduj zmodyfikowaną iterację polityki: w kroku ewaluacji wykonaj tylko `k` przebiegów zamiast do zbieżności. Narysuj wykres błędu `V*(0,0)` vs `k` dla `k ∈ {1, 2, 5, 10, 50}`. Co krzywa mówi o kompromisie między ewaluacją a poprawą?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Iteracja polityki | "Algorytm PD" | Naprzemienna ewaluacja (`V^π`) i poprawa (zachłanna `π` wzgl. `V^π`) aż polityka przestanie się zmieniać. |
| Iteracja wartości | "Szybsze PD" | Krok wstecz Bellmana optymalności w jednym przebiegu; zbiega geometrycznie do `V*`. |
| Operator Bellmana | "Rekurencja" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; `γ`-kontrakcja w normie supremum. |
| Kontrakcja | "Dlaczego PD zbiega" | Każdy operator `T` z `||T x - T y|| ≤ γ ||x - y||` ma unikalny punkt stały. |
| GPI | "Wszystko jest PD" | Uogólniona iteracja polityki: dowolna metoda prowadząca `V` i `π` do wzajemnej spójności. |
| Aktualizacja synchroniczna | "Styl Jacobiego" | Używaj starego `V` przez cały przebieg; czysta analitycznie, ale wolniejsza. |
| Aktualizacja lokalna | "Styl Gaussa-Seidela" | Używaj `V` podczas aktualizacji; zbiega szybciej w praktyce. |

## Dalsza Lektura

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — kanoniczna prezentacja iteracji polityki i iteracji wartości.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — rygorystyczne traktowanie argumentów kontrakcji.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — zmodyfikowana iteracja polityki i jej analiza zbieżności.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — oryginalna praca o iteracji polityki.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — most od PD do przybliżonego PD / głębokiego RL używany przez każdą kolejną lekcję.