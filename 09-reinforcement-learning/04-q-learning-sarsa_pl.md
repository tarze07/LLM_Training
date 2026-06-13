# Różnica Temporalna — Q-learning i SARSA

> Monte Carlo czeka, aż epizod się skończy. TD aktualizuje po każdym kroku, uruchamiając się na kolejnym oszacowaniu wartości. Q-learning jest poza polityką i optymistyczny; SARSA jest w ramach polityki i ostrożna. Oba to jedna linia kodu. Oba stanowią podstawę każdej metody głębokiego RL w tej fazie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## Problem

Monte Carlo działa, ale ma dwa kosztowne wymagania. Potrzebuje epizodów, które się kończą, i aktualizuje dopiero po uzyskaniu końcowego zwrotu. Jeśli twój epizod ma 1000 kroków, MC czeka 1000 kroków, aby cokolwiek zaktualizować. Ma wysoką wariancję, niskie obciążenie i jest wolne w praktyce.

Programowanie dynamiczne ma przeciwny profil — kroki wsteczne o zerowej wariancji — ale wymaga znanego modelu.

Uczenie przez różnicę temporalną (TD) dzieli różnicę. Z pojedynczej tranzycji `(s, a, r, s')` utwórz jednokrokowy cel `r + γ V(s')` i popraw `V(s)` w jego kierunku. Bez modelu. Bez kompletnych epizodów. Obciążenie z używania przybliżonego `V` po prawej stronie, ale drastycznie niższa wariancja niż MC i aktualizacje online od pierwszego kroku.

To jest punkt zwrotny, na którym opiera się cały nowoczesny RL — DQN, A2C, PPO, SAC. Reszta Fazy 9 to warstwy aproksymacji funkcji i sztuczek zbudowanych na jednokrokowej aktualizacji TD, którą napiszesz w tej lekcji.

## Koncepcja

![Q-learning vs SARSA: max poza polityką vs Q(s', a') w ramach polityki](../assets/td.svg)

**Aktualizacja TD(0) dla V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Wyrażenie w nawiasie to błąd TD `δ = r + γ V(s') - V(s)`. Jest to online'owy odpowiednik `G_t - V(s_t)` w MC. Zbieżność wymaga `α` spełniającego Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) i wszystkich stanów odwiedzanych nieskończenie często.

**Q-learning.** Metoda TD poza polityką do sterowania:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` zakłada, że *zachłanna* polityka będzie stosowana od `s'` w przód, niezależnie od tego, jaką akcję agent faktycznie podejmie. To rozprzężenie sprawia, że Q-learning uczy się `Q*`, podczas gdy agent eksploruje przez ε-zachłanność. Mnih i in. (2015) przekształcili to w głęboki Q-learning na Atari (Lekcja 05).

**SARSA.** Metoda TD w ramach polityki:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Nazwa pochodzi od krotki `(s, a, r, s', a')`. SARSA używa akcji `a'`, którą agent *faktycznie* wykonuje następnie, a nie zachłannego `argmax`. Zbiega do `Q^π` dla dowolnej ε-zachłannej `π`, która w granicy `ε → 0` staje się `Q*`.

**Różnica w chodzeniu po klifie.** W klasycznym zadaniu chodzenia po klifie (spadek z klifu = nagroda -100), Q-learning uczy się optymalnej ścieżki wzdłuż krawędzi klifu, ale czasami ponosi karę podczas eksploracji. SARSA uczy się bezpieczniejszej ścieżki o krok od klifu, ponieważ uwzględnia szum eksploracyjny w swojej wartości Q. Po wytrenowaniu oba osiągają optimum przy `ε → 0`. W praktyce ma to znaczenie: gdy eksploracja faktycznie zachodzi podczas wdrożenia, zachowanie SARSA jest bardziej konserwatywne.

**Oczekiwana SARSA.** Zastąp `Q(s', a')` jego wartością oczekiwaną przy `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Niższa wariancja niż SARSA (brak próbki `a'`), ten sam cel w ramach polityki. Często domyślna w nowoczesnych podręcznikach.

**n-krokowe TD i TD(λ).** Interpoluj między TD(0) a MC, czekając `n` kroków przed uruchomieniem. `n=1` to TD, `n=∞` to MC. TD(λ) uśrednia po wszystkich `n` z geometrycznymi wagami `(1-λ)λ^{n-1}`. Większość głębokiego RL używa `n` między 3 a 20.

```figure
qlearning-gridworld
```

## Zbuduj To

### Krok 1: SARSA na ε-zachłannej polityce

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Osiem linii. *Jedyna* różnica od Q-learningu to linia celu.

### Krok 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` odłącza cel od zachowania. Ten jeden symbol to różnica między w ramach polityki a poza polityką.

### Krok 3: krzywe uczenia

Śledź średni zwrot na 100 epizodów. Q-learning zbiega szybciej na prostym deterministycznym GridWorld; SARSA jest bardziej konserwatywna na chodzeniu po klifie. Na GridWorld 4×4 w `code/main.py` oba są blisko optymalne po ~2000 epizodach z `α=0.1, ε=0.1`.

### Krok 4: porównaj z prawdą PD

Uruchom iterację wartości (Lekcja 02), aby uzyskać `Q*`. Sprawdź `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Zdrowy tabelaryczny agent TD osiąga wynik w granicach `~0.5` na GridWorld 4×4 po 10 000 epizodach.

## Pułapki

- **Początkowe wartości Q mają znaczenie.** Optymistyczna inicjacja (`Q = 0` dla zadania z ujemną nagrodą) zachęca do eksploracji. Pesymistyczna inicjacja może uwięzić zachłanną politykę na zawsze.
- **Harmonogram α.** Stałe `α` jest w porządku dla problemów niestacjonarnych. Zanikające `α_n = 1/n` daje zbieżność w teorii, ale jest zbyt wolne w praktyce — ustaw `α` w `[0.05, 0.3]` i monitoruj krzywą uczenia.
- **Harmonogram ε.** Zacznij wysoko (`ε=1.0`), zmniejszaj do `ε=0.05`. "GLIE" (zachłanne w granicy z nieskończoną eksploracją) to warunek zbieżności.
- **Błąd maksimum w Q-learningu.** Operator `max` jest obciążony w górę, gdy `Q` jest zaszumione. Prowadzi do przeszacowania — podwójny Q-learning Hasselta (używany przez DDQN w Lekcji 05) naprawia to za pomocą dwóch tablic Q.
- **Niekończące się epizody.** TD może uczyć się bez terminali, ale musisz albo ograniczyć kroki, albo poprawnie obsłużyć uruchomienie przy ograniczeniu. Standard: traktuj ograniczenie jako nieterminalne, kontynuuj uruchamianie.
- **Haszowanie stanów.** Jeśli stany są krotkami/tensorami, użyj haszowalnego klucza (krotka, nie lista; krotka zaokrąglonych liczb zmiennoprzecinkowych, nie surowych).

## Użyj Tego

Krajobraz TD w 2026:

| Zadanie | Metoda | Powód |
|------|--------|--------|
| Małe środowiska tabelaryczne | Q-learning | Uczy się bezpośrednio optymalnej polityki. |
| Bezpieczeństwo w ramach polityki | SARSA / Oczekiwana SARSA | Konserwatywna podczas eksploracji. |
| Wysoko-wymiarowy stan | DQN (Faza 9 · 05) | Neuronowa funkcja Q z replay i siecią docelową. |
| Ciągłe akcje | SAC / TD3 (Faza 9 · 07) | Aktualizacja TD na sieci Q; sieć polityki emituje akcje. |
| RL dla LLM (oparty na modelu nagrody) | PPO / GRPO (Faza 9 · 08, 12) | Aktor-krytyk z przewagą w stylu TD przez GAE. |
| RL offline | CQL / IQL (Faza 9 · 08) | Q-learning z konserwatywną regularyzacją. |

Dziewięćdziesiąt procent "RL", o którym czytasz w artykułach z 2026, to jakaś elaboracja Q-learningu lub SARSA. Zrozum tabelaryczną aktualizację na wyczucie, zanim sięgniesz głębiej.

## Dostarcz To

Zapisz jako `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj Q-learning i SARSA na GridWorld 4×4. Narysuj krzywe uczenia (średni zwrot na 100 epizodów) dla 2000 epizodów. Kto zbiega szybciej?
2. **Średnie.** Zbuduj środowisko chodzenia po klifie (4×12, ostatni rząd to klif z nagrodą -100 i resetem do startu). Porównaj końcowe polityki Q-learningu i SARSA. Zrzut ekranu ścieżek, które każda wybiera. Która jest bliżej klifu?
3. **Trudne.** Zaimplementuj podwójny Q-learning. Na GridWorld z zaszumioną nagrodą (szum Gaussa σ=5 dodany do nagrody za krok), pokaż, że Q-learning przeszacowuje `V*(0,0)` o znaczącą wartość, podczas gdy podwójny Q-learning nie.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Błąd TD | "Sygnał aktualizacji" | `δ = r + γ V(s') - V(s)`, uruchomiona reszta. |
| TD(0) | "Jednokrokowe TD" | Aktualizacja po każdej tranzycji używająca tylko oszacowania następnego stanu. |
| Q-learning | "RL poza polityką 101" | Aktualizacja TD z `max` po akcjach następnego stanu; uczy się `Q*` niezależnie od polityki behawioralnej. |
| SARSA | "Q-learning w ramach polityki" | Aktualizacja TD używająca faktycznej następnej akcji; uczy się `Q^π` dla bieżącej ε-zachłannej π. |
| Oczekiwana SARSA | "SARSA o niskiej wariancji" | Zastąp próbkowane `a'` jego wartością oczekiwaną przy π. |
| GLIE | "Poprawny harmonogram eksploracji" | Zachłanne w granicy z nieskończoną eksploracją; potrzebne do zbieżności Q-learningu. |
| Uruchamianie | "Używanie bieżącego oszacowania w celu" | To, co odróżnia TD od MC. Źródło obciążenia, ale ogromna redukcja wariancji. |
| Błąd maksymalizacji | "Q-learning przeszacowuje" | `max` nad zaszumionymi estymatami jest obciążony w górę; naprawiony przez podwójny Q-learning. |

## Dalsza Lektura

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — oryginalna praca i dowód zbieżności.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, oczekiwana SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — poprawka na błąd maksymalizacji.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — motywacja oczekiwanej SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — praca, która ukuła SARSA (wtedy nazwana "zmodyfikowanym connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — uogólnia TD(0) do TD(n), ścieżka od Q-learningu do śladów kwalifikowalności, a później GAE w PPO.