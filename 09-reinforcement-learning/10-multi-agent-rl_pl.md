# Wieloagentowy RL

> Jednoagentowy RL zakłada, że środowisko jest stacjonarne. Umieść dwóch uczących się agentów w tym samym świecie, a to założenie się załamuje: każdy agent jest częścią środowiska drugiego i obaj się zmieniają. Wieloagentowy RL to zestaw sztuczek, które sprawiają, że uczenie zbiega, gdy założenie Markowa już nie obowiązuje.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## Problem

Robot uczący się nawigować po pokoju to jednoagentowy problem RL. Drużyna piłkarska — nie. AlphaStar przeciwko przeciwnikom w StarCraft — nie. Rynek agentów licytujących — nie. Dwa samochody negocjujące skrzyżowanie z czterema stopami — nie. Problemy rzeczywistego świata z wieloma agentami nie są jednoagentowe.

W każdym ustawieniu wieloagentowym, z perspektywy dowolnego pojedynczego agenta, inni agenci *są* częścią środowiska. Gdy się uczą i zmieniają swoje zachowanie, środowisko staje się niestacjonarne. Własność Markowa — "następny stan zależy tylko od bieżącego stanu i mojej akcji" — zostaje naruszona, ponieważ następny stan zależy również od tego, co wybrali *inni* agenci, a ich polityki są ruchomymi celami.

To załamuje tabelaryczne dowody zbieżności (gwarancja Q-learningu zakłada stacjonarne środowisko). Załamuje też naiwny głęboki RL: agenci gonią się w pętlach, nigdy nie zbiegając do stabilnej polityki. Potrzebujesz technik specyficznych dla wieloagentowego RL: scentralizowane trenowanie / zdecentralizowane wykonywanie, kontrfaktyczne wartości bazowe, gra ligowa, samodzielna gra.

Zastosowania w 2026: roje robotów, routing ruchu, floty pojazdów autonomicznych, symulatory rynku, wieloagentowe systemy LLM (Faza 16) i każda gra z więcej niż jednym inteligentnym graczem.

## Koncepcja

![Cztery reżimy MARL: niezależny, scentralizowany krytyk, samodzielna gra, liga](../assets/marl.svg)

**Formalizm: Gra Markowa.** Uogólnienie MDP: stany `S`, wspólna akcja `a = (a_1, …, a_n)`, przejście `P(s' | s, a)` i nagrody na agenta `R_i(s, a, s')`. Każdy agent `i` maksymalizuje swój własny zwrot przy swojej własnej polityce `π_i`. Jeśli nagrody są identyczne, jest to **w pełni kooperacyjne**. Jeśli o sumie zerowej, jest to **adwersarialne**. Jeśli mieszane, jest to **o sumie ogólnej**.

**Główne wyzwania:**

- **Niestacjonarność.** `P(s' | s, a_i)` z perspektywy agenta `i` zależy od `π_{-i}`, która się zmienia.
- **Przypisanie odpowiedzialności.** Przy wspólnej nagrodzie, który agent ją spowodował?
- **Koordynacja eksploracji.** Agenci muszą eksplorować komplementarne strategie, nie redundacyjnie eksplorować ten sam stan.
- **Skalowalność.** Wspólna przestrzeń akcji rośnie wykładniczo z `n`.
- **Częściowa obserwowalność.** Każdy agent widzi tylko własną obserwację; globalny stan jest ukryty.

**Cztery dominujące reżimy:**

**1. Niezależny Q-learning / niezależne PPO (IQL, IPPO).** Każdy agent uczy się własnego Q lub polityki, traktując innych jako część środowiska. Proste, czasami działa (szczególnie gdy bufor doświadczeń działa jako wygładzająca sztuczka modelowania agenta). Teoretyczna zbieżność: żadna. W praktyce: w porządku dla luźno sprzężonych zadań, złe dla ściśle sprzężonych.

**2. Scentralizowane trenowanie, zdecentralizowane wykonywanie (CTDE).** Najpopularniejszy nowoczesny paradygmat. Każdy agent ma własną *politykę* `π_i`, która warunkuje na lokalnej obserwacji `o_i` — standardowe zdecentralizowane wykonywanie przy wdrożeniu. Podczas *trenowania*, scentralizowany krytyk `Q(s, a_1, …, a_n)` warunkuje na pełnym globalnym stanie i wspólnej akcji. Przykłady:
- **MADDPG** (Lowe i in. 2017): DDPG ze scentralizowanym krytykiem na agenta.
- **COMA** (Foerster i in. 2017): kontrfaktyczna wartość bazowa — zapytaj "jaka byłaby moja nagroda, gdybym wykonał akcję `a'` zamiast?" — izoluje mój wkład.
- **MAPPO** / **IPPO** ze współdzielonym krytykiem (Yu i in. 2022): PPO ze scentralizowaną funkcją wartości. Dominujące w 2026 dla kooperacyjnego MARL.
- **QMIX** (Rashid i in. 2018): dekompozycja wartości — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` z monotonicznym mieszaniem.

**3. Samodzielna gra.** Dwie kopie tego samego agenta grają przeciwko sobie. Polityka przeciwnika *jest* moją polityką z przeszłej migawki. AlphaGo / AlphaZero / MuZero. OpenAI Five. Działa najlepiej dla gier o sumie zerowej; sygnał treningowy jest symetryczny.

**4. Gra ligowa.** Rozszerzenie samodzielnej gry na środowiska o sumie ogólnej / adwersarialne: utrzymuj populację przeszłych i obecnych polityk, losuj przeciwnika z ligi, trenuj przeciwko niemu. Dodaje eksploatatorów (specjalizujących się w pokonywaniu najlepszej bieżącej polityki) i głównych eksploatatorów (specjalizujących się w pokonywaniu eksploatatorów). AlphaStar (StarCraft II). Potrzebne, gdy gra dopuszcza cykle strategii "kamień-papier-nożyce."

**Komunikacja.** Pozwól agentom wysyłać wyuczone wiadomości `m_i` do siebie nawzajem. Działa w ustawieniach kooperacyjnych. Foerster i in. (2016) pokazali, że różniczkowalna komunikacja między agentami może być trenowana end-to-end. Dzisiejsze wieloagentowe systemy oparte na LLM (Faza 16) komunikują się zasadniczo w języku naturalnym.

## Zbuduj To

Ta lekcja używa GridWorld 6×6 z dwoma kooperacyjnymi agentami. Zaczynają w przeciwnych rogach i muszą dotrzeć do wspólnego celu. Wspólna nagroda: `-1` za krok, gdy któryś agent wciąż się porusza, `+10`, gdy obaj dotrą. Zobacz `code/main.py`.

### Krok 1: środowisko wieloagentowe

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Wspólna przestrzeń akcji to `|A|² = 16`. Globalny stan to dwie pozycje.

### Krok 2: niezależny Q-learning

Każdy agent prowadzi własną tablicę Q kluczowaną na wspólnym stanie. Na każdym kroku: obaj wybierają ε-zachłanne akcje, zbierają wspólną tranzycję, każdy aktualizuje własne Q ze wspólną nagrodą.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Działa w tym zadaniu, ponieważ nagrody są gęste i współliniowe. Załamuje się na ściśle sprzężonych zadaniach (np. gdy jeden agent musi *czekać* na drugiego).

### Krok 3: Scentralizowane Q z aktualizacją zdekomponowanej wartości

Użyj jednego Q nad wspólnymi akcjami `Q(s, a_1, a_2)`. Aktualizuj ze wspólnej nagrody. Decentralizuj przy wykonywaniu przez marginalizację: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Wymienia wykładniczą wspólną przestrzeń akcji na *poprawny* globalny widok.

### Krok 4: Prosta samodzielna gra (adwersarialna 2-agentowa)

Ten sam agent, dwie role. Trenuj agenta A przeciwko agentowi B; po `K` epizodach, skopiuj wagi A do B. Symetryczne trenowanie, spójny postęp. Przepis AlphaZero w miniaturze.

## Pułapki

- **Niestacjonarny replay.** Bufor doświadczeń z niezależnymi agentami jest gorszy niż jednoagentowy, ponieważ stare tranzycje zostały wygenerowane przez już nieaktualnych przeciwników. Poprawka: przeetykietuj lub waż ze względu na świeżość.
- **Niejednoznaczność przypisania odpowiedzialności.** Wspólna nagroda po długim epizodzie; brak jasnego sposobu, by stwierdzić, który agent się przyczynił. Poprawka: kontrfaktyczne wartości bazowe (COMA) lub kształtowanie nagrody na agenta.
- **Dryf polityki / gonienie.** Najlepsza odpowiedź każdego agenta zmienia się z każdą aktualizacją innego. Poprawka: scentralizowany krytyk, wolne współczynniki uczenia lub zamrażanie jednego na raz.
- **Hakowanie nagrody przez koordynację.** Agenci znajdują skoordynowane exploity, których projektant nie przewidział. Agenci aukcyjni zbiegają do oferowania zera. Poprawka: staranny projekt nagrody, ograniczenia behawioralne.
- **Redundancja eksploracji.** Obaj agenci eksplorują te same pary stan-akcja. Poprawka: bonusy entropii na agenta lub warunkowanie na rolę.
- **Cykle ligowe.** Czysta samodzielna gra może utknąć w cyklu dominacji. Poprawka: gra ligowa z różnorodnymi przeciwnikami.
- **Eksplozja próbek.** `n` agentów × przestrzeń stanów × wspólne akcje. Aproksymuj funkcją; sfaktoryzowane przestrzenie akcji (jedna głowa wyjściowa polityki na agenta).

## Użyj Tego

Mapa zastosowań MARL w 2026:

| Domena | Metoda | Uwagi |
|--------|--------|-------|
| Kooperacyjna nawigacja / manipulacja | MAPPO / QMIX | CTDE; współdzielony krytyk + zdecentralizowani aktorzy. |
| Gry dwuosobowe (szachy, Go, poker) | Samodzielna gra z MCTS (AlphaZero) | O sumie zerowej; symetryczne trenowanie. |
| Złożone gry wieloosobowe (Dota, StarCraft) | Gra ligowa + wstępny trening imitacyjny | OpenAI Five, AlphaStar. |
| Floty pojazdów autonomicznych | CTDE MAPPO / PPO z uwagą | Częściowa obserwacja; zmienne rozmiary drużyn. |
| Rynki aukcyjne | Równowaga teorii gier + RL | Mean-field RL, gdy `n` → ∞. |
| Wieloagentowe systemy LLM (Faza 16) | Komunikacja w języku naturalnym + warunkowanie na rolę | Pętla RL na warstwie planowania agenta. |

W 2026, największym obszarem wzrostu MARL są systemy oparte na LLM: roje agentów modeli językowych negocjujących, debatujących, budujących oprogramowanie. RL pojawia się jako optymalizacja preferencji na wynikach na poziomie *trajektorii*, nie tokena (Faza 16 · 03).

## Dostarcz To

Zapisz jako `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Ćwiczenia

1. **Łatwe.** Trenuj niezależny Q-learning na 2-agentowym kooperacyjnym GridWorld. Ile epizodów, zanim średni zwrot > 0? Narysuj wspólną krzywą uczenia.
2. **Średnie.** Dodaj zadanie "koordynacji": cel jest osiągnięty tylko wtedy, gdy obaj agenci wejdą na niego w tej samej turze. Czy niezależny Q wciąż zbiega? Co się załamuje?
3. **Trudne.** Zaimplementuj scentralizowanego krytyka dla trenowania w stylu MAPPO i porównaj prędkość zbieżności z niezależnym PPO na zadaniu koordynacji.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Gra Markowa | "Wieloagentowy MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; każdy agent ma własną nagrodę. |
| CTDE | "Scentralizowane trenowanie, zdecentralizowane wykonywanie" | Wspólny krytyk w czasie treningu; polityka każdego agenta używa tylko lokalnych obserwacji. |
| IPPO | "Niezależne PPO" | Każdy agent uruchamia PPO osobno. Prosta wartość bazowa; często niedoceniane. |
| MAPPO | "Wieloagentowe PPO" | PPO ze scentralizowaną funkcją wartości warunkującą na globalnym stanie. |
| QMIX | "Monotoniczna dekompozycja wartości" | `Q_tot = f_monotone(Q_1, …, Q_n)` umożliwia zdecentralizowany argmax. |
| COMA | "Kontrfaktyczny wieloagentowy" | Przewaga = moje Q minus oczekiwane Q marginalizujące po mojej akcji. |
| Samodzielna gra | "Agent przeciw przeszłemu sobie" | Pojedynczy agent, dwie role; standard dla gier o sumie zerowej. |
| Gra ligowa | "Trenowanie populacyjne" | Przechowuj przeszłe polityki, próbkuj przeciwników z puli; obsługuje cykle strategii. |

## Dalsza Lektura

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE ze scentralizowanym krytykiem.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — kontrfaktyczne wartości bazowe do przypisania odpowiedzialności.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — dekompozycja wartości z monotonicznością.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO jest zaskakująco mocny dla MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — gra ligowa w skali.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — czysta samodzielna gra w grach o sumie zerowej.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — zawiera podręcznikowe krótkie opracowanie ustawień wieloagentowych i problemu niestacjonarności, który CTDE ma rozwiązać.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — przegląd obejmujący kooperacyjny, konkurencyjny i mieszany MARL z wynikami zbieżności.