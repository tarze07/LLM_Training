# Głębokie Sieci Q (DQN)

> 2013: Mnih wytrenował jedną sieć Q-learningu na surowych pikselach, pokonał każdy klasyczny agent RL na siedmiu grach Atari. 2015: rozszerzone do 49 gier, opublikowane w Nature, zapoczątkowało erę głębokiego RL. DQN to Q-learning plus trzy sztuczki, które czynią aproksymację funkcji stabilną.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## Problem

Tabelaryczny Q-learning potrzebuje osobnej wartości Q dla każdej pary (stan, akcja). Szachownica ma ~10⁴³ stanów. Ramka Atari to 210×160×3 = 100 800 cech. Tabelaryczny RL umiera przy tysiącach stanów, a co dopiero miliardach.

Poprawka jest oczywista z perspektywy czasu: zastąp tablicę Q siecią neuronową, `Q(s, a; θ)`. Ale oczywiste-z-perspektywy-czasu zajęło dziesięciolecia. Naiwna aproksymacja funkcji z Q-learningiem rozbiega się w ramach "śmiertelnej triady" — aproksymacja funkcji + uruchamianie + uczenie poza polityką. Mnih i in. (2013, 2015) zidentyfikowali trzy inżynieryjne sztuczki, które stabilizują uczenie:

1. **Bufor doświadczeń** dekoreluje tranzycje.
2. **Sieć docelowa** zamraża cel uruchamiania.
3. **Przycinanie nagród** normalizuje wielkości gradientów.

DQN na Atari było pierwszym przypadkiem, gdy pojedyncza architektura z jednym zestawem hiperparametrów rozwiązała dziesiątki problemów sterowania z surowych pikseli. Wszystko, co zbudowano od tego czasu w "głębokim RL" — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — jest ułożone na tej trzysztuczkowej podstawie.

## Koncepcja

![Pętla treningowa DQN: środowisko, bufor replay, sieć online, sieć docelowa, strata TD Bellmana](../assets/dqn.svg)

**Cel.** DQN minimalizuje jednokrokową stratę TD na neuronowej funkcji Q:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = sieć online, aktualizowana co krok przez gradient prosty. `θ^-` = sieć docelowa, okresowo kopiowana z `θ` (co ~10 000 kroków). `D` = bufor replay przeszłych tranzycji.

**Trzy sztuczki, w kolejności ważności:**

**Bufor doświadczeń.** Bufor cykliczny o pojemności `~10⁶` tranzycji. Każdy krok treningowy próbkuje minipartię jednostajnie losowo. To przełamuje korelację czasową (kolejne ramki są prawie identyczne), pozwala sieci uczyć się z rzadkich nagradzających tranzycji wielokrotnie i dekoreluje kolejne aktualizacje gradientowe. Bez niego, TD w ramach polityki z siecią neuronową rozbiega się na Atari.

**Sieć docelowa.** Używanie tej samej sieci `Q(·; θ)` po obu stronach równania Bellmana sprawia, że cel przesuwa się przy każdej aktualizacji — "gonienie własnego ogona". Poprawka: utrzymuj drugą sieć `Q(·; θ^-)` z zamrożonymi wagami. Co `C` kroków, kopiuj `θ → θ^-`. To stabilizuje cel regresji na tysiące kroków gradientowych naraz. Miękkie aktualizacje `θ^- ← τ θ + (1-τ) θ^-` (używane w DDPG, SAC) są gładszym wariantem.

**Przycinanie nagród.** Wielkości nagród w Atari wahają się od 1 do 1000+. Przycięcie do `{-1, 0, +1}` zapobiega dominacji gradientu przez pojedynczą grę. Błędne, gdy wielkość nagrody ma znaczenie; w porządku dla Atari, gdzie liczy się tylko znak.

**Podwójne DQN.** Hasselt (2016) naprawia błąd maksymalizacji: użyj sieci online do *wyboru* akcji, sieci docelowej do *oceny* jej.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Wymienna poprawka, konsekwentnie lepsza. Używaj domyślnie.

**Inne ulepszenia (Rainbow, 2017):** priorytetowy replay (próbkuj tranzycje z wysokim błędem TD częściej), architektura duelingowa (oddzielne głowy `V(s)` i przewagi), sieci zaszumione (wyuczona eksploracja), zwroty n-krokowe, rozkładowy Q (C51/QR-DQN), uruchamianie wielokrokowe. Każde dodaje kilka procent; zyski są w przybliżeniu addytywne.

## Zbuduj To

Kod tutaj jest stdlib-only bez numpy — używamy ręcznie zrobionego MLP z jedną warstwą ukrytą na małym ciągłym GridWorld, więc każdy krok treningowy działa w mikrosekundach. Algorytm jest identyczny z DQN na Atari w skali.

### Krok 1: bufer replay

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50 000 pojemności dla Atari; 5000 wystarcza dla naszego zabawkowego środowiska.

### Krok 2: mała sieć Q (ręczny MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Przebieg w przód: liniowa → ReLU → liniowa. To cała sieć.

### Krok 3: aktualizacja DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Kształt jest taki sam jak Q-learning z Lekcji 04 z dwiema różnicami: (a) propagujemy wstecz przez różniczkowalny `Q(·; θ)` zamiast indeksować tablicę, (b) cel używa `Q(·; θ^-)`.

### Krok 4: pętla zewnętrzna

Dla każdego epizodu, działaj ε-zachłannie na `Q(·; θ)`, wypychaj tranzycje do bufora, próbkuj minipartię, wykonaj krok gradientowy, okresowo synchronizuj `θ^- ← θ`. Wzorzec:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Na naszym małym GridWorld z 16-wymiarowym stanem one-hot, agent uczy się blisko optymalnej polityki w ~500 epizodach. Na Atari, skalować to do 200M ramek i dodać ekstraktor cech CNN.

## Pułapki

- **Śmiertelna triada.** Aproksymacja funkcji + poza polityką + uruchamianie mogą się rozejść. DQN łagodzi to przez sieć docelową + replay; nie usuwaj żadnego.
- **Eksploracja.** ε musi zanikać, typowo od 1.0 do 0.01 w ciągu pierwszych ~10% treningu. Bez wystarczającej wczesnej eksploracji sieć Q zbiega do lokalnego basenu.
- **Przeszacowanie.** `max` nad zaszumionym Q jest obciążony w górę. Zawsze używaj podwójnego DQN w produkcji.
- **Skala nagrody.** Przytnij lub normalizuj nagrody; wielkość gradientu jest proporcjonalna do wielkości nagrody.
- **Zimny start bufora replay.** Nie trenuj, dopóki bufor nie ma kilku tysięcy tranzycji. Wczesne gradienty na ~20 próbkach przeuczają się.
- **Częstotliwość synchronizacji docelowej.** Zbyt częsta ≈ brak sieci docelowej; zbyt rzadka ≈ nieaktualne cele. Atari DQN używa 10 000 kroków środowiska. Zasada kciuka: synchronizuj co ~1/100 horyzontu treningowego.
- **Przetwarzanie wstępne obserwacji.** Atari DQN układa 4 ramki, aby stan był Markowski. Każde środowisko z informacją o prędkości potrzebuje układania ramek lub stanu rekurencyjnego.

## Użyj Tego

W 2026 DQN rzadko jest najnowocześniejszy, ale pozostaje referencyjnym algorytmem poza polityką:

| Zadanie | Wybrana metoda | Dlaczego nie DQN? |
|------|------------------|--------------|
| Dyskretne akcje Atari-podobne | Rainbow DQN lub Muesli | Te same ramy, więcej sztuczek. |
| Sterowanie ciągłe | SAC / TD3 (Faza 9 · 07) | DQN nie ma sieci polityki. |
| W ramach polityki / wysoka przepustowość | PPO (Faza 9 · 08) | Brak bufora replay; łatwiejsze do skalowania. |
| RL offline | CQL / IQL / Decision Transformer | Konserwatywne cele Q, brak eksplozji uruchamiania. |
| Duże dyskretne przestrzenie akcji (rekomendacje) | DQN z osadzaniem akcji, lub IMPALA | W porządku; dekoracja ma znaczenie. |
| RL dla LLM | PPO / GRPO | Na poziomie sekwencji, nie kroku; inna strata. |

Lekcje wciąż podróżują. Bufor replay i sieci docelowe pojawiają się w SAC, TD3, DDPG, SAC-X, buforze samodzielnej gry AlphaZero i każdej metodzie RL offline. Przycinanie nagród żyje dalej jako normalizacja przewagi w PPO. Architektura jest planem.

## Dostarcz To

Zapisz jako `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Narysuj krzywą zwrotu na epizod. Ile epizodów, zanim średnia krocząca przekroczy -10?
2. **Średnie.** Wyłącz sieć docelową (użyj sieci online dla obu stron celu Bellmana). Zmierz niestabilność treningu — czy zwrot oscyluje, czy się rozbiega?
3. **Trudne.** Dodaj podwójne DQN: użyj sieci online do wyboru `argmax a'`, sieci docelowej do oceny. Porównaj błąd `Q(s_0, best_a)` vs prawdziwe `V*(s_0)` po 1000 epizodach z i bez podwójnego DQN na GridWorld z zaszumioną nagrodą.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| DQN | "Głęboki Q-learning" | Q-learning z neuronową funkcją Q, buforem replay i siecią docelową. |
| Bufor doświadczeń | "Przetasowane tranzycje" | Bufor cykliczny próbkowany jednostajnie na każdy krok gradientowy; dekoreluje dane. |
| Sieć docelowa | "Zamrożone uruchamianie" | Okresowa kopia Q używana w celu Bellmana; stabilizuje trening. |
| Śmiertelna triada | "Dlaczego RL się rozbiega" | Aproksymacja funkcji + uruchamianie + poza polityką = brak gwarancji zbieżności. |
| Podwójne DQN | "Poprawka na błąd maksymalizacji" | Sieć online wybiera akcję, sieć docelowa ją ocenia. |
| Dueling DQN | "Głowy V i A" | Dekompozycja Q = V + A - mean(A); ten sam wynik, lepszy przepływ gradientu. |
| Rainbow | "Wszystkie sztuczki" | DDQN + PER + dueling + n-krok + szum + rozkładowy w jednym. |
| PER | "Priorytetowy replay" | Próbkuj tranzycje proporcjonalnie do wielkości błędu TD. |

## Dalsza Lektura

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — praca z warsztatów NeurIPS 2013, która zapoczątkowała głęboki RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — praca w Nature, DQN na 49 grach.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — praca o ułożonych sztuczkach.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — jasny nowoczesny wykład.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznikowe traktowanie "śmiertelnej triady" (aproksymacja funkcji + uruchamianie + poza polityką), którą sieć docelowa i bufor replay DQN mają okiełznać.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — referencyjny jednowątkowy DQN używany w badaniach ablacyjnych; dobrze przeczytać obok tej lekcji od zera.