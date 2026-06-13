# RL dla Gier — AlphaZero, MuZero i Era Rozumowania LLM

> 1992: TD-Gammon pokonał ludzkich mistrzów w backgammona czystym TD. 2016: AlphaGo pokonał Lee Sedola. 2017: AlphaZero zdominował szachy, shogi i Go od zera. 2024: DeepSeek-R1 udowodnił, że ten sam przepis, z GRPO zastępującym PPO, działa na rozumowanie. Gry są benchmarkiem, który napędza każdy przełom w tej fazie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## Problem

Gry mają wszystko, czego RL chce. Czystą nagrodę (wygrana/przegrana). Nieskończone epizody (samodzielna gra się resetuje). Doskonałą symulację (gra *jest* symulatorem). Dyskretne lub małe ciągłe przestrzenie akcji. Wieloagentową strukturę, która wymusza adwersarialną odporność.

A gry są tym, na czym testowano każdy większy przełom RL. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (wyuczony model, 2019). AlphaTensor (mnożenie macierzy, 2022). AlphaDev (algorytmy sortowania, 2023). DeepSeek-R1 (rozumowanie matematyczne, 2025) — najnowsza demonstracja, że techniki RL z gier działają na tekście.

To podsumowanie przegląda trzy przełomowe architektury — AlphaZero, MuZero i GRPO — przez jedną jednoczącą soczewkę: **samodzielna gra + wyszukiwanie + poprawa polityki**. Każda uogólnia poprzednią; GRPO w szczególności to przepis AlphaZero zastosowany do rozumowania LLM, z tokenami jako akcjami i weryfikacją matematyczną jako sygnałem wygranej.

## Koncepcja

![AlphaZero ↔ MuZero ↔ GRPO: ta sama pętla, różne środowiska](../assets/rl-games.svg)

**Jednocząca pętla.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver i in. Dla gry (szachy, shogi, Go) ze znanymi regułami:

- Sieć polityka-wartość: jedna wieża `f_θ(s) → (p, v)`. `p` to rozkład a priori nad legalnymi ruchami. `v` to oczekiwany wynik gry.
- Monte Carlo Tree Search (MCTS): przy każdym ruchu, rozwijaj drzewo możliwych kontynuacji. Użyj `(p, v)` jako rozkładu a priori + uruchomienia. Wybieraj węzły przez UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Samodzielna gra: graj gry agent-vs-agent. Przy ruchu `t`, rozkład odwiedzin MCTS `π_t` staje się celem treningowym polityki.
- Strata: `L = (v - z)² - π · log p + c · ||θ||²`. `z` to wynik gry (+1 / 0 / -1).

Zero ludzkiej wiedzy. Zero ręcznie wykonanych heurystyk. Jeden przepis, który opanował szachy, shogi i Go po kilkudziesięciu milionach gier samodzielnej gry każda.

**MuZero (2019).** Schrittwieser i in. Usuwa wymóg, że reguły muszą być znane.

- Zamiast ustalonego środowiska, naucz się *ukrytego modelu dynamiki* `(h, g, f)`:
  - `h(s)`: zakoduj obserwację do ukrytego stanu.
  - `g(s_latent, a)`: przewidź następny ukryty stan + nagrodę.
  - `f(s_latent)`: przewidź rozkład a priori polityki + wartość.
- MCTS działa w *wyuczonej ukrytej przestrzeni*. To samo wyszukiwanie, ta sama pętla treningowa.
- Działa na Go, szachy, shogi *i* Atari — jeden algorytm, żadnej znajomości reguł.

**Stochastyczny MuZero (2022).** Dodaje stochastyczną dynamikę i węzły szansy; rozszerza na gry klasy backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Ulepszenia efektywności próbkowej i deterministycznego wyszukiwania.

**GRPO (2024-2025).** Przepis DeepSeek-R1. Ta sama pętla w kształcie AlphaZero, zastosowana do rozumowania modelu językowego:

- "Gra": odpowiedz na problem matematyczny / kodowania / rozumowania. "Wygrana" = weryfikator (test przechodzi, odpowiedź numeryczna się zgadza) zwraca 1.
- Polityka: LLM. Akcje: tokeny. Stan: prompt + dotychczasowa odpowiedź.
- Żaden krytyk (V_φ w stylu PPO). Zamiast tego, dla każdego promptu, próbkuj `G` kompletacji z polityki. Oblicz nagrodę dla każdej. Użyj **przewagi względnej grupowo** `A_i = (r_i - mean_r) / std_r` jako sygnału do aktualizacji w stylu REINFORCE.
- Kara KL do polityki referencyjnej, aby zapobiec dryfowi (jak w RLHF).
- Pełna strata:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Żaden model nagrody, żaden krytyk, żadne MCTS. Wartość bazowa względna grupowo zastępuje wszystkie trzy. Dorównuje lub przewyższa jakość PPO-RLHF na benchmarkach rozumowania za ułamek mocy obliczeniowej.

**Pełny przepis R1.** DeepSeek-R1 (DeepSeek 2025) to dwa modele w jednej pracy:

- **R1-Zero.** Zacznij od modelu bazowego DeepSeek-V3. Żadnego SFT. Zastosuj GRPO bezpośrednio z dwoma składnikami nagrody: *nagrodą dokładności* (opartą na regułach — czy końcowa odpowiedź sparsowała się do poprawnej liczby / czy kod przeszedł testy jednostkowe) i *nagrodą formatu* (czy kompletacja zawinęła swój łańcuch myśli w tagi `thinking…response`). W ciągu tysięcy kroków, średnia długość odpowiedzi rośnie z ~100 do ~10 000 tokenów, a wyniki benchmarków matematycznych wspinają się do poziomu bliskiego o1-preview. Model uczy się rozumować od zera. Wada: jego łańcuchy myśli są często nieczytelne, mieszają języki i brak im stylistycznego wykończenia.
- **R1.** Napraw problemy z czytelnością R1-Zero za pomocą czteroetapowego pipeline'u:
  1. **Zimny start SFT.** Zbierz kilka tysięcy demonstracji długiego CoT z czystym formatowaniem. Nadzorczo dostrój model bazowy na nich. To daje czytelny punkt startowy.
  2. **GRPO zorientowane na rozumowanie.** Zastosuj GRPO z nagrodami dokładności+formatu plus *nagrodą spójności językowej*, aby zapobiec przełączaniu kodów.
  3. **Próbkowanie odrzucające + SFT runda 2.** Próbkuj ~600K trajektorii rozumowania z punktu kontrolnego RL, zachowaj tylko te z poprawnymi końcowymi odpowiedziami i czytelnym CoT, i połącz z ~200K przykładów SFT innych niż rozumowanie (pisanie, QA, samopoznanie). Dostrój bazę ponownie.
  4. **GRPO pełnego spektrum.** Jeszcze jedna runda RL obejmująca zarówno rozumowanie (nagrody oparte na regułach), jak i ogólne dopasowanie (nagrody oparte na preferencjach pomocności/nieszkodliwości).

Wynik dorównuje o1 na AIME i MATH-500 z otwartymi wagami i jest wystarczająco mały, by dystylować. Ta sama praca wydaje również sześć dystylowanych gęstych modeli (Qwen-1.5B przez Llama-70B) przez SFT na śladach rozumowania R1 — żadnego RL u ucznia. Dystylacja silnego nauczyciela RL konsekwentnie bije RL od zera w skali ucznia.

**Dlaczego GRPO zamiast PPO dla rozumowania.** Trzy powody w pracy DeepSeekMath (luty 2024): (1) brak sieci wartości do trenowania, zmniejszenie pamięci o połowę; (2) wartość bazowa grupy naturalnie obsługuje rzadką nagrodę na końcu trajektorii, którą produkują zadania rozumowania; (3) normalizacja na prompt czyni przewagi porównywalnymi między problemami o radykalnie różnym poziomie trudności, czego pojedynczy krytyk PPO nie może.

**Bez wyszukiwania vs z wyszukiwaniem.** Gry się rozgałęziły:

- *Gry o doskonałej informacji z długimi horyzontami* (Go, szachy): wciąż oparte na wyszukiwaniu. AlphaZero / MuZero dominują.
- *Rozumowanie LLM*: żadnego MCTS jeszcze w produkcji; GRPO na pełnych wykonaniach, Best-of-N dla mocy obliczeniowej wnioskowania. Modele nagrody procesu (PRM) sugerują, że wyszukiwanie na poziomie kroku jest dodawane z powrotem.

## Zbuduj To

Kod w `code/main.py` implementuje **GRPO w miniaturze** — bandytę z wieloma grupami próbek. Algorytm jest taki sam jak na LLM; tylko polityka i środowisko są prostsze. Uczy *straty* i *przewagi względnej grupowo*, która jest innowacją 2025.

### Krok 1: małe środowisko weryfikatora

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

W prawdziwym GRPO weryfikator uruchamia testy jednostkowe lub sprawdza równość matematyczną.

### Krok 2: polityka: softmax nad K tokenami odpowiedzi na prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Odpowiednik wyjścia ostatniej warstwy LLM warunkowanego na prompcie.

### Krok 3: próbkowanie grupowe i przewaga względna grupowo

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Przewaga względna grupowo to sztuczka DeepSeek z 2024. Żaden krytyk nie jest potrzebny. "Wartością bazową" jest średnia grupy, a normalizacja używa odchylenia standardowego grupy.

### Krok 4: porównaj z wartością bazową REINFORCE (bez wartości)

To samo ustawienie, ta sama moc obliczeniowa, zwykły REINFORCE. GRPO zbiega szybciej i stabilniej.

### Krok 5: obserwuj entropię i KL

Te same diagnostyki co w RLHF: średnie KL do referencji, entropia polityki, nagroda w czasie. Gdy się ustabilizują, trening jest zakończony.

## Pułapki

- **Hakowanie nagrody przez oszukiwanie weryfikatora.** GRPO dziedziczy ryzyko RLHF: jeśli weryfikator jest błędny lub możliwy do wykorzystania, LLM znajdzie exploit. Solidne weryfikatory (wiele przypadków testowych, formalne dowody) mają znaczenie.
- **Zbyt mały rozmiar grupy.** Wariancja wartości bazowej grupy zachowuje się jak `1/√G`. Poniżej `G = 4`, sygnał przewagi jest zaszumiony; standardowy wybór to `G = 8` do `64`.
- **Błąd długości.** Kompletacje LLM o różnych długościach mają różne log-prawdopodobieństwa. Normalizuj przez liczbę tokenów lub używaj log-prawdopodobieństwa na poziomie sekwencji, lub obcinaj do maksymalnej długości.
- **Cykle czystej samodzielnej gry.** Trening w stylu AlphaZero może utknąć w pętlach dominacji w grach o sumie ogólnej. Złagodzone przez różnorodne pule przeciwników (gra ligowa, Lekcja 10).
- **Niedopasowanie wyszukiwanie-polityka.** AlphaZero trenuje politykę, aby naśladować wynik wyszukiwania. Jeśli sieć polityki jest zbyt mała, by reprezentować rozkład wyszukiwania, trening się zatrzymuje.
- **Podłoga obliczeniowa.** MuZero / AlphaZero potrzebują ogromnej mocy obliczeniowej. Pojedyncza ablacja to często setki godzin GPU. Istnieją miniaturowe dema (np. AlphaZero na Connect Four) do nauki.
- **Pokrycie weryfikatora.** Testy jednostkowe, które przechodzą dla błędnego rozwiązania, wzmacniają błąd. Projektuj weryfikatory, które łapią przypadki brzegowe.

## Użyj Tego

Krajobraz RL dla gier w 2026, według domeny:

| Domena | Dominująca metoda |
|--------|-----------------|
| Dwuosobowe gry planszowe o sumie zerowej (Go, szachy, shogi) | AlphaZero / MuZero / KataGo |
| Karciane gry z niepełną informacją (poker) | CFR + głębokie uczenie (DeepStack, Libratus, Pluribus) |
| Gry Atari / pikselowe | Muesli / MuZero / IMPALA-PPO |
| Duże wieloosobowe strategie (Dota, StarCraft) | PPO + samodzielna gra + liga (OpenAI Five, AlphaStar) |
| Rozumowanie matematyczne/kodowe LLM | GRPO (DeepSeek-R1, Qwen-RL, otwarte replikacje) |
| Dopasowanie LLM | DPO / RLHF-PPO (nie GRPO; weryfikator to preferencja, nie weryfikowalne) |
| Robotyka | PPO + DR (nie RL dla gier, ale używa tych samych narzędzi gradientu polityki) |
| Problemy kombinatoryczne | Warianty AlphaZero (AlphaTensor, AlphaDev) |

*Przepis* — samodzielna gra, poprawa wspomagana wyszukiwaniem, dystylacja polityki — obejmuje tekst, piksele i sterowanie fizyczne. GRPO jest najmłodszą instancją; nadchodzą kolejne.

## Dostarcz To

Zapisz jako `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj bandytę GRPO w `code/main.py`. Trenuj na 2 promptach × 4 tokeny odpowiedzi każdy. Zbiegnij w < 1000 aktualizacji z `G=8`.
2. **Średnie.** Podłącz PPO (przycięte) i zwykły REINFORCE. Porównaj efektywność próbkową i wariancję nagrody z GRPO na tym samym bandycie.
3. **Trudne.** Rozszerz do "łańcucha rozumowania" o długości 2: agent emituje dwa tokeny, a weryfikator nagradza parę. Zmierz, jak GRPO radzi sobie z przypisaniem odpowiedzialności w sekwencjach dwukrokowych. (Wskazówka: oblicz przewagę grupowo na *pełną sekwencję*, propaguj do obu pozycji tokenów.)

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| MCTS | "Wyszukiwanie drzewiaste z wyuczoną siecią" | Monte Carlo Tree Search; selekcja UCB1/PUCT z wyuczonymi rozkładami a priori `(p, v)`. |
| AlphaZero | "Samodzielna gra + MCTS" | Sieć polityka-wartość trenowana do dopasowania odwiedzin MCTS i wyniku gry. |
| MuZero | "AlphaZero z wyuczonym modelem" | Ta sama pętla, ale w ukrytej przestrzeni przez wyuczoną dynamikę. |
| GRPO | "PPO bez krytyka" | Grupowa Optymalizacja Polityki Względnej; REINFORCE z wartością bazową średniej grupy + KL. |
| PUCT | "UCB AlphaZero" | `Q + c · p · √N / (1 + N_a)` — równoważy estymatę wartości z rozkładem a priori. |
| Samodzielna gra | "Agent przeciw przeszłemu sobie" | Standard dla gier o sumie zerowej; symetryczny sygnał treningowy. |
| Gra ligowa | "Samodzielna gra oparta na populacji" | Przeszli + obecni + eksploatatorzy próbkowani jako przeciwnicy. |
| Nagroda weryfikatora | "Weryfikowalny RL" | Nagroda pochodzi z deterministycznego sprawdzacza (testy przechodzą, odpowiedź się zgadza). |
| Nagroda procesu | "PRM" | Ocenia każdy krok rozumowania, nie tylko końcową odpowiedź. |

## Dalsza Lektura

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — praca, która wprowadziła GRPO i wartości bazowe względne grupowo.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — pełny czteroetapowy przepis R1 plus ablacja R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + głębokie uczenie w skali.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — praca, która to wszystko zapoczątkowała.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — produkcyjna referencja do stosowania GRPO z niestandardowymi funkcjami nagrody.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — otwarta replikacja przepisu R1 w wielu skalach.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznikowe ujęcie samodzielnej gry, wyszukiwania i "zaprojektowanej nagrody", które R1 instancjonuje w skali LLM.