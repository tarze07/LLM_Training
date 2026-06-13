# Tree of Thoughts i LATS: Świadome przeszukiwanie

> Pojedyncza trajektoria łańcucha myśli nie ma miejsca na wycofanie się. ToT (Yao et al., 2023) zamienia rozumowanie w drzewo z samooceną na każdym węźle. LATS (Zhou et al., 2024) jednoczy ToT z ReAct i Reflexion pod Monte Carlo Tree Search. Gra 24 idzie z 4% (CoT) do 74% (ToT); LATS osiąga 92.7% pass@1 na HumanEval.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## Learning Objectives

- Ramuj rozumowanie jako przeszukiwanie: węzły to "myśli", krawędzie to "ekspansje", wartość to "jak obiecujące".
- Zaimplementuj w stdlib przeszukiwanie drzewa BFS w stylu ToT z punktacją samooceny.
- Rozszerz do zabawkowej pętli MCTS LATS z select / expand / simulate / backpropagate.
- Zdecyduj, kiedy przeszukiwanie jest warte mnożnika tokenów (Gra 24, generowanie kodu), a kiedy wystarczy pojedyncza trajektoria (proste Q&A).

## The Problem

Chain-of-thought to liniowy spacer. Jeśli pierwszy krok jest błędny, każdy kolejny krok działa na złym założeniu. Na Grze 24 (użyj czterech cyfr z + − × ÷, aby uzyskać 24), GPT-4 CoT osiąga 4% dokładności. Model wybiera wcześnie złe podwyrażenie i nie może się odratować.

To, czego potrzebuje rozumowanie, to zdolność do proponowania wielu kandydatów, oceniania ich, wybierania obiecujących i wycofywania się, gdy pojawiają się ślepe uliczki. To jest przeszukiwanie. Tree of Thoughts i LATS to dwie kanoniczne formulacje.

## The Concept

### Tree of Thoughts (Yao et al., NeurIPS 2023)

Każdy węzeł to spójny pośredni krok ("myśl"). Każdy węzeł może rozszerzyć się do K myśli-dzieci. LLM samoocenia każdy węzeł za pomocą promptu punktacji. Przeszukiwanie eksploruje drzewo — BFS, DFS lub beam.

```
                      (root: "znajdź 24 z 4 6 4 1")
                     /               |            \
            ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Wynik: HIGH
               /   \              |                  |
           ...    ...          ...                finish
```

Samoocena jest kluczowym elementem. Praca pokazuje trzy warianty: klasyfikacja `sure / likely / impossible`, numeryczna ocena `1..10` i głosowanie między kandydatami. Wszystkie trzy znacząco biją CoT na Grze 24 (4% -> 74% z GPT-4).

### LATS (Zhou et al., ICML 2024)

LATS jednoczy ToT, ReAct i Reflexion pod MCTS. LLM gra trzy role:

- **Polityka**: proponuj kandydatów na następne akcje (w stylu ReAct).
- **Funkcja wartości**: oceniaj częściową trajektorię (samoocena w stylu ToT).
- **Samo-reflektor**: w razie porażki napisz refleksję w języku naturalnym (w stylu Reflexion) i użyj jej do ponownego zasiania przyszłych zrzutów.

Informacja zwrotna ze środowiska (obserwacje) miesza się z funkcją wartości, więc przeszukiwanie jest informowane przez rzeczywiste wyniki narzędzi, a nie tylko opinie modelu. Wyniki w czasie publikacji: HumanEval pass@1 92.7% z GPT-4 (SOTA), WebShop średnio 75.9 z GPT-3.5 (zbliżając się do dostrajania gradientowego).

### MCTS, minimalistycznie

Cztery fazy na iterację:

1. **Select** — przejdź od korzenia do liścia używając UCT (górna granica ufności dla drzew).
2. **Expand** — wygeneruj K dzieci przez politykę.
3. **Simulate** — zrzut z dziecka używając polityki, oceń liść funkcją wartości (lub nagrodą środowiska).
4. **Backpropagate** — zaktualizuj liczby odwiedzin i estymacje wartości w górę ścieżki.

Wzór UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Pierwszy człon to eksploatacja; drugi to eksploracja. Dostrój `c` per zadanie.

### Rzeczywistość kosztów

Przeszukiwanie eksploduje tokenami. ToT na Grze 24 używa 100–1000x tokenów CoT. LATS podobnie. To nie jest darmowe; zarezerwuj przeszukiwanie dla:

- Zadań, gdzie pojedyncza trajektoria jest demonstrowalnie niewystarczająca (Gra 24, złożony kod).
- Zadań, gdzie czas ścienny jest mniej ważny niż poprawność.
- Zadań z tanią, niezawodną funkcją wartości (testy jednostkowe dla kodu, jawny cel dla matematyki).

Jeśli twoje zadanie ma jedną poprawną odpowiedź i głośny ewaluator, przeszukiwanie często pogarsza sytuację — znajduje "dobrze punktowaną" błędną odpowiedź.

### Pozycjonowanie 2026

Większość produkcyjnych agentów nie uruchamia LATS. Uruchamiają ReAct z weryfikacją opartą na narzędziach (CRITIC, Lekcja 05). Przeszukiwanie pojawia się w wyspecjalizowanych niszach:

- Agenci kodowania, którzy uruchamiają testy jako funkcję wartości (styl HumanEval).
- Agenci głębokiego badania, którzy eksplorują wiele ścieżek zapytań.
- Workflowy intensywnie planujące wewnątrz podgrafów LangGraph.

AlphaEvolve (Lekcja 11) to ekstremum z 2025 roku: ewolucyjne przeszukiwanie kodu, sprawdzalna maszynowo sprawność, przełomowe zyski (pierwsza poprawa 4x4 matmul od 56 lat).

## Build It

`code/main.py` implementuje:

- Miniaturowe BFS ToT na stylizowanym zadaniu "wybierz operacje arytmetyczne".
- Zabawkową pętlę MCTS LATS na tym samym zadaniu (Select / Expand / Simulate / Backpropagate) z selekcją UCT.
- Funkcję wartości, która składa symboliczny wynik plus wynik samooceny.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje ToT rozszerzające trzech kandydatów na węzeł z BFS, w porównaniu do LATS zbiegającego do najlepszego zrzutu przez MCTS. Liczby tokenów wydrukowane dla obu.

## Use It

LangGraph dostarcza eksplorację w stylu ToT jako wzorce podgrafów; blog zespołu LangChain o LATS (maj 2024) jest referencyjnym tutorialem. LlamaIndex dostarcza agenta `TreeOfThoughts`. Dla większości produkcyjnych agentów 2026 ten wzorzec żyje za bramką `if task_complexity > threshold: use_search()` — zobacz wzorzec evaluator-optimizer w Lekcji 05.

## Ship It

`outputs/skill-search-policy.md` wybiera między liniowym ReAct, ToT, LATS i przeszukiwaniem ewolucyjnym, biorąc pod uwagę kształt zadania, budżet i wierność ewaluatora.

## Exercises

1. Uruchom zabawkowe LATS z UCT c=0.1 vs c=2.0. Co zmienia się w śladzie?
2. Zamień funkcję wartości na bardziej zaszumioną punktację (dodaj losowe jitter). Czy MCTS wciąż znajduje najlepszy liść? Jaki jest minimalny stosunek sygnału do szumu, który toleruje?
3. Zaimplementuj ToT z przeszukiwaniem beam (utrzymuj top-k na każdym poziomie) i porównaj z BFS. Który jest lepszy przy ciasnym budżecie tokenów?
4. Przeczytaj LATS Sekcję 5.1. Odtwórz liczbę trajektorii HumanEval: ile zrzutów potrzeba, by osiągnąć raportowane pass@1?
5. Przeczytaj dyskusję pracy LATS o "kiedy LATS pomaga mniej." Napisz jednoakapitową regułę decyzyjną mapującą kształt zadania na strategię przeszukiwania.

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Tree of Thoughts | "Rozgałęzione CoT" | Yao et al. — drzewo węzłów myśli z samooceną |
| LATS | "MCTS dla LLM" | Zhou et al. — jednoczy ToT + ReAct + Reflexion pod MCTS |
| UCT | "Górna granica ufności" | Wzór selekcji równoważący eksploatację (Q) i eksplorację (ln N / n) |
| Funkcja wartości | "Jak dobry jest ten stan" | Punktacja LLM lub nagroda środowiska; zasila backpropagację |
| Polityka | "Proponent akcji" | Generator w stylu ReAct; emituje kandydackie następne myśli/akcje |
| Zrzut | "Symulowana trajektoria" | Przejście od węzła do liścia używając polityki, ocena funkcją wartości |
| Backpropagować | "Aktualizuj przodków" | Wepchnij nagrodę liścia w górę ścieżki, aktualizując liczniki odwiedzin i Q |
| Koszt przeszukiwania | "Eksplozja tokenów" | 100-1000x CoT na Grze 24; budżet przed adopcją |

## Further Reading

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — kanoniczna praca
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS z informacją zwrotną Reflexion
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — wzorce podgrafów dla przeszukiwania
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — ewolucyjne przeszukiwanie z programowalnymi ewaluatorami

(End of file - total 130 lines)