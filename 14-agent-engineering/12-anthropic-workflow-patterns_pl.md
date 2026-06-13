# Wzorce przepływów pracy Anthropic: Proste ponad złożone

> Schluntz i Zhang (Anthropic, grudzień 2024) rozróżniają przepływy pracy (predefiniowane ścieżki) od agentów (dynamiczne użycie narzędzi). Pięć wzorców przepływów pokrywa większość przypadków. Zacznij od bezpośrednich wywołań API. Dodawaj agentów tylko wtedy, gdy kroków nie można przewidzieć.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Wymień pięć wzorców przepływów Anthropic: łańcuch promptów, routing, parallelizacja, orchestrator-pracownicy, ewaluator-optymalizator.
- Wyjaśnij rozróżnienie agent vs przepływ pracy i koszt inżynieryjny każdego z nich.
- Zidentyfikuj, kiedy wybrać przepływ pracy zamiast agenta (i odwrotnie).
- Zaimplementuj wszystkie pięć wzorców w stdlib przeciwko skryptowanemu LLM.

## The Problem

Zespoły sięgają po wieloagentowe frameworki dla problemów, które wymagają pojedynczego wywołania funkcji. Koszt jest realny: frameworki dodają warstwy, które zaciemniają prompty, ukrywają przepływ sterowania i zapraszają do przedwczesnej złożoności. Post Schluntza i Zhanga z grudnia 2024 jest najczęściej cytowanym branżowym sprzeciwem: zaczynaj prosto, dodawaj złożoność tylko wtedy, gdy zapracuje na swój koszt.

## The Concept

### Przepływy pracy vs agenci

- **Przepływ pracy.** LLM i narzędzia orkiestrowane przez predefiniowane ścieżki kodu. Inżynierowie są właścicielami grafu.
- **Agent.** LLM dynamicznie kieruje własnymi narzędziami i podejmuje własne kroki. Model jest właścicielem grafu.

Oba mają swoje miejsce. Przepływy pracy są tańsze, szybsze i łatwiejsze do debugowania. Agenci odblokowują problemy otwarte, ale utrudniają wnioskowanie o trybach awarii.

### Rozszerzony LLM

Podstawa dla wszystkich pięciu wzorców: jeden LLM z trzema wbudowanymi zdolnościami — wyszukiwanie (retrieval), narzędzia (akcje), pamięć (trwałość). Każde wywołanie API może ich użyć.

### Pięć wzorców

1. **Łańcuch promptów.** Wynik wywołania 1 jest wejściem do wywołania 2. Używaj, gdy zadanie ma czystą liniową dekompozycję. Opcjonalne programowalne bramki między krokami.

2. **Routing.** Klasyfikator LLM wybiera, który downstream LLM lub narzędzie wywołać. Używaj, gdy kategorycznie różne wejścia wymagają różnego przetwarzania (wsparcie tier-1 vs zwrot vs błąd vs sprzedaż).

3. **Parallelizacja.** Uruchom N wywołań LLM współbieżnie, agreguj wyniki. Dwa kształty: sekcjonowanie (różne fragmenty) i głosowanie (ten sam prompt, N uruchomień, większość/synteza).

4. **Orchestrator-pracownicy.** Orchestrator LLM dynamicznie decyduje, których pracowników (również LLM) uruchomić i syntetyzuje ich wyniki. Podobne do pętli agenta, ale orchestrator nie zapętla się w nieskończoność.

5. **Ewaluator-optymalizator.** Jeden LLM proponuje odpowiedź, inny LLM ją ocenia. Iteruj, aż ewaluator zda. Jest to uogólniony Self-Refine (Lekcja 05).

### Gdzie przepływy pracy biją agentów

- **Przewidywalne zadania.** Jeśli możesz wyliczyć kroki, powinieneś.
- **Zadania z ograniczeniem kosztów.** Przepływy pracy mają ograniczoną liczbę kroków; agenci mogą się zapętlić.
- **Zadania związane zgodnością.** Audytorzy chcą czytać graf, a nie wnioskować go z trajektorii.

### Gdzie agenci biją przepływy pracy

- **Otwarte badania.** Gdy następny krok zależy od tego, co zwrócił poprzedni.
- **Zadania o zmiennej długości.** Minuty do godzin pracy, gdzie liczba kroków jest nieznana.
- **Nowe domeny.** Gdy nie znasz jeszcze właściwego przepływu pracy — najpierw eksploracja, potem kodyfikacja.

### Towarzysząca inżynieria kontekstu

"Effective context engineering for AI agents" (Anthropic 2025) formalizuje sąsiednią dyscyplinę: okno 200k to budżet, a nie pojemnik. Co uwzględnić, kiedy skompresować, kiedy pozwolić kontekstowi rosnąć. Omówione szczegółowo w lekcji Fazy 14 na temat kompresji kontekstu (Faza 14, wcześniejsza lekcja 06 w tym programie nauczania przed przenumerowaniem).

## Build It

`code/main.py` implementuje wszystkie pięć wzorców przepływów przeciwko `ScriptedLLM`:

- `prompt_chain(input, steps)` — sekwencyjny.
- `route(input, classifier, handlers)` — klasyfikacja + wysyłka.
- `parallel_vote(prompt, n, aggregator)` — N uruchomień, agregacja.
- `orchestrator_workers(task, workers)` — orchestrator wybiera pracowników.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — pętla do zaliczenia.

Uruchom:

```
python3 code/main.py
```

Każdy wzorzec wypisuje swój ślad. Łączna liczba linii kodu na wzorzec to ~10-15; koszt frameworka mierzy się w tysiącach.

## Use It

- Bezpośrednie wywołania API dla większości zadań.
- Framework tylko wtedy, gdy wzorzec naprawdę potrzebuje trwałego stanu (LangGraph), współbieżności modelu aktora (AutoGen v0.4) lub szablonowania ról (CrewAI).
- Sięgaj po Claude Agent SDK, gdy chcesz kształt uprzęży Claude Code bez odbudowywania go.

## Ship It

`outputs/skill-workflow-picker.md` wybiera odpowiedni wzorzec dla danego opisu zadania, wraz z uzasadnieniem decyzji i ścieżką refaktoryzacji do agenta, jeśli przepływy pracy nie wystarczą.

## Exercises

1. Zaimplementuj routing z progiem ufności. Poniżej progu -> eskaluj do człowieka. Gdzie ląduje próg dla przypadku użycia wsparcia tier-1?
2. Dodaj limit czasu do `parallel_vote`. Co się dzieje, gdy jedno wywołanie zawiesza się? Jak agregujesz z brakującymi głosami?
3. Zamień `evaluator_optimizer` w bandytę: przechowuj top-2 wyniki z iteracji, aby późny dobry wynik nie został nadpisany przez późny zły.
4. Połącz łańcuch promptów z routingiem: router wybiera jeden z trzech łańcuchów. Zmierz koszt tokenów w porównaniu z pojedynczą alternatywą z dużym promptem.
5. Wybierz jedną ze swoich funkcji produkcyjnych. Narysuj graf przepływu pracy. Policz kroki. Czy agent byłby tutaj faktycznie lepszy?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Przepływ pracy | "Predefiniowany przepływ" | Graf należący do inżyniera, wywołań LLM i narzędzi |
| Agent | "Autonomiczna AI" | Graf należący do modelu; dynamiczne kierowanie narzędziami |
| Rozszerzony LLM | "LLM z narzędziami" | LLM + wyszukiwanie + narzędzia + pamięć; jednostka atomowa |
| Łańcuch promptów | "Wywołania sekwencyjne" | Wynik wywołania N jest wejściem do wywołania N+1 |
| Routing | "Wysyłka klasyfikatora" | Wybierz, który łańcuch/model obsłuży wejście |
| Parallelizacja | "Rozgłaszanie" | N współbieżnych wywołań; agreguj przez sekcjonowanie lub głosowanie |
| Orchestrator-pracownicy | "Agent dyspozytor" | Orchestrator LLM dynamicznie wybiera wyspecjalizowane LLM |
| Ewaluator-optymalizator | "Proponujący + sędzia" | Iteruj, aż ewaluator zda; uogólniony Self-Refine |

## Further Reading

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — pięć wzorców przepływów pracy
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — sąsiednia dyscyplina
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — kiedy grafy stanowe zapracowują na swój koszt
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — wzór orchestrator-pracownicy, sproduktyzowany