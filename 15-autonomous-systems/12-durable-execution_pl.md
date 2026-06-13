# Długo Działający Agenci w Tle: Trwałe Wykonanie

> Produkcyjni długohoryzontowi agenci nie działają w `while True`. Każde wywołanie LLM staje się aktywnością z punktem kontrolnym, ponawianiem i odtwarzaniem. Integracja Temporal z OpenAI Agents SDK weszła w GA w marcu 2026. Claude Code Routines (Anthropic) uruchamia zaplanowane wywołania Claude Code bez trwałego lokalnego procesu. Sesje wstrzymują się na wejściu człowieka, przetrwają wdrożenia i wznawiają od najnowszego punktu kontrolnego kluczowanego przez `thread_id`. Za nową ergonomią kryje się stary wzorzec — orkiestracja przepływu pracy — z jednym nowym elementem: wywołania LLM jako niedeterministyczne aktywności, które muszą być deterministycznie odtworzone przy odzyskiwaniu.

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## Problem

Rozważ agenta, który działa przez cztery godziny. Wywołuje trzy narzędzia, monituje użytkownika dwa razy i wykonuje czterdzieści wywołań LLM. W połowie, host, na którym działa, uruchamia się ponownie. Co się dzieje?

- W naiwnej pętli `while True`: wszystko jest stracone. Uruchomienie zaczyna się od nowa. Trzy wywołania narzędzi (z prawdziwymi efektami ubocznymi) wykonują się ponownie. Użytkownik jest ponownie monitowany o rzeczy, które już zatwierdził. Czterdzieści wywołań LLM jest ponownie fakturowanych.
- Z trwałym wykonaniem: uruchomienie wznawia się od najnowszego punktu kontrolnego. Już ukończone aktywności nie są ponownie wykonywane; ich wyniki są odtwarzane z trwałego dziennika. Użytkownik nie zatwierdza ponownie rzeczy, które już zatwierdził. Wywołania LLM już wykonane nie są ponownie fakturowane.

To ten sam wzorzec, który silniki przepływu pracy dostarczają od dekady (Temporal, Cadence, Cherami Ubera). Nowością jest to, że wywołania LLM są teraz rodzajem aktywności — niedeterministyczne, kosztowne, z efektami ubocznymi — i pasują do tego wzorca czysto.

Temat przewodni lekcji: niezawodność długiego horyzontu spada (METR obserwuje „35-minutową degradację" — wskaźnik sukcesu spada mniej więcej kwadratowo z horyzontem). Trwałe wykonanie umożliwia uruchomienia dłuższe, niż wspiera profil niezawodności, co jest nowym sposobem na bezpieczną awarię, jeśli projekt jest poprawny, i niebezpieczną, jeśli projekt jest błędny.

## Koncepcja

### Aktywności, przepływy pracy i odtwarzanie

- **Przepływ pracy**: deterministyczny kod orkiestracji. Definiuje sekwencję aktywności, gałęzie, oczekiwania. Musi być deterministyczny, aby mógł być odtworzony z dziennika zdarzeń bez zaskakującej rozbieżności.
- **Aktywność**: niedeterministyczna, potencjalnie zawodna jednostka pracy. Wywołanie LLM, wywołanie narzędzia, zapis pliku, żądanie HTTP. Każda aktywność jest rejestrowana z jej wejściami i (po zakończeniu) jej wyjściami.
- **Dziennik zdarzeń**: trwałe przechowywanie zapasowe. Każdy start, zakończenie, niepowodzenie, ponowienie i każda decyzja przepływu pracy jest rejestrowana.
- **Odtwarzanie**: przy odzyskiwaniu, kod przepływu pracy uruchamia się ponownie od początku; każda aktywność, która już się zakończyła, zwraca swój zarejestrowany wynik bez ponownego wykonania. Tylko aktywności, które się nie zakończyły, są faktycznie uruchamiane.

To ten sam kształt, co ponowne renderowanie Reacta przeciwko wirtualnemu DOM, lub Git odbudowujący drzewo robocze z commitów. Deterministyczność w orkiestratorze jest tym, co czyni trwałość tanią.

### Dlaczego wywołania LLM pasują do wzorca

Wywołania LLM są:
- Niedeterministyczne (temperatura > 0; nawet temperatura 0 dryfuje między wersjami modelu).
- Kosztowne (pieniądze i opóźnienie).
- Potencjalnie zawodne (limity stawek, przekroczenia czasu).
- Z efektami ubocznymi (jeśli wywołują narzędzia).

To jest dokładnie profil aktywności. Opakowanie każdego wywołania LLM jako aktywności daje ponawianie z wykładniczym backoffem, punktowanie kontrolne między restartami i możliwą do odtworzenia ścieżkę do debugowania.

### Punkty kontrolne kluczowane przez `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects i Claude Code Routines wszystkie zbiegły się do tego samego kształtu API: `thread_id` (lub odpowiednik) identyfikuje sesję; każde przejście stanu utrwala się do backendu (PostgreSQL domyślnie, SQLite do dev, Redis do cache); wznowienie odczytuje najnowszy punkt kontrolny.

Wybór backendu ma znaczenie:

- **PostgreSQL**: trwały, możliwy do zapytania, przetrwający wdrożenia. Domyślny dla LangGraph.
- **SQLite**: tylko lokalny dev; traci dane między hostami.
- **Redis**: szybki, ale ulotny, chyba że skonfigurowano AOF/snapshot.
- **Cloudflare Durable Objects**: przezroczysto rozproszony; zakresiony przez unikalny klucz; trwa od godzin do tygodni.

### Wejście człowieka jako stan pierwszej klasy

Proponuj-następnie-zatwierdź (Lekcja 15) wymaga trwałego stanu „oczekiwanie na człowieka". Przepływ pracy wstrzymuje się, zewnętrzna kolejka przechowuje oczekujące żądanie, a zatwierdzenie wznawia dokładnie od tego punktu. Bez trwałości jest to najlepszy wysiłek; z nią, zatwierdzenie z dnia poprzedniego dociera i przepływ pracy wznawia się rano.

### 35-minutowa degradacja

METR zaobserwował, że każda mierzona klasa agenta wykazuje spadek niezawodności po ~35 minutach ciągłej pracy. Podwojenie czasu trwania zadania mniej więcej podwaja wskaźnik awarii. Trwałe wykonanie tego nie naprawia; pozwala działać dłużej, niż wspiera profil niezawodności. Bezpiecznym wzorcem jest połączenie trwałości z punktami kontrolnymi, które wymagają świeżego HITL przy ponownym wejściu, oraz z budżetowymi wyłącznikami awaryjnymi (Lekcja 13), które ograniczają całkowite obliczenia niezależnie od czasu zegarowego.

### Kiedy trwałe wykonanie jest złą odpowiedzią

- Uruchomienia krótsze niż kilka minut bez wejścia człowieka. Narzut > korzyść.
- Ściśle tylko-do-odczytu wyszukiwanie informacji.
- Zadania, gdzie poprawność wymaga end-to-end w ramach jednego okna kontekstowego (niektóre zadania rozumowania; niektóre jednorazowe generacje).

```figure
memory-consolidation
```

## Użyj Tego

`code/main.py` implementuje minimalny silnik trwałego wykonania w stdlib Python. Obsługuje:

- Dekorator `@activity`, który rejestruje wejścia i wyjścia w dzienniku zdarzeń JSON.
- Funkcję przepływu pracy, która szereguje aktywności.
- Funkcję `run_or_replay(workflow, event_log)`, która odtwarza ukończone aktywności bez ich ponownego wykonywania.

Sterownik symuluje trzyaktywnościowy przepływ pracy, ulega awarii w połowie i pokazuje (a) naiwne ponowienie wykonujące wszystko od nowa versus (b) odtwarzanie uruchamiające tylko brakującą aktywność.

## Wdróż To

`outputs/skill-durable-execution-review.md` recenzuje proponowane wdrożenie długo działającego agenta pod kątem poprawnego kształtu trwałego wykonania: aktywności, determinizm, backend punktów kontrolnych, stan wejścia człowieka i polityka HITL-przy-wznowieniu.

## Ćwiczenia

1. Uruchom `code/main.py`. Zaobserwuj różnicę w liczbie wykonanych aktywności między naiwnym ponowieniem a odtwarzaniem. Zmień punkt awarii i pokaż, że liczba odtworzeń zmienia się odpowiednio.

2. Przekształć zabawkowy silnik, aby jawnie używać `thread_id`. Zasymuluj dwie współbieżne sesje współdzielące silnik i potwierdź, że ich dzienniki zdarzeń nie kolidują.

3. Weź jedną aktywność w zabawkowym silniku. Wprowadź niedeterministyczność (znacznik czasu zegara ściennego wewnątrz decyzji przepływu pracy). Zademonstruj rozbieżność przy odtwarzaniu. Wyjaśnij, jak prawdziwe silniki sobie z tym radzą (rejestracja efektów ubocznych, API `Workflow.now()`).

4. Przeczytaj wpis LangChain „Runtime behind production deep agents". Wypisz każdy stan, który środowisko uruchomieniowe utrwala i nazwij, który tryb awarii każdy pokrywa.

5. Zaprojektuj politykę punktów kontrolnych dla 6-godzinnego autonomicznego zadania kodowania. Gdzie robisz punkty kontrolne? Jak wygląda wznowienie po awarii? Co wymaga świeżego HITL?

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|---|---|---|
| Przepływ pracy | „Skrypt agenta" | Deterministyczny kod orkiestracji; możliwy do odtworzenia z dziennika zdarzeń |
| Aktywność | „Krok" | Niedeterministyczna jednostka (wywołanie LLM, wywołanie narzędzia); logowana przed i po |
| Dziennik zdarzeń | „Magazyn zapasowy" | Trwały zapis każdego przejścia stanu |
| Odtwarzanie | „Wznowienie" | Ponowne uruchomienie przepływu pracy; ukończone aktywności zwracają zarejestrowane wyniki bez ponownego wykonania |
| Punkt kontrolny | „Punkt zapisu" | Utrwalony stan kluczowany przez thread_id; najnowszy wygrywa przy wznowieniu |
| thread_id | „Klucz sesji" | Identyfikator zakresiający trwały stan |
| 35-minutowa degradacja | „Spadek niezawodności" | METR: wskaźnik sukcesu spada ~kwadratowo z horyzontem |
| Niedeterministyczność | „Dryf przy odtwarzaniu" | Zegar ścienny, losowość, wynik LLM; musi być zarejestrowana jako efekt uboczny |

## Dalsza Lektura

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budżet, tury i semantyka wznawiania.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — kształt RequestInfoEvent.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — konkretne wymagania środowiska uruchomieniowego.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — kształt aktywności dla wywołań LLM.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — referencja 35-minutowej degradacji.
