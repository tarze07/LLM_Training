# ReWOO oraz Plan-and-Execute: Rozdzielone planowanie

> ReAct przeplata myśl i działanie w jednym strumieniu. ReWOO rozdziela je: jeden duży plan z góry, potem wykonanie. 5x mniej tokenów, +4% dokładności na HotpotQA, a planisty można wydestynować do modelu 7B. Plan-and-Execute uogólniło to; Plan-and-Act przeskalowało do nawigacji po stronach internetowych.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnij, dlaczego podział ReWOO na Planisty / Workerów / Rozwiązującego oszczędza tokeny i poprawia solidność względem przeplatanej pętli ReAct.
- Zaimplementuj DAG planu, executor zależności i rozwiązującego, który składa wyniki workerów — wszystko w stdlib.
- Zdecyduj, kiedy zadanie powinno działać jako plan-then-execute vs przeplatany ReAct, używając ram "pięciu wzorców workflow" z 2026 (Anthropic).
- Rozpoznaj, kiedy syntetyczne dane planu Plan-and-Act są potrzebne do długoterminowych zadań internetowych lub mobilnych.

## The Problem

Przeplatana pętla myśl-działanie-obserwacja ReAct jest prosta i elastyczna, ale każde wywołanie narzędzia musi nieść pełny wcześniejszy kontekst — w tym każdą poprzednią myśl. Użycie tokenów rośnie kwadratowo z głębokością. Co gorsza: gdy narzędzie zawiedzie w środku pętli, model musi wyprowadzić cały plan od nowa z obserwacji błędu.

ReWOO (Xu et al., arXiv:2305.18323, maj 2023) zauważyło to i postawiło zakład: zaplanuj całość z góry, pobierz dowody równolegle, skomponuj odpowiedź na końcu. Jedno wywołanie LLM do planowania, N wywołań narzędzi dla dowodów (mogą być równoległe), jedno wywołanie LLM do rozwiązania. Cena to mniejsza elastyczność (plan jest statyczny) za znacznie lepszą wydajność tokenową i wyraźniejsze tryby awarii.

## The Concept

### Trzy role

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (wywołania narzędzi, możliwie równoległe)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planista tworzy DAG. Każdy węzeł podaje nazwę narzędzia, jego argumenty i od których wcześniejszych węzłów zależy (referencje jak `#E1`, `#E2`). Workerzy wykonują węzły w porządku topologicznym. Rozwiązujący składa wszystko w całość.

### Dlaczego 5x mniej tokenów

ReAct zwiększa długość promptu liniowo z liczbą kroków. W kroku 10 prompt zawiera myśl 1 plus działanie 1 plus obserwację 1 plus myśl 2 plus działanie 2 plus obserwację 2, i tak dalej. Każdy pośredni krok również nadmiarowo zawiera oryginalny prompt.

ReWOO płaci jeden duży prompt planisty, N małych promptów workerów (każdy tylko wywołanie narzędzia, bez łańcucha) i jeden prompt rozwiązującego. Na HotpotQA praca mierzy ~5x mniej tokenów, osiągając +4 absolutnej dokładności.

### Dlaczego jest bardziej odporny

Jeśli worker 3 zawiedzie w ReAct, pętla musi wydedukować wyjście z błędu w środku strumienia. W ReWOO worker 3 zwraca string błędu; rozwiązujący widzi go w kontekście z oryginalnym planem i może elegancko zdegradować działanie. Lokalizacja awarii jest per-węzeł, a nie per-krok.

### Dystylacja planisty

Drugi wynik pracy: ponieważ planista nie widzi obserwacji, można dostroić model 7B na wyjściach planisty z nauczyciela 175B. Mały model zajmuje się planowaniem; duży model nie jest potrzebny podczas wnioskowania. Jest to teraz standard — wielu produkcyjnych agentów w 2026 używa małego planisty i dużego executora lub odwrotnie.

### Plan-and-Execute (LangChain, 2023)

Postęp zespołu LangChain z sierpnia 2023 uogólnił ReWOO do wzorca o nazwie: Plan-and-Execute. Planista z góry emituje listę kroków, executor wykonuje każdy krok, opcjonalny replanista może zrewidować po zaobserwowaniu wyników. Jest to bliższe ReAct niż ReWOO (replanista wprowadza obserwacje z powrotem do planowania), ale zachowuje oszczędność tokenów.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act skaluje wzorzec do długoterminowych agentów internetowych i mobilnych. Kluczowym wkładem są syntetyczne dane planu: etykietowany generator trajektorii produkuje dane treningowe, w których plan jest jawny. Używane do dostrajania modeli planistów, które działają poprawnie przez 30–50 kroków w zadaniach typu WebArena, gdzie pojedyncza trajektoria ReAct traci spójność.

### Kiedy wybierać który

| Wzorzec | Kiedy |
|---------|------|
| ReAct | Krótkie zadania, nieznane środowisko, potrzeba reaktywnej obsługi wyjątków |
| ReWOO | Ustrukturyzowane zadania ze znanymi narzędziami, wrażliwość na tokeny, możliwa równoległość dowodów |
| Plan-and-Execute | Jak ReWOO, ale z replanowaniem po częściowym wykonaniu |
| Plan-and-Act | Długoterminowe (>30 kroków), internet/mobilne/użycie komputera |
| Tree of Thoughts | Wyszukiwanie jest warte dopłaty (Lekcja 04) |

Wskazówka Anthropic z grudnia 2024: zacznij od najprostszego. Jeśli zadanie to jedno wywołanie narzędzia plus podsumowanie, nie buduj ReWOO. Jeśli zadanie to 40-krokowe zadanie badawcze, nie używaj samego ReAct.

## Build It

`code/main.py` implementuje zabawkowe ReWOO:

- `Planner` — skryptowana polityka, która emituje DAG planu z promptu.
- `Worker` — wysyła wywołanie narzędzia każdego węzła przez rejestr.
- `Solver` — skryptowana kompozycja, która czyta dowody i produkuje ostateczną odpowiedź.
- Rozdzielczość zależności — referencje jak `#E1` są zastępowane wcześniejszymi wynikami workerów.

Demo odpowiada "Jaka jest populacja stolicy Francji, zaokrąglona do milionów?" za pomocą dwuetapowego planu: (1) znajdź stolicę, (2) znajdź populację, następnie rozwiąż.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje najpierw pełny plan, potem wyniki workerów, potem kompozycję rozwiązującego. Porównaj liczbę tokenów (drukujemy przybliżoną liczbę znaków) do przeplatanej pętli w stylu ReAct — ReWOO wygrywa w tego typu ustrukturyzowanych zadaniach.

## Use It

LangGraph dostarcza Plan-and-Execute jako przepis (`create_react_agent` dla ReAct, niestandardowe grafy dla plan-execute). CrewAI Flows kodują wzorzec bezpośrednio: definiujesz zadania z góry i DAG Flow je wykonuje. Podejście z syntetycznymi danymi Plan-and-Act jest w większości wciąż badawcze; wzorzec uruchomieniowy (jawny DAG planu) działa w produkcji przez LangGraph i CrewAI Flows.

## Ship It

`outputs/skill-rewoo-planner.md` generuje DAG planu ReWOO z żądania użytkownika, mając katalog narzędzi. Weryfikuje plan (acykliczny, każda referencja rozwiązana, każde narzędzie istnieje) przed przekazaniem do executora.

## Exercises

1. Zrównoleglij wykonanie workerów dla niezależnych węzłów planu. Co to daje na 6-węzłowym DAG z 2 równoległymi grupami?
2. Dodaj węzeł replanisty, który uruchamia się, jeśli któryś worker zwróci błąd. Jaka jest najmniejsza zmiana w ReWOO, która czyni je Plan-and-Execute?
3. Zastąp `Planner` małym modelem (klasa 7B) i utrzymaj `Solver` na modelu granicznym. Porównaj jakość od końca do końca — gdzie zawodzi podział?
4. Przeczytaj Sekcję 4 pracy ReWOO o dystylacji planisty. Odtwórz koncepcyjnie wynik 175B -> 7B: jakich danych treningowych potrzebujesz i jak oceniasz jakość planu?
5. Przenieś zabawkę do kształtu trajektorii Plan-and-Act: plan jest sekwencją, a nie DAGiem. Jakie zmieniają się kompromisy?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| ReWOO | "Rozumowanie bez obserwacji" | Zaplanuj, potem zbierz dowody równolegle, potem rozwiąż — brak obserwacji w prompcie planowania |
| Plan-and-Execute | "Wzorzec plan-wykonaj LangChain" | ReWOO z opcjonalnym węzłem replanisty po wykonaniu |
| Plan-and-Act | "Skalowane plan-wykonaj" | Jawny podział planista/executor z syntetycznymi danymi treningowymi planu dla długoterminowych zadań |
| Referencja dowodu | "#E1, #E2, ..." | Symbol zastępczy węzła planu zastępowany wcześniejszym wynikiem workera w momencie wysyłki |
| Dystylacja planisty | "Mały planista, duży executor" | Dostrój mały model na śladach planisty z dużego nauczyciela |
| Wydajność tokenowa | "Mniej rund" | 5x mniej tokenów na HotpotQA vs ReAct w pracy |
| Executor DAG | "Topologiczny dyspozytor" | Uruchamia węzły planu w kolejności zależności; równolegle na każdym poziomie |

## Further Reading

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — kanoniczna praca
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — przeskalowany planista-executor z syntetycznymi planami
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — przepis frameworka
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — wybierz najprostszy wzorzec, który działa

(End of file - total 121 lines)