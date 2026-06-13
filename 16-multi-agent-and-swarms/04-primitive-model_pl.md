# Model Prymitywów Wieloagentowych

> Każdy framework wieloagentowy wydany w 2026 roku — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — jest punktem w czterowymiarowej przestrzeni projektowej. Cztery prymitywy, nic więcej: agent, przekazanie, współdzielony stan, orkiestrator. Ta lekcja buduje je od zera, uruchamia zabawkowy system na wszystkich czterech, a następnie mapuje każdy główny framework na te same osie, abyś mógł przeczytać każde nowe wydanie w jednym akapicie.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Problem

Co sześć miesięcy pojawia się nowy framework wieloagentowy. AutoGen w 2023. CrewAI w 2024. LangGraph i OpenAI Swarm w 2024. Google ADK w kwietniu 2025. Microsoft Agent Framework RC w lutym 2026. Każdy komunikat prasowy twierdzi, że to „właściwa abstrakcja."

Jeśli spróbujesz uczyć się ich jeden po drugim, wypalisz się. API wyglądają inaczej. Dokumentacje nie zgadzają się co do tego, czym jest „agent." Jeden framework nazywa swoją współdzieloną pamięć „blackboard", inny nazywa ją „pulą komunikatów", trzeci nazywa ją „StateGraph." Zaczynasz podejrzewać, że dziedzina tylko się kręci.

Nie jest. Pod marketingiem cztery prymitywy są stabilne. Naucz się ich raz, czytaj każdy nowy framework w jednym akapicie.

## Concept

### Cztery prymitywy

1. **Agent** — prompt systemowy plus lista narzędzi. Bezstanowy; każde uruchomienie zaczyna się od promptu systemowego i bieżącej historii wiadomości.
2. **Przekazanie** — strukturalne przeniesienie kontroli z jednego agenta do drugiego. Mechanicznie: wywołanie narzędzia, które zwraca nowego agenta, lub krawędź grafu, która podąża za warunkiem.
3. **Współdzielony stan** — dowolna struktura danych, którą więcej niż jeden agent może czytać (czasami zapisywać). Pula komunikatów, blackboard, magazyn klucz-wartość, pamięć wektorowa.
4. **Orkiestrator** — osoba decydująca, kto mówi następny. Opcje: jawny graf (deterministyczny), selektor mówcy LLM (miękki), wywołanie przekazania ostatniego mówcy (OpenAI Swarm) lub scheduler nad kolejką (architektura roju).

To jest cała przestrzeń projektowa. Każdy framework wybiera domyślne wartości dla każdej osi; reszta to składnia powierzchniowa.

### Jak mapuje się każdy framework z 2026 roku

| Framework | Agent | Przekazanie | Współdzielony stan | Orkiestrator |
|-----------|-------|-------------|-------------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | narzędzie zwraca Agent | problem wywołującego | następne wywołanie przekazania LLM |
| AutoGen v0.4 / AG2 | `ConversableAgent` | selektor mówcy na GroupChat | pula komunikatów | funkcja selektora (LLM lub round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Wyniki zadań połączone w łańcuch | menedżer LLM lub statyczna kolejność |
| LangGraph | funkcja węzła | krawędź grafu + warunek | reducer `StateGraph` | graf, deterministyczny |
| Microsoft Agent Framework | agent + wzorce orkiestracji | specyficzne dla wzorca | wątek / kontekst | specyficzny dla wzorca |
| Google ADK | agent + karta A2A | zadanie A2A | artefakty A2A | host decyduje |

Różnice powierzchniowe wyglądają ogromnie. Pod spodem: te same cztery pokrętła.

### Dlaczego to ma znaczenie

Gdy już zobaczysz prymitywy, porównywanie frameworków staje się krótką listą kontrolną:

- Czy orkiestrator ufa LLM w kwestii routingu (Swarm), czy przypina routing w kodzie (LangGraph)?
- Czy współdzielony stan to pełna historia (GroupChat) czy rzutowana (reducer StateGraph)?
- Czy agenci mogą modyfikować swoje nawzajem prompty (menedżer CrewAI) czy tylko przekazywać (Swarm)?

Te trzy pytania odpowiadają na 80% tego, który framework pasuje do danego problemu. Przestajesz szukać „najlepszego frameworka wieloagentowego" i zaczynasz projektować dla osi, która faktycznie ma dla ciebie znaczenie.

### Bezstanowy wgląd

Każdy prymityw oprócz współdzielonego stanu jest bezstanowy. Agent to funkcja (prompt, narzędzia). Przekazanie to wywołanie funkcji. Orkiestrator to scheduler. **Jedyną stanową rzeczą w systemie jest współdzielony stan.** To tam mieszkają wszystkie interesujące błędy: zatrucie pamięci (Lekcja 15), kolejność wiadomości, wersjonowanie, rywalizacja o zapis.

Frameworki, które ukrywają współdzielony stan (Swarm), przesuwają problem na wywołującego. Frameworki, które go centralizują (punkt kontrolny LangGraph, pula AutoGen), czynią go inspekcjonowalnym, ale przenoszą koszt koordynacji na implementację współdzielonego stanu.

### Anatomia pojedynczego prymitywu

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

Bez pamięci. Bez stanu. Dwóch agentów z tym samym promptem systemowym i narzędziami jest wymiennych. Wszystko, co wygląda jak stan per-agent, jest tak naprawdę we współdzielonym stanie lub protokole przekazania.

#### Przekazanie

```
Przekazanie = (from_agent, to_agent, reason, payload)
```

Dominują trzy implementacje:

- **Zwrot funkcji** — narzędzie zwraca następnego agenta. To wzór OpenAI Swarm. Agenci noszą routing w swoich schematach narzędzi.
- **Krawędź grafu** — LangGraph. Krawędzie są deklaratywne. LLM produkuje wartość; warunek wybiera następny węzeł.
- **Wybór mówcy** — AutoGen GroupChat. Funkcja selektora (czasami samo wywołanie LLM) czyta pulę i wybiera, kto mówi następny.

#### Współdzielony stan

```
WspółdzielonyStan = { messages: [], artifacts: {}, context: {} }
}

Minimum: lista wiadomości. Często więcej: strukturalne artefakty (wyniki zadań CrewAI), typowany kontekst (reducery LangGraph), pamięć zewnętrzna (MCP, baza wektorowa).

Dwie topologie: **pełna pula** (każdy agent widzi każdą wiadomość) i **rzutowana** (agenci widzą widok ograniczony rolą). Pełne pule są proste i źle się skalują. Rzutowane pule się skalują, ale wymagają wcześniejszego projektowania schematu.

#### Orkiestrator

```
Orkiestrator = ({state, last_speaker}) -> next_agent
```

Cztery warianty:

- **Statyczny** — graf jest ustalony w czasie budowania (LangGraph deterministyczny, CrewAI Sequential).
- **Wybrany przez LLM** — LLM czyta pulę i wybiera następnego mówcę (AutoGen, CrewAI Hierarchical).
- **Sterowany przekazaniem** — bieżący agent decyduje, wywołując narzędzie przekazania (Swarm).
- **Sterowany kolejką** — pracownicy pobierają ze współdzielonej kolejki; bez jawnego następnego mówcy (architektury roju, Matrix).

### Co zmienia się między frameworkami

Gdy prymitywy są ustalone, pozostałe decyzje projektowe to:

- **Strategia pamięci** — efemeryczne vs trwałe punktowanie kontrolne (checkpointer LangGraph).
- **Granica bezpieczeństwa** — kto może zatwierdzić przekazanie (human-in-the-loop).
- **Rozliczanie kosztów** — budżety tokenów per agent.
- **Obserwowalność** — śledzenie przekazań, utrwalanie stanu do odtworzenia.

Wszystkie implementowalne na bazie prymitywów. Żadna z nich nie jest nowym prymitywem.

## Build It

`code/main.py` implementuje cztery prymitywy w ~150 liniach stdlibowego Pythona. Bez prawdziwego LLM — każdy agent to skryptowana polityka, aby skupić się na strukturze koordynacji.

Plik eksportuje:

- `Agent` — dataclass z nazwą, promptem systemowym, narzędziami, funkcją polityki.
- `Handoff` — funkcja zwracająca nowego agenta.
- `SharedState` — bezpieczna wątkowo pula komunikatów.
- `Orchestrator` — trzy warianty: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (symulowany).

Demo uruchamia ten sam trzyagentowy potok (badania → pisz → recenzuj) przez wszystkie trzy typy orkiestratorów i wypisuje pulę komunikatów na końcu. Możesz zobaczyć, że wyniki różnią się tylko w *kto wybiera następny*; agenci i współdzielony stan są identyczne we wszystkich uruchomieniach.

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyjście: trzy uruchomienia orkiestratora, jeden na wzór. Każdy wypisuje końcową pulę komunikatów. Uruchomienie sterowane przekazaniem dociera do mniejszej liczby agentów, jeśli badacz zdecyduje, że skończył wcześniej — to kompromis routingu LLM w miniaturze.

## Use It

`outputs/skill-primitive-mapper.md` to umiejętność, która czyta dowolną wieloagentową bazę kodu lub dokumentację frameworka i zwraca mapowanie czterech prymitywów. Uruchom ją na nowym wydaniu frameworka, aby zyskać jednoakapitowe zrozumienie przed dogłębnym czytaniem dokumentacji.

## Ship It

Przed przyjęciem nowego frameworka, napisz dla niego mapowanie prymitywów. Jeśli nie możesz, dokumentacja jest niekompletna lub framework wynajduje piąty prymityw (rzadko — sprawdź, czy istnieje wariant współdzielonego stanu, którego nie widziałeś).

Przypnij mapowanie w swoim dokumencie architektury. Gdy nowy członek zespołu dołącza, wyślij mu mapowanie przed dokumentacją API. Gdy wersje frameworka się zmieniają, porównaj mapowanie, a nie dziennik zmian.

## Exercises

1. Uruchom `code/main.py` trzy razy z różnymi politykami agentów. Zaobserwuj, jak wybór orkiestratora zmienia, którzy agenci są uruchamiani.
2. Zaimplementuj czwarty typ orkiestratora: sterowany kolejką, w którym agenci odpytywają współdzielony stan o pracę. Jaki zakleszczenie może wystąpić i jak go wykryć?
3. Weź LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents) i przepisz go jako cztery prymitywy. Które z abstrakcji LangGraph mapują 1:1, a które są wygodnymi opakowaniami?
4. Przeczytaj OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Zidentyfikuj, który z czterech prymitywów Swarm czyni najbardziej ergonomicznym, a który przesuwa na wywołującego.
5. Znajdź jeden framework w tej tabeli, który całkowicie ukrywa współdzielony stan. Wyjaśnij, co się psuje, gdy agenci muszą koordynować między przekazaniami bez ponownego czytania historii.

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Agent | „LLM z narzędziami" | Trójka `(system_prompt, tools, model)`. Bezstanowy. |
| Przekazanie | „Przeniesienie kontroli" | Strukturalne wywołanie, które nazywa następnego agenta i opcjonalny ładunek. Trzy implementacje: zwrot funkcji, krawędź grafu, wybór mówcy. |
| Współdzielony stan | „Pamięć" / „kontekst" | Jedyna stanowa część systemu wieloagentowego. Pula komunikatów lub blackboard. |
| Orkiestrator | „Koordynator" | Osoba decydująca, kto uruchamia się następny. Graf statyczny, selektor LLM, sterowany przekazaniem lub sterowany kolejką. |
| Prymityw | „Abstrakcja" | Jedna z czterech osi, które parametryzuje każdy framework. Nie cecha frameworka. |
| Pula komunikatów | „Współdzielona historia czatu" | Współdzielony stan z pełną historią. Łatwy w analizie, źle się skaluje. |
| Stan rzutowany | „Widok zakresowy" | Widok specyficzny dla roli we współdzielonym stanie. Skaluje się, wymaga projektowania schematu. |
| Wybór mówcy | „Kto mówi następny" | Wzorzec orkiestratora, w którym funkcja (często LLM) wybiera następnego agenta z grupy. |

## Further Reading

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — najjaśniejsze sformułowanie orkiestracji sterowanej przekazaniem
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + wybór mówcy to referencja dla orkiestracji wybranej przez LLM
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — orkiestracja krawędzi grafu i współdzielony stan oparty na reducerze
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — agenci rola-cel-historia, procesy Sequential / Hierarchical
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — żywa linia AutoGen v0.2 po przeniesieniu v0.4 przez Microsoft do utrzymania