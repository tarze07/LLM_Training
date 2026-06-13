# Kompromisy frameworków agentowych — LangGraph vs CrewAI vs AutoGen vs Agno

> Każdy framework sprzedaje to samo demo (agent badawczy tworzy raport) i ukrywa ten sam błąd (schemat stanu walczy z warstwą orkiestracji). Wybierz framework, którego abstrakcje pasują do kształtu Twojego problemu; wszystko inne to klej, który piszesz dwa razy.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## Problem

Masz zadanie, które wymaga więcej niż jednego wywołania LLM. Może to być przepływ pracy badawczej (zaplanuj, przeszukaj, podsumuj, zacytuj). Może to być potok przeglądu kodu (sparsuj diff, skrytykuj, załataj, zatwierdź). Może to być asystent wieloturny, który rezerwuje loty, pisze e-maile i zgłasza wydatki. Wybierasz framework.

Trzy dni później odkrywasz, że abstrakcje frameworka przeciekają. CrewAI daje role, ale walczy z Tobą, gdy "badacz" musi przekazać strukturalny plan "pisarzowi." AutoGen daje czat między agentami, ale nie ma pełnoprawnego stanu, więc Twój punkt kontrolny to pikiel dziennika rozmów. LangGraph daje graf stanu, ale zmusza Cię do nazwania każdego przejścia, zanim dowiesz się, co agent zrobi. Agno daje abstrakcję pojedynczego agenta, która krzyczy, gdy próbujesz rozgałęzić się na trzech równoczesnych pracowników.

Rozwiązaniem nie jest "wybierz najlepszy framework." Chodzi o dopasowanie podstawowej abstrakcji frameworka do kształtu Twojego problemu. Ta lekcja rysuje tę mapę.

## Koncepcja

![Macierz frameworków agentowych: podstawowa abstrakcja vs kształt problemu](../assets/framework-matrix.svg)

Cztery frameworki dominują w krajobrazie 2026. Ich podstawowe abstrakcje nie są takie same.

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents needing time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Role-playing or persona-driven workflows with a short linear/hierarchical plan. | Anything stateful beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pair — two or more agents that speak in turns until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where the thinking emerges from the chat. | Deterministic workflows with a known DAG; anything needing durable state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-build single agents and lightweight teams; strong multi-modality and built-in storage drivers. | Deep, explicitly-branched graphs with custom reducers. |

### Co "abstrakcja" naprawdę oznacza

Podstawowa abstrakcja frameworka to rzecz, którą rysujesz na tablicy, gdy przedstawiasz architekturę.

- **LangGraph** → rysujesz graf. Węzły to kroki, krawędzie to przejścia, a obiekt stanu w każdym punkcie jest typowany. Model mentalny to maszyna stanowa.
- **CrewAI** → rysujesz schemat organizacyjny. Każda rola ma opis stanowiska, a menedżer rozdziela zadania. Model mentalny to mały zespół specjalistów.
- **AutoGen** → rysujesz DM na Slacku. Dwóch agentów wymienia wiadomości; trzeci dołącza, jeśli potrzebujesz moderatora. Model mentalny to czat.
- **Agno** → rysujesz pojedyncze pudełko z narzędziami zwisającymi z niego. Ustaw pudełka obok siebie dla zespołu. Model mentalny to "agent z bateriami w zestawie."

### Kwestia stanu

Stan to miejsce, w którym większość wyborów frameworka załamuje się w produkcji.

- **LangGraph.** Typowany stan (`TypedDict` lub model Pydantic), reduktory na pole, pełnoprawny checkpointer (SQLite/Postgres/Redis). Wznowienie, przerwanie i podróż w czasie są gratis. *(Zobacz Faza 11 · 16.)*
- **CrewAI.** Stan przepływa jako ciągi znaków między zadaniami przez pole `context` lub strukturalnie przez `output_pydantic`. Brak trwałego magazynu dla crew po wyjęciu z pudełka; dokładasz własny, jeśli crew ma przetrwać ponowne uruchomienie.
- **AutoGen.** Stan to historia czatu i dowolny `context` zdefiniowany przez użytkownika. Transkrypty rozmów są trwałe; dowolny stan przepływu pracy nie jest, chyba że napiszesz adaptery.
- **Agno.** Wbudowane sterowniki pamięci (SQLite, Postgres, Mongo, Redis, DynamoDB) dołączone do `Agent` przez `storage=` — sesje rozmów i pamięci użytkownika są trwale zapisywane automatycznie. Nie jest to pełnoprawny checkpointer grafu; to magazyn sesji.

### Kwestia rozgałęziania

Każdy nietrywialny agent się rozgałęzia. Kto decyduje o gałęzi, ma znaczenie.

- **LangGraph** — Ty decydujesz, przez warunkowe krawędzie. Routing to funkcja Pythona z nazwanymi gałęziami. Gałęzie są pełnoprawne w skompilowanym grafie; checkpointer rejestruje, która gałąź została wybrana.
- **CrewAI** — menedżer decyduje w trybie hierarchicznym; w trybie sekwencyjnym decydujesz w czasie budowania. Routing jest niejawny w liście zadań; nie ma pełnoprawnego "if" poza promptem menedżera.
- **AutoGen** — agenci decydują przez czat. Rozgałęzianie wyłania się z tego, kto mówi następny. `GroupChatManager` wybiera następnego mówcę; możesz napisać własną `speaker_selection_method`, ale domyślnie jest sterowana przez LLM.
- **Agno** — agent decyduje przez to, które narzędzie wywołać dalej. Zespoły mają tryb koordynatora/routera/współpracy; rozgałęzianie poza tym jest odpowiedzialnością programisty.

### Kwestia obserwowalności

- **LangGraph** — OpenTelemetry przez LangSmith lub dowolny eksporter OTel. Każde przejście węzła to span śledzenia; punkty kontrolne służą również jako odtwarzalne ślady. LangSmith jest opcją pierwszej strony; Langfuse/Phoenix również mają adaptery.
- **CrewAI** — pełnoprawne OpenTelemetry od końca 2025; integracje z Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — Integracja OpenTelemetry przez `autogen-core`; AgentOps i Opik mają łączniki. Ziarnistość śledzenia to na wiadomość agenta, nie na węzeł.
- **Agno** — wbudowana flaga `monitoring=True` plus eksportery OpenTelemetry; ścisła integracja z Langfuse dla śladów sesji.

### Koszt i opóźnienie

Wszystkie cztery frameworki dodają narzut na wywołanie (logika frameworka, walidacja, serializacja). Orientacyjna kolejność rosnącego narzutu: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Różnica jest zdominowana przez to, ile dodatkowego routingu LLM framework wykonuje. Hierarchiczny menedżer CrewAI wydaje tokeny na decydowanie, kto działa dalej; `GroupChatManager` AutoGen podobnie. LangGraph wydaje tokeny tylko tam, gdzie piszesz `llm.invoke`. Ścieżka pojedynczego agenta Agno jest cienka.

Gdy koszt na uruchomienie ma znaczenie, preferuj jawny routing (krawędzie LangGraph, `speaker_selection_method` AutoGen) nad routingiem wybranym przez LLM.

### Interoperacyjność

- **LangGraph** ↔ narzędzia, retrievery, LLM LangChain. Pełnoprawny adapter MCP (narzędzia importowane jako serwery MCP).
- **CrewAI** ↔ narzędzia dziedziczą z `BaseTool`; narzędzia LangChain, LlamaIndex i MCP wszystkie się adaptują. Delegacja crew-do-crew przez `allow_delegation=True`.
- **AutoGen** → `FunctionTool` opakowuje dowolny wywoływalny obiekt Pythona; dostępny adapter MCP. Ścisłe powiązanie z ekosystemem AG2 dla wzorców agent-agent.
- **Agno** → dekorator `@tool` lub podklasa `BaseTool`; adapter MCP; narzędzia mogą być współdzielone między agentami i zespołami.

## Umiejętność

> Potrafisz wyjaśnić w jednym zdaniu, dlaczego dany framework jest odpowiedni dla danego problemu agentowego.

Lista kontrolna przed budową:

1. **Narysuj kształt.** Czy to graf (typizowany stan, nazwane przejścia)? Gra ról (specjaliści przekazują sobie pracę)? Czat (agenci rozmawiają, aż skończą)? Pojedynczy agent z narzędziami?
2. **Zdecyduj, kto rozgałęzia.** Rozgałęzianie decydowane przez programistę → LangGraph. Decydowane przez menedżera-agent → CrewAI hierarchiczne. Wyłaniające się z czatu → AutoGen. Decydowane przez wywołanie narzędzia → Agno.
3. **Sprawdź budżet stanu.** Czy potrzebujesz wznowienia z punktu kontrolnego? Podróży w czasie? Przerwań człowieka w trakcie działania? Jeśli tak, LangGraph jest domyślnym wyborem; sesje Agno pokrywają stan w zakresie rozmowy.
4. **Sprawdź budżet kosztów.** Routing wybrany przez LLM kosztuje dodatkowe tokeny na turę. Jeśli agent działa tysiące razy dziennie, preferuj jawny routing.
5. **Zaplanuj narzut frameworka.** Każdy framework to kolejna zależność. Jeśli zadanie to dwa wywołania LLM i narzędzie, napisz 30 linii czystego Pythona; żaden framework nie jest tańszy niż brak frameworka.

Odmów sięgnięcia po framework, zanim będziesz mógł narysować graf, schemat organizacyjny, czat lub pudełko agenta. Odmów wybrania takiego, który zmusza Cię do walki z jego modelem stanu dla rzeczy, której faktycznie potrzebujesz.

## Macierz decyzyjna

| Kształt problemu | Preferowany framework | Dlaczego |
|---------------|---------------------|-----|
| DAG przepływu pracy z typowanym stanem, zatwierdzeniami człowieka, długotrwały | LangGraph | Pełnoprawny stan, checkpointer, przerwania, podróż w czasie. |
| Potok badawczy/pisarski z wyraźnymi rolami | CrewAI (sekwencyjny) lub podgrafy LangGraph | Rola-na-zadanie jest tania do wyrażenia w CrewAI; skaluj do LangGraph, gdy rozgałęzianie staje się złożone. |
| Dialog proponujący-krytyk lub nauczyciel-uczeń | AutoGen | Czat dwuagentowy to jego natywny kształt. |
| Pojedynczy agent z narzędziami, sesjami, pamięcią | Agno | Najcieńsze ustawienie, wbudowane przechowywanie i pamięć. |
| Tysiące równoległych rozgałęzień z reduktorami | LangGraph + `Send` | Jedyny z pełnoprawnym API równoległego wysyłania. |
| Szybki prototyp, bez zobowiązań do frameworka | Czysty Python + SDK dostawcy | Żaden framework nie jest najszybszym frameworkiem. |

## Ćwiczenia

1. **Łatwe.** Weź to samo zadanie — "zbadaj siedzibę główną Anthropica, napisz 200-słowne streszczenie, zacytuj źródła" — i zaimplementuj je w LangGraph (cztery węzły: planuj, szukaj, pisz, cytuj) oraz w CrewAI (trzy role: badacz, pisarz, redaktor). Raportuj koszt tokenów na uruchomienie i liczbę linii kodu.
2. **Średnie.** Zbuduj to samo zadanie w AutoGen (czat badacz ↔ pisarz, redaktor dołącza przez `GroupChat`) i Agno (pojedynczy agent z `search_tools` i `write_tools`, plus magazyn sesji). Uszereguj cztery implementacje pod względem (a) kosztu na uruchomienie, (b) zdolności do wznowienia po awarii, (c) zdolności do wstrzyknięcia zatwierdzenia człowieka przed krokiem pisania.
3. **Trudne.** Zbuduj skrypt drzewa decyzyjnego `pick_framework.py`, który przyjmuje krótki opis problemu (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) i zwraca rekomendację z jednotorowym uzasadnieniem. Zweryfikuj na sześciu przypadkach, które sam zaprojektujesz.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Orchestration | "Jak agenci koordynują" | Warstwa, która decyduje, który węzeł/rola/agent działa dalej. |
| Durable state | "Wznowienie po restarcie" | Stan, który przetrwa śmierć procesu, dołączony do punktu kontrolnego lub magazynu sesji. |
| LLM-selected routing | "Niech model zdecyduje" | Planujący LLM wybiera następny krok w każdej turze; elastyczny, ale płaci tokenami przy każdej decyzji. |
| Explicit routing | "Programista decyduje" | Funkcja Pythona lub statyczna krawędź wybiera następny krok; tani i możliwy do audytu. |
| Crew | "Zespół CrewAI" | Role + zadania + proces (sekwencyjny lub hierarchiczny) związane w jeden wykonywalny obiekt. |
| GroupChat | "Czat wieloagentowy AutoGen" | Zarządzana rozmowa między N agentami z selektorem mówcy. |
| Team (Agno) | "Wieloagentowe Agno" | Tryb route / coordinate / collaborate na zestawie agentów. |
| StateGraph | "Graf LangGraph" | Abstrakcja typowanego stanu, węzła, warunkowej krawędzi, checkpointra. |

## Dalsze czytanie

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, interrupts, time-travel.
- [CrewAI documentation](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes.
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools.
- [Agno documentation](https://docs.agno.com/) — Agent, Team, Workflow, storage, memory.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — pattern library (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) framework-agnostic.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — the loop every framework dresses up.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen's design paper.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — role-play foundation that CrewAI-style persona stacks build on.
- Phase 11 · 16 (LangGraph) — the framework this lesson benchmarks against.
- Phase 11 · 19 (Reflexion) — a pattern that maps cleanly to LangGraph but awkwardly to CrewAI.
- Phase 11 · 22 (Production observability) — how to instrument whichever framework you pick.