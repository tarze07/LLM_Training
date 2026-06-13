# LangGraph — Maszyny stanowe dla agentów

> Pętla ReAct napisana ręcznie to `while True`. Pętla ReAct napisana w LangGraph to graf, który możesz punktować kontrolnie, przerywać, rozgałęziać i podróżować w czasie. Agent się nie zmienił. Zmieniło się oprzyrządowanie wokół niego.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## Problem

Wdrażasz agenta z wywołaniami funkcji. Działa przez trzy tury, a potem coś idzie nie tak: model próbuje użyć narzędzia, które zwraca 500, użytkownik zmienia zdanie w trakcie zadania, albo agent decyduje się zwrócić pieniądze za zamówienie bez zgody człowieka. Pętla `while True:` nie ma punktów zaczepienia. Nie możesz jej wstrzymać, nie możesz jej cofnąć i nie możesz rozgałęzić się na "co by było, gdyby model wybrał inne narzędzie." W momencie, gdy wyślesz to poza demo, agent staje się czarną skrzynką, która albo działała, albo nie.

Następny krok jest oczywisty, gdy go zobaczysz. Agent jest już maszyną stanową — prompt systemowy plus historia wiadomości plus oczekujące wywołania narzędzi plus następna akcja. Uczyń maszynę stanową jawną: węzły dla "model myśli," "narzędzie działa," "człowiek zatwierdza" i krawędzie dla warunkowych przejść między nimi. Gdy graf jest jawny, oprzyrządowanie dostaje cztery rzeczy za darmo: punktowanie kontrolne (zapisz stan między krokami), przerwania (pauza dla człowieka), strumieniowanie (strumieniuj tokeny i zdarzenia pośrednie) i podróż w czasie (cofnij do poprzedniego stanu i wypróbuj inną gałąź).

LangGraph to biblioteka, która dostarcza tę abstrakcję. Nie jest to framework agentowy w sensie LangChain ("oto AgentExecutor, powodzenia"). Jest to środowisko uruchomieniowe grafów z pełnoprawnym stanem, pełnoprawną trwałością i pełnoprawnymi przerwaniami. Pętla agenta to coś, co rysujesz, a nie coś, co piszesz ręcznie.

## Koncepcja

![StateGraph LangGraph: węzły, krawędzie i checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` ma trzy rzeczy.

1. **Stan.** Typowany słownik (TypedDict lub model Pydantic), który przepływa przez graf. Każdy węzeł otrzymuje pełny stan i zwraca częściową aktualizację, którą LangGraph scala za pomocą *reduktora* na pole — `operator.add` dla list, które powinny się kumulować, nadpisywanie domyślnie.
2. **Węzły.** Funkcje Pythona `stan -> częściowy_stan`. Każdy jest dyskretnym krokiem: "wywołaj model," "uruchom narzędzia," "podsumuj."
3. **Krawędzie.** Przejścia między węzłami. Statyczne krawędzie prowadzą w jedno miejsce. Warunkowe krawędzie przyjmują funkcję routera `stan -> nazwa_następnego_węzła`, aby graf mógł się rozgałęziać na podstawie wyjścia modelu.

Kompilujesz graf. Kompilacja wiąże topologię, dołącza checkpointer (opcjonalny, ale niezbędny w produkcji) i zwraca obiekt uruchamialny. Wywołujesz go z początkowym stanem i `thread_id`. Każdy krok wykonania utrwala punkt kontrolny kluczowany przez `(thread_id, checkpoint_id)`.

### Cztery supermoce

**Punktowanie kontrolne.** Każde przejście węzła zapisuje nowy stan w magazynie (w pamięci dla testów, Postgres/Redis/SQLite dla produkcji). Wznów, wywołując graf ponownie z tym samym `thread_id`. Graf kontynuuje od miejsca, w którym się zatrzymał.

**Przerwania.** Oznacz węzeł za pomocą `interrupt_before=["human_review"]`, a wykonanie zatrzymuje się przed uruchomieniem tego węzła. Stan jest utrwalany. Twoje API odpowiada użytkownikowi komunikatem "oczekiwanie na zatwierdzenie." Późniejsze żądanie do tego samego `thread_id` z `Command(resume=...)` wznawia wykonanie.

**Strumieniowanie.** `graph.stream(state, mode="updates")` zwraca delty stanu w miarę ich występowania. `mode="messages"` strumieniuje tokeny LLM wewnątrz węzłów modelu. `mode="values"` zwraca pełne migawki. Wybierasz, co pokazać w swoim interfejsie.

**Podróż w czasie.** `graph.get_state_history(thread_id)` zwraca pełny dziennik punktów kontrolnych. Przekaż dowolny wcześniejszy `checkpoint_id` do `graph.invoke`, a rozgałęzisz się od tego punktu. Świetne do debugowania ("co by było, gdyby model wybrał narzędzie B?") i do testów regresyjnych odtwarzających ślady produkcyjne.

### Reduktory są sednem

Każde pole stanu ma reduktor. Większość domyślnych jest w porządku — nowa wartość nadpisuje starą. Ale listy wiadomości potrzebują `operator.add`, aby nowe wiadomości się dodawały, zamiast zastępować. Równoległe krawędzie łączą swoje aktualizacje przez reduktor. Jeśli dwa węzły aktualizują `messages` i zapomniałeś o `Annotated[list, add_messages]`, drugi wygrywa po cichu i tracisz połowę tury. Reduktor to jedyna subtelna rzecz w bibliotece; zrób to dobrze, a reszta się składa.

### Graf ReAct w czterech węzłach

Produkcyjny agent ReAct to cztery węzły i dwie krawędzie:

1. `agent` — wywołuje LLM z bieżącą historią wiadomości. Zwraca wiadomość asystenta (która może zawierać tool_calls).
2. `tools` — wykonuje wszystkie tool_calls w ostatniej wiadomości asystenta, dodaje wyniki narzędzi jako wiadomości narzędziowe.
3. Warunkowa krawędź z `agent`, która kieruje do `tools`, jeśli ostatnia wiadomość ma tool_calls, w przeciwnym razie do `END`.
4. Statyczna krawędź z `tools` z powrotem do `agent`.

To wszystko. Otrzymujesz pełną pętlę ReAct (Myśl → Akcja → Obserwacja → Myśl → …) z punktowaniem kontrolnym, przerwaniami i strumieniowaniem, w około 40 liniach kodu.

### StateGraph vs Send (fanout)

`Send(nazwa_węzła, stan)` pozwala węzłowi wysyłać równoległe podgrafy. Przykład: agent decyduje się zapytać trzy retrievery jednocześnie. Każde `Send` tworzy równoległe wykonanie docelowego węzła; ich wyniki łączą się przez reduktor stanu. W ten sposób LangGraph wyraża wzorzec orkiestrator-pracownicy bez prymitywów wątkowych.

### Podgrafy

Skompilowany graf może być węzłem w innym grafie. Zewnętrzny graf widzi pojedynczy węzeł; wewnętrzny graf ma własny stan i własne punkty kontrolne. W ten sposób zespoły budują agentów nadzorca-pracownik: graf nadzorcy kieruje intencję użytkownika do podgrafu pracownika właściwego dla domeny.

## Zbuduj to

### Krok 1: stan i węzły

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` to reduktor, który sprawia, że lista wiadomości się kumuluje, zamiast nadpisywać. Zapomnienie o nim to najczęstszy błąd w LangGraph.

### Krok 2: uruchom z wątkiem

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Każda aktualizacja to słownik `{nazwa_węzła: delta_stanu}`. Twój frontend może strumieniować je do interfejsu, aby użytkownicy widzieli "agent myśli… wywołuje search_web… otrzymał wynik… odpowiada."

### Krok 3: dodaj przerwanie z człowiekiem w pętli

Oznacz węzeł, aby wykonanie wstrzymało się przed jego uruchomieniem.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pauza przed każdym wywołaniem narzędzia
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] jest ustawione. Sprawdź proponowane wywołania narzędzi.
# Jeśli zatwierdzone:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# Jeśli odrzucone: napisz wiadomość odrzucenia i wznów
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Stan, punkt kontrolny i wątek są utrwalane przez cały czas trwania przerwania. Nic nie jest w pamięci poza czasem wykonania.

### Krok 4: podróż w czasie do debugowania

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Rozgałęź się od poprzedniego punktu kontrolnego
target = history[3].config  # trzy kroki wstecz
for event in app.stream(None, target, stream_mode="values"):
    pass  # odtwórz od tego punktu do przodu
```

Przekazanie `None` jako wejścia odtwarza od danego punktu kontrolnego; przekazanie wartości dodaje ją jako aktualizację do stanu tego punktu kontrolnego przed wznowieniem. W ten sposób odtwarzasz złe wykonanie agenta bez ponownego uruchamiania całej rozmowy.

### Krok 5: zamień checkpointer na produkcyjny

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis i Postgres są dostarczane. `MemorySaver` jest do testów. Wszystko, co ma przetrwać ponowne uruchomienie, potrzebuje prawdziwego magazynu.

## Umiejętność

> Buduj agentów jako grafy, a nie jako pętle `while True`.

Zanim sięgniesz po LangGraph, wykonaj 60-sekundowy projekt:

1. **Nazwij węzły.** Każda dyskretna decyzja lub akcja z efektami ubocznymi to węzeł. "Agent myśli," "narzędzie działa," "recenzent zatwierdza," "odpowiedź jest strumieniowana." Jeśli nie możesz ich wymienić, zadanie nie ma jeszcze kształtu agenta.
2. **Zadeklaruj stan.** Minimalny TypedDict z reduktorem dla każdego pola listy. Nie pakuj wszystkiego do `messages`; wynieś pola specyficzne dla zadania (roboczy `plan`, licznik `budget`, listę `retrieved_docs`) na najwyższy poziom.
3. **Narysuj krawędzie.** Statyczne, chyba że następny krok zależy od wyjścia modelu. Każda warunkowa krawędź potrzebuje funkcji routera z nazwanymi gałęziami.
4. **Wybierz checkpointer z góry.** `MemorySaver` do testów, Postgres/Redis/SQLite do wszystkiego innego. Nie wdrażaj bez niego — brak checkpointra oznacza brak wznowienia, brak przerwania, brak podróży w czasie.
5. **Zdecyduj o przerwaniach przed uruchomieniem narzędzi, nie po.** Zatwierdzenia idą na krawędzi do węzła z efektami ubocznymi, abyś mógł anulować przed szkodą; walidacja idzie na krawędzi wyjścia z modelu, abyś mógł tanio odrzucić złe wywołania.
6. **Strumieniuj domyślnie.** `mode="updates"` dla interfejsu, `mode="messages"` dla strumieniowania tokenów wewnątrz węzłów modelu, `mode="values"` dla pełnych migawek podczas ewaluacji.

Odmów wdrożenia agenta LangGraph bez checkpointra. Odmów wdrożenia takiego, który przerywa *po* efekcie ubocznym. Odmów wdrożenia pola `messages` bez `add_messages` jako jego reduktora.

## Ćwiczenia

1. **Łatwe.** Zaimplementuj czterowęzłowy graf ReAct powyżej z narzędziem kalkulatora i narzędziem wyszukiwania internetowego. Zweryfikuj, że `list(app.get_state_history(config))` zwraca co najmniej cztery punkty kontrolne dla dwuturowej rozmowy.
2. **Średnie.** Dodaj węzeł `planner`, który działa przed `agent` i zapisuje strukturalny `plan: list[str]` w stanie. Niech `agent` oznacza kroki planu jako wykonane. Spraw, aby test failował, jeśli `plan` zostanie utracony po wznowieniu punktu kontrolnego (zły reduktor).
3. **Trudne.** Zbuduj graf nadzorcy, który kieruje między trzema podgrafami (`researcher`, `writer`, `reviewer`) używając `Send`. Każdy podgraf ma własny stan i checkpointer. Dodaj `interrupt_before=["writer"]` na zewnętrznym grafie, aby człowiek mógł zatwierdzić brief badawczy. Potwierdź, że podróż w czasie z poprzedniego punktu kontrolnego uruchamia ponownie tylko rozgałęzioną gałąź.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| StateGraph | "Graf LangGraph" | Obiekt budowniczy, do którego dodajesz węzły i krawędzie przed kompilacją. |
| Reducer | "Jak pole się scala" | Funkcja `(stare, nowe) -> scalone` stosowana, gdy węzeł zwraca aktualizację dla tego pola; domyślnie nadpisanie, `add_messages` dodaje. |
| Thread | "ID rozmowy" | Ciąg `thread_id`, który zakresuje wszystkie punkty kontrolne dla jednej sesji. |
| Checkpoint | "Zatrzymany stan" | Utrwalona migawka pełnego stanu grafu po przejściu węzła, kluczowana przez `(thread_id, checkpoint_id)`. |
| Interrupt | "Pauza dla człowieka" | `interrupt_before` / `interrupt_after` zatrzymuje wykonanie na granicy węzła; wznów za pomocą `Command(resume=...)`. |
| Time-travel | "Rozgałęzienie z poprzedniego kroku" | `graph.invoke(None, config_with_old_checkpoint_id)` odtwarza od tego punktu kontrolnego do przodu. |
| Send | "Równoległe wysłanie podgrafu" | Konstruktor, który węzeł może zwrócić, aby utworzyć N równoległych wykonań docelowego węzła. |
| Subgraph | "Skompilowany graf jako węzeł" | Skompilowany StateGraph używany jako węzeł w innym grafie; zachowuje własny zakres stanu. |

## Dalsze czytanie

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — canonical reference for StateGraph, reducers, checkpointers, and interrupts.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — the mental model this lesson uses, straight from the source.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — the detail on Postgres/SQLite/Redis stores, checkpoint namespaces, and thread IDs.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)`, and the edit-state pattern.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — the pattern every LangGraph agent implements; read it for the reasoning trace rationale.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — which graph shapes (chain, router, orchestrator-workers, evaluator-optimizer) to prefer and when.
- Phase 11 · 09 (Function Calling) — the tool-call primitive every LangGraph agent node reuses.
- Phase 11 · 14 (Model Context Protocol) — external tool discovery that plugs into a LangGraph `ToolNode` via the MCP adapter.
- Phase 11 · 17 (Agent framework tradeoffs) — when to pick LangGraph over CrewAI, AutoGen, or Agno.