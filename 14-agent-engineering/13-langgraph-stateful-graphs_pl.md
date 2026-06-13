# LangGraph: Grafy stanowe i trwałe wykonywanie

> LangGraph jest referencją 2026 dla niskopoziomowej orkiestracji stanowej. Agent to maszyna stanów; węzły to funkcje; krawędzie to przejścia; stan jest niezmienny i punktowany kontrolnie po każdym kroku. Wznów z dowolnej awarii dokładnie tam, gdzie się zatrzymałeś.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Opisz podstawowy model LangGraph: maszyna stanów z niezmiennym stanem, węzłami funkcyjnymi, warunkowymi krawędziami i punktami kontrolnymi po każdym kroku.
- Wymień cztery zdolności, które dokumentacja podkreśla: trwałe wykonywanie, strumieniowanie, człowiek-w-pętli, kompleksowa pamięć.
- Wyjaśnij trzy topologie orkiestracji obsługiwane przez LangGraph: nadzorca, peer-to-peer (rój), hierarchiczna (zagnieżdżone podgrafy).
- Zaimplementuj w stdlib graf stanowy z niezmiennym stanem, warunkowymi krawędziami i cyklem punkt kontrolny/wznów.

## The Problem

Agenci i przepływy pracy mają wspólny problem: gdy 40-etapowe uruchomienie nie powiedzie się na etapie 38, chcesz wznowić od etapu 38, a nie zaczynać od nowa. Modele stanu drugiej klasy pozostawiają operatorów majstrujących przy ponownych próbach wokół biblioteki, która zakłada świeże uruchomienia.

Odpowiedź projektowa LangGraph: stan jest obiektem pierwszej klasy z typowaniem, mutacje są jawne, a punkty kontrolne utrwalane po każdym węźle. Wznowienie to wywołanie `load_state(session_id)`.

## The Concept

### Graf

Graf jest definiowany przez:

- **Typ stanu.** Typowany słownik (lub model Pydantic), który każdy węzeł czyta i mutuje.
- **Węzły.** Czyste funkcje `(state) -> state_update`. Aktualizacje są scalane ze stanem po powrocie.
- **Krawędzie.** Warunkowe lub bezpośrednie przejścia między węzłami.
- **Wejście i wyjście.** Węzły wartownicze `START` i `END` oznaczają granicę.

Przykład: agent z węzłami `classify`, `refund`, `bug`, `sales`, `done` — przepływ routingu jako graf.

### Trwałe wykonywanie

Po każdym węźle środowisko uruchomieniowe serializuje stan i zapisuje go do punktu kontrolnego (SQLite, Postgres, Redis, niestandardowy). W przypadku awarii na kroku N, środowisko może `resume(session_id)` i kontynuować od kroku N+1 z dokładnym stanem.

Dokumentacja LangGraph wyraźnie podkreśla użytkowników produkcyjnych, dla których ma to znaczenie: Klarna, Uber, J.P. Morgan. Twierdzenie nie dotyczy kształtu grafu; chodzi o to, że kształt grafu plus punktacja kontrolna czynią odtwarzanie tanim.

### Strumieniowanie

Każdy węzeł może emitować częściowe wyniki. Graf przesyła zdarzenia per-węzeł-delta do wywołującego, aby interfejsy użytkownika aktualizowały się w miarę działania grafu.

### Człowiek-w-pętli

Sprawdzaj i modyfikuj stan między węzłami. Implementacje: wstrzymaj przed krytycznym węzłem, pokaż stan człowiekowi, przyjmij modyfikacje, wznów. Punkt kontrolny ułatwia to, ponieważ stan jest już serializowany.

### Pamięć

Krótkoterminowa (w ramach jednego uruchomienia — historia konwersacji w stanie) i długoterminowa (między uruchomieniami — trwała przez punkt kontrolny plus oddzielny magazyn długoterminowy). LangGraph integruje się z zewnętrznymi systemami pamięci (Mem0, niestandardowe) przez narzędzia.

### Trzy topologie

1. **Nadzorca.** Centralny router LLM wysyła do wyspecjalizowanych podagentów. `create_supervisor()` w `langgraph-supervisor` (choć zespół LangChain w 2026 zaleca robienie tego przez bezpośrednie wywołania narzędzi dla większej kontroli kontekstu).
2. **Rój / peer-to-peer.** Agenci przekazują sobie bezpośrednio przez współdzieloną powierzchnię narzędzi. Bez centralnego routera.
3. **Hierarchiczna.** Nadzorcy zarządzający pod-nadzorcami, zaimplementowani jako zagnieżdżone podgrafy.

### Gdzie ten wzór zawodzi

- **Zbyt małe punkty kontrolne.** Punktowanie tylko tur konwersacji pozostawia stan narzędzi i zapisy pamięci nie do odzyskania. Pełny stan musi być serializowany.
- **Niedeterministyczne węzły.** Wznowienie zakłada, że wejścia węzła produkują tę samą aktualizację stanu. Ziarna losowe, czas zegarowy, zewnętrzne API muszą być przechwycone.
- **Nadmierne użycie warunkowych krawędzi.** Graf, w którym każda krawędź jest warunkowa, to maszyna stanów, o której nie można wnioskować. Preferuj liniowe łańcuchy z okazjonalnymi rozgałęzieniami.

## Build It

`code/main.py` implementuje w stdlib graf stanowy:

- `State` — typowany słownik z `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — wywoływalny, przyjmujący stan i zwracający słownik aktualizacji.
- `StateGraph` — węzły + krawędzie + warunkowe krawędzie + uruchom + wznów.
- `SQLiteCheckpointer` (atrapa w pamięci) — serializuje stan po każdym węźle; `load(session_id)` przywraca.
- Graf demonstracyjny: klasyfikuj -> rozgałęź (zwrot / błąd / sprzedaż) -> bramka ludzka -> wyślij.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje pierwsze uruchomienie nieudane przy bramce ludzkiej, utrwalenie, a następnie wznowienie produkujące końcowe wyjście.

## Use It

- **LangGraph** — referencja, gotowa do produkcji. Użyj `create_react_agent`, `create_supervisor` lub zbuduj własny graf.
- **AutoGen v0.4** (Lekcja 14) — alternatywa modelu aktora dla scenariuszy o wysokiej współbieżności.
- **Claude Agent SDK** (Lekcja 17) — zarządzana uprząż z wbudowanym magazynem sesji.
- **Niestandardowe** — gdy potrzebujesz dokładnej kontroli nad kształtem stanu lub backendem punktu kontrolnego.

## Ship It

`outputs/skill-state-graph.md` generuje graf stanowy w kształcie LangGraph w dowolnym środowisku docelowym z wbudowanym punktowaniem kontrolnym i wznawianiem.

## Exercises

1. Dodaj warunkową krawędź od `classify` do `end`, gdy ufność klasyfikacji jest poniżej progu. Wznów uruchomienie po ręcznym ustawieniu `route` przez człowieka.
2. Zamień atrapę podobną do SQLite na prawdziwy punkt kontrolny SQLite. Zmierz narzut serializacji na krok.
3. Zaimplementuj równoległe krawędzie: dwa węzły uruchomione współbieżnie, scalone przez niestandardowy reduktor. Co daje tu niezmienny stan?
4. Przeczytaj dokumentację `langgraph-supervisor`. Przenieś zabawkę na `create_supervisor`. Porównaj kształty śladów.
5. Dodaj strumieniowanie: każdy węzeł emituje częściowy stan podczas działania. Wypisz delty w miarę ich pojawiania się.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Graf stanowy | "Agent jako maszyna stanów" | Typowany stan + węzły + krawędzie + reduktory |
| Punkt kontrolny | "Backend trwałości" | Serializuje stan po każdym węźle; umożliwia wznowienie |
| Reduktor | "Scalacz stanu" | Funkcja łącząca bieżący stan z aktualizacją węzła |
| Warunkowa krawędź | "Rozgałęzienie" | Krawędź wybrana przez funkcję stanu |
| Podgraf | "Zagnieżdżony graf" | Graf używany jako węzeł wewnątrz innego grafu |
| Trwałe wykonywanie | "Wznów z awarii" | Restart przy ostatnim udanym węźle z dokładnym stanem |
| Nadzorca | "Router LLM" | Centralny dyspozytor dla wyspecjalizowanych podagentów |
| Rój | "Agenci P2P" | Agenci przekazują przez współdzielone narzędzia; bez centralnego routera |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — dokumentacja referencyjna
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — API wzorca nadzorcy
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — alternatywa modelu aktora
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — magazyn sesji i podagenci